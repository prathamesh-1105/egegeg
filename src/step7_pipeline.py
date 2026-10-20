import os
import sys
sys.path.insert(0, os.path.abspath("."))
import glob
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import mne
from scipy import signal, integrate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.load_eeg import load_eeg_raw
from src.events import get_eeg_events

LABEL_MAPPING = {
    "T0": "REST",
    "T1": "LEFT_FIST",
    "T2": "RIGHT_FIST"
}

EXPECTED_CLASSES = ["REST", "LEFT_FIST", "RIGHT_FIST"]

BASELINE_FEATURES = [
    "mean", "variance", "std", "rms", "min", "max", "peak_to_peak", "energy",
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"
]

EXTENDED_FEATURES = BASELINE_FEATURES + [
    "total_power",
    "alpha_relative_power",
    "beta_relative_power",
    "theta_relative_power",
    "mu_power",
    "mu_relative_power",
    "alpha_beta_ratio"
]

# Cache raw EDF data in memory
EDF_CACHE = {}


def save_baseline_info(models_dir: str = "models") -> dict:
    baseline_path = os.path.join(models_dir, "baseline_results.json")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    baseline_data = {
        "baseline_model": "Random Forest",
        "baseline_accuracy": 0.5250596658711217,
        "baseline_test_subjects": ["S009", "S010"],
        "baseline_feature_count": 13,
        "baseline_features": BASELINE_FEATURES,
        "training_subjects": ["S001", "S002", "S003", "S004", "S005", "S006", "S007", "S008"],
        "note": "Original Step 5 baseline preserved intact."
    }

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                baseline_data["rf_metrics"] = meta.get("rf_metrics", {})
                baseline_data["svm_metrics"] = meta.get("svm_metrics", {})
        except Exception:
            pass

    os.makedirs(models_dir, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline_data, f, indent=2)

    return baseline_data


def discover_available_subjects(data_dir: str = "data") -> list:
    found_subjects = set()
    search_dirs = [data_dir, os.path.join(data_dir, "raw")]
    for d in search_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".edf") and f.startswith("S") and len(f) >= 7:
                        sub_id = f[:4]
                        if sub_id[1:].isdigit():
                            found_subjects.add(sub_id)

    sorted_subs = sorted(list(found_subjects))
    return sorted_subs


def find_edf_files_for_subjects(subjects: list, data_dir: str = "data") -> list:
    """
    Finds absolute paths to EDF files for specified subjects.
    Ensures only ONE file per (subject, run) combination is returned to prevent duplicate reading.
    """
    search_dirs = [data_dir, os.path.join(data_dir, "raw")]
    seen_runs = {}

    for sub in subjects:
        for d in search_dirs:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for f in files:
                        if f.endswith(".edf") and f.startswith(sub):
                            filename = os.path.basename(f)
                            sub_id = filename[:4]
                            run_id = filename[4:7] if len(filename) >= 7 else "R03"
                            key = (sub_id, run_id)
                            if key not in seen_runs:
                                seen_runs[key] = os.path.abspath(os.path.join(root, f))

    matched_files = sorted(list(seen_runs.values()))
    return matched_files


def get_channel_indices(raw_ch_names: list, group_type: str = "Group A") -> list:
    group_a_targets = ["C3", "CZ", "C4"]
    group_b_targets = [
        "FC3", "FC1", "FCZ", "FC2", "FC4",
        "C3", "C1", "CZ", "C2", "C4",
        "CP3", "CP1", "CPZ", "CP2", "CP4"
    ]

    clean_map = {}
    for idx, ch in enumerate(raw_ch_names):
        cleaned = ch.strip().rstrip('.').upper()
        clean_map[cleaned] = idx

    if group_type == "Group A":
        matched_indices = [clean_map[target] for target in group_a_targets if target in clean_map]
        if not matched_indices:
            for target in group_a_targets:
                for cleaned_ch, idx in clean_map.items():
                    if target in cleaned_ch:
                        matched_indices.append(idx)
                        break
        return matched_indices if matched_indices else [0]

    elif group_type == "Group B":
        matched_indices = [clean_map[target] for target in group_b_targets if target in clean_map]
        if not matched_indices:
            return get_channel_indices(raw_ch_names, "Group A")
        return matched_indices

    else:  # Group C - All channels
        return list(range(len(raw_ch_names)))


def load_and_preprocess_cached(edf_path: str, channel_group: str):
    cache_key = (edf_path, channel_group)
    if cache_key in EDF_CACHE:
        return EDF_CACHE[cache_key]

    raw = load_eeg_raw(edf_path)
    df_events = get_eeg_events(raw)
    
    ch_indices = get_channel_indices(raw.ch_names, group_type=channel_group)
    raw_sub = raw.copy().pick(ch_indices)

    # Fast Scipy Bandpass Filter (1-45 Hz) directly on array data
    sfreq = raw_sub.info['sfreq']
    if sfreq > 90.0:
        nyq = 0.5 * sfreq
        b, a = signal.butter(3, [1.0 / nyq, 45.0 / nyq], btype='band')
        raw_sub._data = signal.filtfilt(b, a, raw_sub._data, axis=-1)

    EDF_CACHE[cache_key] = (raw_sub, df_events)
    return raw_sub, df_events


def extract_features_vectorized(sig_data_uv: np.ndarray, sfreq: float, feature_set_type: str = "Extended (20 features)") -> dict:
    if sig_data_uv.ndim == 1:
        sig_data_uv = sig_data_uv.reshape(1, -1)

    n_chans, n_samples = sig_data_uv.shape
    if n_samples == 0:
        sig_data_uv = np.zeros((1, 16))
        n_chans, n_samples = 1, 16

    means = np.mean(sig_data_uv, axis=1)
    vars_ = np.var(sig_data_uv, axis=1)
    stds = np.std(sig_data_uv, axis=1)
    rmss = np.sqrt(np.mean(sig_data_uv ** 2, axis=1))
    mins = np.min(sig_data_uv, axis=1)
    maxs = np.max(sig_data_uv, axis=1)
    ptps = maxs - mins
    energies = np.sum(sig_data_uv ** 2, axis=1)

    nperseg = min(n_samples, int(sfreq * 2))
    if nperseg < 8:
        nperseg = n_samples

    freqs, psd = signal.welch(sig_data_uv, fs=sfreq, nperseg=nperseg, axis=-1)

    bands = {
        "Delta": (0.5, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta": (13.0, 30.0),
        "Gamma": (30.0, 45.0)
    }

    b_powers = {}
    for b_name, (f_low, f_high) in bands.items():
        idx_b = np.logical_and(freqs >= f_low, freqs <= f_high)
        if np.any(idx_b) and np.sum(idx_b) > 1:
            try:
                pow_val = integrate.simpson(psd[:, idx_b], freqs[idx_b], axis=-1)
            except Exception:
                pow_val = np.trapz(psd[:, idx_b], freqs[idx_b], axis=-1)
        else:
            pow_val = np.zeros(n_chans)
        b_powers[b_name] = np.maximum(0.0, pow_val)

    d_pow = b_powers["Delta"]
    t_pow = b_powers["Theta"]
    a_pow = b_powers["Alpha"]
    b_pow = b_powers["Beta"]
    g_pow = b_powers["Gamma"]

    feats = {
        "mean": float(np.mean(means)),
        "variance": float(np.mean(vars_)),
        "std": float(np.mean(stds)),
        "rms": float(np.mean(rmss)),
        "min": float(np.mean(mins)),
        "max": float(np.mean(maxs)),
        "peak_to_peak": float(np.mean(ptps)),
        "energy": float(np.mean(energies)),
        "delta_power": float(np.mean(d_pow)),
        "theta_power": float(np.mean(t_pow)),
        "alpha_power": float(np.mean(a_pow)),
        "beta_power": float(np.mean(b_pow)),
        "gamma_power": float(np.mean(g_pow))
    }

    if "Extended" in feature_set_type or feature_set_type == "20_features":
        tot_pow = d_pow + t_pow + a_pow + b_pow + g_pow + 1e-9
        alpha_rel = a_pow / tot_pow
        beta_rel = b_pow / tot_pow
        theta_rel = t_pow / tot_pow
        mu_pow = a_pow
        mu_rel = alpha_rel
        ab_ratio = a_pow / (b_pow + 1e-8)

        feats.update({
            "total_power": float(np.mean(tot_pow)),
            "alpha_relative_power": float(np.mean(alpha_rel)),
            "beta_relative_power": float(np.mean(beta_rel)),
            "theta_relative_power": float(np.mean(theta_rel)),
            "mu_power": float(np.mean(mu_pow)),
            "mu_relative_power": float(np.mean(mu_rel)),
            "alpha_beta_ratio": float(np.mean(ab_ratio))
        })

    return feats


def build_experiment_dataset(
    subjects: list,
    data_dir: str = "data",
    channel_group: str = "Group A",
    window_sec: float = 2.0,
    overlap: float = 0.5,
    feature_set_type: str = "Extended (20 features)"
) -> pd.DataFrame:
    edf_files = find_edf_files_for_subjects(subjects, data_dir=data_dir)
    rows = []

    for edf_path in edf_files:
        filename = os.path.basename(edf_path)
        subject_id = filename[:4] if filename.startswith("S") else "S001"
        run_id = filename[4:7] if len(filename) >= 7 else "R03"

        try:
            raw_sub, df_events = load_and_preprocess_cached(edf_path, channel_group)
        except Exception:
            continue

        if df_events.empty:
            continue

        sfreq = raw_sub.info['sfreq']
        win_samples = int(window_sec * sfreq)
        step_samples = int(win_samples * (1.0 - overlap))
        if step_samples < 1:
            step_samples = 1

        for _, ev in df_events.iterrows():
            code = ev['event_code']
            if code not in LABEL_MAPPING:
                continue

            label = LABEL_MAPPING[code]
            onset_sample = int(ev['onset_sec'] * sfreq)
            end_sample = int(ev['end_sec'] * sfreq)

            current_sample = onset_sample
            while current_sample + win_samples <= end_sample:
                win_end_sample = current_sample + win_samples
                win_start_time = round(current_sample / sfreq, 4)
                win_end_time = round(win_end_sample / sfreq, 4)

                sig_data_uv = raw_sub.get_data(start=current_sample, stop=win_end_sample) * 1e6

                if np.isnan(sig_data_uv).any() or np.isinf(sig_data_uv).any():
                    current_sample += step_samples
                    continue

                avg_feats = extract_features_vectorized(sig_data_uv, sfreq, feature_set_type=feature_set_type)

                row_dict = {
                    "subject": subject_id,
                    "run": run_id,
                    "window_start": win_start_time,
                    "window_end": win_end_time,
                    "label": label
                }
                row_dict.update(avg_feats)
                rows.append(row_dict)

                current_sample += step_samples

    return pd.DataFrame(rows)


def run_step7_experiments(data_dir: str = "data", models_dir: str = "models") -> dict:
    print("==================================================")
    print("STARTING STEP 7 EXPERIMENT SUITE")
    print("==================================================")

    # 1. Baseline Preservation
    baseline_info = save_baseline_info(models_dir=models_dir)

    # 2. Subject Discovery
    all_subjects = discover_available_subjects(data_dir=data_dir)
    if not all_subjects:
        all_subjects = [f"S{i:03d}" for i in range(1, 11)]

    test_subjects = [s for s in all_subjects if s in ["S009", "S010"]]
    if not test_subjects:
        test_subjects = all_subjects[-2:]

    remaining_subs = [s for s in all_subjects if s not in test_subjects]
    val_subjects = [remaining_subs[-1]] if len(remaining_subs) > 1 else remaining_subs
    train_subjects = [s for s in remaining_subs if s not in val_subjects]

    print(f"Available Subjects ({len(all_subjects)}): {all_subjects}")
    print(f"Train Subjects      ({len(train_subjects)}): {train_subjects}")
    print(f"Validation Subjects ({len(val_subjects)}): {val_subjects}")
    print(f"Test Subjects       ({len(test_subjects)}): {test_subjects}")
    print("==================================================")

    channel_groups = ["Group A", "Group B", "Group C"]
    window_sizes = [1.0, 2.0, 3.0]
    feature_sets = ["Baseline (13 features)", "Extended (20 features)"]

    experiments_log = []
    exp_id = 1

    best_val_score = -1.0
    best_config = None

    for ch_grp in channel_groups:
        for win_sec in window_sizes:
            for feat_type in feature_sets:
                df_train = build_experiment_dataset(
                    subjects=train_subjects,
                    data_dir=data_dir,
                    channel_group=ch_grp,
                    window_sec=win_sec,
                    overlap=0.5,
                    feature_set_type=feat_type
                )

                df_val = build_experiment_dataset(
                    subjects=val_subjects,
                    data_dir=data_dir,
                    channel_group=ch_grp,
                    window_sec=win_sec,
                    overlap=0.5,
                    feature_set_type=feat_type
                )

                if df_train.empty or df_val.empty:
                    continue

                feat_cols = BASELINE_FEATURES if "Baseline" in feat_type else EXTENDED_FEATURES
                feat_cols = [c for c in feat_cols if c in df_train.columns]

                X_tr = df_train[feat_cols].values
                y_tr = df_train["label"].values

                X_va = df_val[feat_cols].values
                y_va = df_val["label"].values

                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_va_sc = scaler.transform(X_va)

                models_to_test = {
                    "Random Forest": [
                        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                        RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=4, random_state=42)
                    ],
                    "SVM (RBF)": [
                        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
                        SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
                    ],
                    "Logistic Regression": [
                        LogisticRegression(max_iter=1000, C=1.0, random_state=42)
                    ]
                }

                for model_name, model_instances in models_to_test.items():
                    for idx_inst, m_inst in enumerate(model_instances):
                        m_inst.fit(X_tr_sc, y_tr)
                        y_val_pred = m_inst.predict(X_va_sc)
                        val_acc = float(accuracy_score(y_va, y_val_pred))
                        val_f1 = float(f1_score(y_va, y_val_pred, average="macro", zero_division=0))

                        log_entry = {
                            "experiment_id": f"EXP_{exp_id:03d}",
                            "channel_group": ch_grp,
                            "window_size": win_sec,
                            "overlap": 0.5,
                            "feature_set": feat_type,
                            "model": f"{model_name} (v{idx_inst+1})",
                            "parameters": f"max_depth={getattr(m_inst, 'max_depth', None)}, C={getattr(m_inst, 'C', None)}",
                            "validation_score": val_f1,
                            "validation_accuracy": val_acc,
                            "test_score": "Pending",
                            "notes": f"Trained on {len(train_subjects)} subs, validated on {len(val_subjects)} sub."
                        }
                        experiments_log.append(log_entry)
                        exp_id += 1

                        if val_f1 > best_val_score:
                            best_val_score = val_f1
                            best_config = {
                                "channel_group": ch_grp,
                                "window_size": win_sec,
                                "overlap": 0.5,
                                "feature_set": feat_type,
                                "model_name": model_name,
                                "model_instance": m_inst,
                                "val_f1": val_f1,
                                "val_acc": val_acc,
                                "feature_cols": feat_cols
                            }

    df_exp_log = pd.DataFrame(experiments_log)
    csv_exp_path = os.path.join(models_dir, "experiment_results.csv")
    df_exp_log.to_csv(csv_exp_path, index=False)

    print(f"Logged {len(df_exp_log)} experiments to '{csv_exp_path}'.")

    # FINAL FIT & UNTOUCHED TEST EVALUATION
    df_train_val = build_experiment_dataset(
        subjects=train_subjects + val_subjects,
        data_dir=data_dir,
        channel_group=best_config["channel_group"],
        window_sec=best_config["window_size"],
        overlap=0.5,
        feature_set_type=best_config["feature_set"]
    )

    df_test = build_experiment_dataset(
        subjects=test_subjects,
        data_dir=data_dir,
        channel_group=best_config["channel_group"],
        window_sec=best_config["window_size"],
        overlap=0.5,
        feature_set_type=best_config["feature_set"]
    )

    feat_cols = best_config["feature_cols"]

    X_train_val = df_train_val[feat_cols].values
    y_train_val = df_train_val["label"].values

    X_test = df_test[feat_cols].values
    y_test = df_test["label"].values

    scaler_v2 = StandardScaler()
    X_tv_scaled = scaler_v2.fit_transform(X_train_val)
    X_test_scaled = scaler_v2.transform(X_test)

    if "Random Forest" in best_config["model_name"]:
        final_model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=4, random_state=42)
    elif "SVM" in best_config["model_name"]:
        final_model = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    else:
        final_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

    final_model.fit(X_tv_scaled, y_train_val)
    y_pred = final_model.predict(X_test_scaled)

    test_acc = float(accuracy_score(y_test, y_pred))
    test_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    test_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    test_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=EXPECTED_CLASSES).tolist()
    rep = classification_report(y_test, y_pred, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)

    # Subject-by-subject evaluation
    subject_results = {}
    for sub in test_subjects:
        sub_mask = df_test["subject"] == sub
        if sub_mask.sum() > 0:
            X_sub = df_test.loc[sub_mask, feat_cols].values
            y_sub = df_test.loc[sub_mask, "label"].values
            X_sub_scaled = scaler_v2.transform(X_sub)
            y_sub_pred = final_model.predict(X_sub_scaled)
            sub_acc = float(accuracy_score(y_sub, y_sub_pred))
            subject_results[sub] = {
                "accuracy": sub_acc,
                "total_windows": int(sub_mask.sum())
            }

    # Feature Importance (if RF)
    feature_importances = {}
    if hasattr(final_model, "feature_importances_"):
        imps = final_model.feature_importances_
        sorted_indices = np.argsort(imps)[::-1]
        for idx in sorted_indices:
            feature_importances[feat_cols[idx]] = float(imps[idx])

    # Sample Predictions
    sample_preds = []
    for i in range(min(10, len(df_test))):
        row = df_test.iloc[i]
        sample_preds.append({
            "subject": str(row["subject"]),
            "run": str(row["run"]),
            "window_time": f"{row['window_start']}–{row['window_end']} sec",
            "actual_label": str(row["label"]),
            "predicted_label": str(y_pred[i])
        })

    model_v2_path = os.path.join(models_dir, "eeg_model_v2.pkl")
    scaler_v2_path = os.path.join(models_dir, "scaler_v2.pkl")
    meta_v2_path = os.path.join(models_dir, "model_metadata_v2.json")

    joblib.dump(final_model, model_v2_path)
    joblib.dump(scaler_v2, scaler_v2_path)

    baseline_acc = baseline_info.get("baseline_accuracy", 0.5250596658711217)
    acc_diff = test_acc - baseline_acc

    metadata_v2 = {
        "model_name": best_config["model_name"],
        "channel_group": best_config["channel_group"],
        "window_size_sec": best_config["window_size"],
        "feature_set": best_config["feature_set"],
        "feature_names": feat_cols,
        "classes": EXPECTED_CLASSES,
        "training_subjects": train_subjects,
        "validation_subjects": val_subjects,
        "testing_subjects": test_subjects,
        "training_date": datetime.datetime.now().isoformat(),
        "baseline_accuracy": baseline_acc,
        "test_metrics": {
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec,
            "f1": test_f1,
            "accuracy_difference": acc_diff,
            "confusion_matrix": cm,
            "report": rep
        },
        "subject_results": subject_results,
        "feature_importances": feature_importances,
        "sample_predictions": sample_preds,
        "dataset_info": {
            "train_val_windows": len(df_train_val),
            "test_windows": len(df_test)
        }
    }

    with open(meta_v2_path, "w") as f:
        json.dump(metadata_v2, f, indent=2)

    print("==================================================")
    print("STEP 7 EXPERIMENT SUITE COMPLETED SUCCESSFULLY")
    print("==================================================")
    print(f"Baseline Accuracy:     {baseline_acc * 100:.2f}%")
    print(f"Improved Test Accuracy:{test_acc * 100:.2f}%")
    print(f"Difference:            {acc_diff * 100:+.2f}%")
    print("==================================================")

    return metadata_v2


if __name__ == "__main__":
    run_step7_experiments()
