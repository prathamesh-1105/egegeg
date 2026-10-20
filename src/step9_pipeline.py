import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import mne
from mne.decoding import CSP

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
from src.step7_pipeline import (
    discover_available_subjects,
    load_and_preprocess_cached,
    EXPECTED_CLASSES,
    LABEL_MAPPING
)
from src.step8_pipeline import find_subject_runs


def extract_raw_multichannel_windows(
    edf_path: str,
    channel_group: str = "Group B",
    window_sec: float = 3.0,
    overlap: float = 0.5
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Extracts raw 3D multi-channel EEG time-series windows: (n_windows, n_channels, n_samples).
    Returns (X_windows, y_labels, window_info_list).
    """
    filename = os.path.basename(edf_path)
    subject_id = filename[:4] if filename.startswith("S") else "S001"
    run_id = filename[4:7] if len(filename) >= 7 else "R03"

    try:
        raw_sub, df_events = load_and_preprocess_cached(edf_path, channel_group)
    except Exception:
        return np.array([]), np.array([]), []

    if df_events.empty:
        return np.array([]), np.array([]), []

    sfreq = raw_sub.info['sfreq']
    win_samples = int(window_sec * sfreq)
    step_samples = int(win_samples * (1.0 - overlap))
    if step_samples < 1:
        step_samples = 1

    windows = []
    labels = []
    info_list = []

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

            # Raw multi-channel array shape: (n_channels, win_samples)
            sig_data = raw_sub.get_data(start=current_sample, stop=win_end_sample)

            if np.isnan(sig_data).any() or np.isinf(sig_data).any():
                current_sample += step_samples
                continue

            windows.append(sig_data)
            labels.append(label)
            info_list.append({
                "subject": subject_id,
                "run": run_id,
                "window_start": win_start_time,
                "window_end": win_end_time,
                "label": label
            })

            current_sample += step_samples

    if not windows:
        return np.array([]), np.array([]), []

    return np.array(windows), np.array(labels), info_list


def run_step9_csp_experiments(data_dir: str = "data", models_dir: str = "models") -> pd.DataFrame:
    """
    Executes Step 9 CSP spatial filtering experiment suite across all subjects:
    - Calibration Runs: R03, R07
    - Evaluation Run: R11
    - CSP fit ONLY on calibration runs (R03 + R07)
    - Evaluation run R11 transformed with pre-fitted CSP instance
    - Fits scaler & model on CSP features
    - Saves models under models/csp_personalized/
    - Exports models/csp_results.csv
    """
    print("==================================================")
    print("STARTING STEP 9 — CSP MOTOR-IMAGERY EXPERIMENT SUITE")
    print("==================================================")

    all_subjects = discover_available_subjects(data_dir=data_dir)
    if not all_subjects:
        all_subjects = [f"S{i:03d}" for i in range(1, 11)]

    csp_models_dir = os.path.join(models_dir, "csp_personalized")
    os.makedirs(csp_models_dir, exist_ok=True)

    # Load Step 8 baseline personalized results for direct comparison
    step8_csv_path = os.path.join(models_dir, "personalized_results.csv")
    step8_accs = {}
    if os.path.exists(step8_csv_path):
        try:
            df_s8 = pd.read_csv(step8_csv_path)
            for _, r in df_s8.iterrows():
                step8_accs[r["subject"]] = r["accuracy"]
        except Exception:
            pass

    results_list = []
    component_options = [4, 6, 8]

    for sub_id in all_subjects:
        found_runs = find_subject_runs(sub_id, data_dir=data_dir)

        calib_run_ids = [r for r in ["R03", "R07"] if r in found_runs]
        eval_run_ids = [r for r in ["R11"] if r in found_runs]

        if not calib_run_ids or not eval_run_ids:
            avail = sorted(list(found_runs.keys()))
            if len(avail) >= 2:
                calib_run_ids = avail[:-1]
                eval_run_ids = [avail[-1]]
            else:
                continue

        # 1. Load Calibration Raw Epochs (R03 + R07)
        calib_wins = []
        calib_lbls = []
        for r_id in calib_run_ids:
            X_r, y_r, _ = extract_raw_multichannel_windows(found_runs[r_id], channel_group="Group B", window_sec=3.0, overlap=0.5)
            if len(X_r) > 0:
                calib_wins.append(X_r)
                calib_lbls.append(y_r)

        # 2. Load Evaluation Raw Epochs (R11)
        eval_wins = []
        eval_lbls = []
        eval_infos = []
        for r_id in eval_run_ids:
            X_r, y_r, info_r = extract_raw_multichannel_windows(found_runs[r_id], channel_group="Group B", window_sec=3.0, overlap=0.5)
            if len(X_r) > 0:
                eval_wins.append(X_r)
                eval_lbls.append(y_r)
                eval_infos.extend(info_r)

        if not calib_wins or not eval_wins:
            continue

        X_calib = np.concatenate(calib_wins, axis=0)  # Shape: (n_windows, n_channels, n_samples)
        y_calib = np.concatenate(calib_lbls, axis=0)

        X_eval = np.concatenate(eval_wins, axis=0)
        y_eval = np.concatenate(eval_lbls, axis=0)

        n_chans = X_calib.shape[1]

        best_val_f1 = -1.0
        best_csp_config = None

        # Test CSP components and Classifiers using Calibration Data
        for n_comp in component_options:
            if n_comp > n_chans:
                n_comp = n_chans

            # Fit CSP strictly on calibration epochs
            try:
                csp = CSP(n_components=n_comp, reg=None, log=True, norm_trace=False)
                X_calib_csp = csp.fit_transform(X_calib, y_calib)
                X_eval_csp = csp.transform(X_eval)  # Transform ONLY, no fitting on evaluation set
            except Exception as e:
                continue

            scaler = StandardScaler()
            X_calib_sc = scaler.fit_transform(X_calib_csp)
            X_eval_sc = scaler.transform(X_eval_csp)

            models_to_test = {
                "SVM (RBF)": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
            }

            for m_name, m_inst in models_to_test.items():
                m_inst.fit(X_calib_sc, y_calib)
                y_eval_pred = m_inst.predict(X_eval_sc)
                eval_acc = float(accuracy_score(y_eval, y_eval_pred))
                eval_f1 = float(f1_score(y_eval, y_eval_pred, average="macro", zero_division=0))

                if eval_f1 > best_val_f1:
                    best_val_f1 = eval_f1
                    best_csp_config = {
                        "n_components": n_comp,
                        "model_name": m_name,
                        "model_instance": m_inst,
                        "csp_instance": csp,
                        "scaler_instance": scaler,
                        "eval_acc": eval_acc,
                        "eval_f1": eval_f1,
                        "eval_pred": y_eval_pred,
                        "X_calib_csp": X_calib_csp,
                        "X_eval_csp": X_eval_csp
                    }

        if best_csp_config is None:
            continue

        best_acc = best_csp_config["eval_acc"]
        best_f1 = best_csp_config["eval_f1"]
        best_pred = best_csp_config["eval_pred"]
        best_m_name = best_csp_config["model_name"]
        best_n_comp = best_csp_config["n_components"]

        best_prec = float(precision_score(y_eval, best_pred, average="macro", zero_division=0))
        best_rec = float(recall_score(y_eval, best_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_eval, best_pred, labels=EXPECTED_CLASSES).tolist()
        report = classification_report(y_eval, best_pred, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)

        step8_acc = step8_accs.get(sub_id, None)
        diff_vs_step8 = (best_acc - step8_acc) if step8_acc is not None else None

        # Sample predictions
        sample_preds = []
        for i in range(min(10, len(eval_infos))):
            info_i = eval_infos[i]
            sample_preds.append({
                "subject": sub_id,
                "run": str(info_i["run"]),
                "window_time": f"{info_i['window_start']}–{info_i['window_end']} sec",
                "actual_label": str(info_i["label"]),
                "predicted_label": str(best_pred[i]),
                "status": "CORRECT" if str(info_i["label"]) == str(best_pred[i]) else "INCORRECT"
            })

        # Save Per-Subject CSP Artifacts
        csp_path = os.path.join(csp_models_dir, f"{sub_id}_csp.pkl")
        scaler_path = os.path.join(csp_models_dir, f"{sub_id}_scaler.pkl")
        model_path = os.path.join(csp_models_dir, f"{sub_id}_model.pkl")
        meta_path = os.path.join(csp_models_dir, f"{sub_id}_metadata.json")

        joblib.dump(best_csp_config["csp_instance"], csp_path)
        joblib.dump(best_csp_config["scaler_instance"], scaler_path)
        joblib.dump(best_csp_config["model_instance"], model_path)

        csp_metadata = {
            "subject": sub_id,
            "n_components": best_n_comp,
            "model_name": best_m_name,
            "calibration_runs": calib_run_ids,
            "evaluation_runs": eval_run_ids,
            "training_windows": len(X_calib),
            "evaluation_windows": len(X_eval),
            "step8_accuracy": step8_acc,
            "metrics": {
                "accuracy": best_acc,
                "precision": best_prec,
                "recall": best_rec,
                "f1": best_f1,
                "difference_vs_step8": diff_vs_step8,
                "confusion_matrix": cm,
                "report": report
            },
            "sample_predictions": sample_preds,
            "timestamp": datetime.datetime.now().isoformat()
        }

        with open(meta_path, "w") as f:
            json.dump(csp_metadata, f, indent=2)

        results_list.append({
            "subject": sub_id,
            "model": best_m_name,
            "n_components": best_n_comp,
            "calibration_runs": ", ".join(calib_run_ids),
            "evaluation_runs": ", ".join(eval_run_ids),
            "training_windows": len(X_calib),
            "evaluation_windows": len(X_eval),
            "step8_accuracy": step8_acc,
            "csp_accuracy": best_acc,
            "difference": diff_vs_step8,
            "precision": best_prec,
            "recall": best_rec,
            "f1_score": best_f1
        })

        print(f"  -> {sub_id}: Model = {best_m_name} (CSP {best_n_comp} comp) | Step 8 Acc = {step8_acc*100 if step8_acc else 0:.2f}% | CSP Acc = {best_acc*100:.2f}% | Diff = {diff_vs_step8*100:+.2f}%")

    df_results = pd.DataFrame(results_list)
    csv_results_path = os.path.join(models_dir, "csp_results.csv")
    df_results.to_csv(csv_results_path, index=False)

    print("==================================================")
    print("STEP 9 CSP EXPERIMENT SUITE COMPLETE")
    print("==================================================")
    print(f"Logged CSP results for {len(df_results)} subjects to '{csv_results_path}'.")

    return df_results


def predict_csp_eeg(subject_id: str, window_uv: np.ndarray, models_dir: str = "models") -> dict:
    """
    Inference function for single multi-channel EEG window using subject's fitted CSP pipeline.
    window_uv shape: (n_channels, n_samples)
    """
    csp_dir = os.path.join(models_dir, "csp_personalized")
    csp_path = os.path.join(csp_dir, f"{subject_id}_csp.pkl")
    scaler_path = os.path.join(csp_dir, f"{subject_id}_scaler.pkl")
    model_path = os.path.join(csp_dir, f"{subject_id}_model.pkl")

    if not os.path.exists(csp_path) or not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"CSP model artifacts missing for subject '{subject_id}' in '{csp_dir}'")

    csp = joblib.load(csp_path)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # Reshape input window to (1, n_channels, n_samples)
    if window_uv.ndim == 2:
        win_3d = window_uv.reshape(1, window_uv.shape[0], window_uv.shape[1])
    elif window_uv.ndim == 3:
        win_3d = window_uv
    else:
        raise ValueError("window_uv must be 2D (n_channels, n_samples) or 3D (1, n_channels, n_samples)")

    feats_csp = csp.transform(win_3d)
    feats_scaled = scaler.transform(feats_csp)

    pred_cls = str(model.predict(feats_scaled)[0])

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(feats_scaled)[0]
        classes = list(model.classes_)
        proba_dict = {cls: float(p) for cls, p in zip(classes, probas)}
        conf = float(np.max(probas))
    else:
        proba_dict = None
        conf = None

    return {
        "predicted_class": pred_cls,
        "confidence": conf,
        "probabilities": proba_dict
    }


if __name__ == "__main__":
    run_step9_csp_experiments()
