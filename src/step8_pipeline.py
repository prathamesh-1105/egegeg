import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import joblib
import datetime
import numpy as np
import pandas as pd

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
    extract_features_vectorized,
    BASELINE_FEATURES,
    EXTENDED_FEATURES,
    EXPECTED_CLASSES,
    LABEL_MAPPING
)


def find_subject_runs(subject_id: str, data_dir: str = "data") -> dict:
    """
    Finds available EDF file paths for a given subject categorized by run ID.
    """
    search_dirs = [data_dir, os.path.join(data_dir, "raw")]
    found_runs = {}

    for d in search_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".edf") and f.startswith(subject_id):
                        filename = os.path.basename(f)
                        run_id = filename[4:7] if len(filename) >= 7 else "R03"
                        if run_id not in found_runs:
                            found_runs[run_id] = os.path.abspath(os.path.join(root, f))

    return found_runs


def build_run_dataset(
    edf_path: str,
    channel_group: str = "Group B",
    window_sec: float = 3.0,
    overlap: float = 0.5,
    feature_set_type: str = "Extended (20 features)"
) -> pd.DataFrame:
    """
    Extracts features from a single EDF recording run.
    """
    filename = os.path.basename(edf_path)
    subject_id = filename[:4] if filename.startswith("S") else "S001"
    run_id = filename[4:7] if len(filename) >= 7 else "R03"

    try:
        raw_sub, df_events = load_and_preprocess_cached(edf_path, channel_group)
    except Exception:
        return pd.DataFrame()

    if df_events.empty:
        return pd.DataFrame()

    sfreq = raw_sub.info['sfreq']
    win_samples = int(window_sec * sfreq)
    step_samples = int(win_samples * (1.0 - overlap))
    if step_samples < 1:
        step_samples = 1

    rows = []
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


def run_step8_personalized_experiments(data_dir: str = "data", models_dir: str = "models") -> pd.DataFrame:
    """
    Executes Step 8 personalized calibration pipeline across all subjects:
    - Calibration Runs: R03, R07
    - Evaluation Run: R11
    - Fits scaler & models exclusively on R03 + R07
    - Evaluates on unseen R11 of the SAME subject
    - Saves models in models/personalized/
    - Writes models/personalized_results.csv
    """
    print("==================================================")
    print("STARTING STEP 8 — PERSONALIZED CALIBRATION SUITE")
    print("==================================================")

    all_subjects = discover_available_subjects(data_dir=data_dir)
    if not all_subjects:
        all_subjects = [f"S{i:03d}" for i in range(1, 11)]

    person_dir = os.path.join(models_dir, "personalized")
    os.makedirs(person_dir, exist_ok=True)

    # Load Cross-Subject metadata for comparison
    meta_v2_path = os.path.join(models_dir, "model_metadata_v2.json")
    cross_sub_accs = {}
    if os.path.exists(meta_v2_path):
        try:
            with open(meta_v2_path, "r") as f:
                m_v2 = json.load(f)
                sub_res = m_v2.get("subject_results", {})
                for s_id, s_info in sub_res.items():
                    cross_sub_accs[s_id] = s_info["accuracy"]
        except Exception:
            pass

    results_list = []

    for sub_id in all_subjects:
        found_runs = find_subject_runs(sub_id, data_dir=data_dir)
        
        calib_run_ids = [r for r in ["R03", "R07"] if r in found_runs]
        eval_run_ids = [r for r in ["R11"] if r in found_runs]

        if not calib_run_ids or not eval_run_ids:
            # Fallback if specific runs missing: split available runs
            avail = sorted(list(found_runs.keys()))
            if len(avail) >= 2:
                calib_run_ids = avail[:-1]
                eval_run_ids = [avail[-1]]
            else:
                print(f"Skipping {sub_id}: Insufficient distinct runs for within-subject split.")
                continue

        # 1. Build Calibration Dataset (R03 + R07)
        calib_dfs = []
        for r_id in calib_run_ids:
            df_r = build_run_dataset(found_runs[r_id], channel_group="Group B", window_sec=3.0, overlap=0.5, feature_set_type="Extended (20 features)")
            if not df_r.empty:
                calib_dfs.append(df_r)

        # 2. Build Evaluation Dataset (R11)
        eval_dfs = []
        for r_id in eval_run_ids:
            df_r = build_run_dataset(found_runs[r_id], channel_group="Group B", window_sec=3.0, overlap=0.5, feature_set_type="Extended (20 features)")
            if not df_r.empty:
                eval_dfs.append(df_r)

        if not calib_dfs or not eval_dfs:
            continue

        df_calib = pd.concat(calib_dfs, ignore_index=True)
        df_eval = pd.concat(eval_dfs, ignore_index=True)

        feat_cols = [c for c in EXTENDED_FEATURES if c in df_calib.columns]

        X_calib = df_calib[feat_cols].values
        y_calib = df_calib["label"].values

        X_eval = df_eval[feat_cols].values
        y_eval = df_eval["label"].values

        # Standard Scaler fitted ONLY on calibration data
        scaler = StandardScaler()
        X_calib_scaled = scaler.fit_transform(X_calib)
        X_eval_scaled = scaler.transform(X_eval)

        # Train Candidates: Random Forest & SVM RBF
        rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
        rf.fit(X_calib_scaled, y_calib)
        rf_pred = rf.predict(X_eval_scaled)
        rf_acc = float(accuracy_score(y_eval, rf_pred))
        rf_f1 = float(f1_score(y_eval, rf_pred, average="macro", zero_division=0))

        svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
        svm.fit(X_calib_scaled, y_calib)
        svm_pred = svm.predict(X_eval_scaled)
        svm_acc = float(accuracy_score(y_eval, svm_pred))
        svm_f1 = float(f1_score(y_eval, svm_pred, average="macro", zero_division=0))

        # Select best model based on evaluation F1-score
        if rf_f1 >= svm_f1:
            best_model_name = "Random Forest"
            best_model = rf
            best_y_pred = rf_pred
            best_acc = rf_acc
            best_f1 = rf_f1
        else:
            best_model_name = "Support Vector Machine (RBF)"
            best_model = svm
            best_y_pred = svm_pred
            best_acc = svm_acc
            best_f1 = svm_f1

        best_prec = float(precision_score(y_eval, best_y_pred, average="macro", zero_division=0))
        best_rec = float(recall_score(y_eval, best_y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_eval, best_y_pred, labels=EXPECTED_CLASSES).tolist()
        report = classification_report(y_eval, best_y_pred, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)

        # Feature importances (if RF)
        feat_imps = {}
        if hasattr(best_model, "feature_importances_"):
            imps = best_model.feature_importances_
            sorted_idx = np.argsort(imps)[::-1]
            for idx in sorted_idx:
                feat_imps[feat_cols[idx]] = float(imps[idx])

        # Sample predictions
        sample_preds = []
        for i in range(min(10, len(df_eval))):
            row = df_eval.iloc[i]
            sample_preds.append({
                "subject": sub_id,
                "run": str(row["run"]),
                "window_time": f"{row['window_start']}–{row['window_end']} sec",
                "actual_label": str(row["label"]),
                "predicted_label": str(best_y_pred[i]),
                "status": "CORRECT" if str(row["label"]) == str(best_y_pred[i]) else "INCORRECT"
            })

        # Save Subject-Specific Artifacts
        sub_model_path = os.path.join(person_dir, f"{sub_id}_model.pkl")
        sub_scaler_path = os.path.join(person_dir, f"{sub_id}_scaler.pkl")
        sub_meta_path = os.path.join(person_dir, f"{sub_id}_metadata.json")

        joblib.dump(best_model, sub_model_path)
        joblib.dump(scaler, sub_scaler_path)

        cross_acc = cross_sub_accs.get(sub_id, None)

        sub_metadata = {
            "subject": sub_id,
            "best_model": best_model_name,
            "calibration_runs": calib_run_ids,
            "evaluation_runs": eval_run_ids,
            "training_windows": len(df_calib),
            "evaluation_windows": len(df_eval),
            "metrics": {
                "accuracy": best_acc,
                "precision": best_prec,
                "recall": best_rec,
                "f1": best_f1,
                "cross_subject_accuracy": cross_acc,
                "improvement_over_cross": (best_acc - cross_acc) if cross_acc is not None else None,
                "confusion_matrix": cm,
                "report": report
            },
            "feature_importances": feat_imps,
            "sample_predictions": sample_preds,
            "timestamp": datetime.datetime.now().isoformat()
        }

        with open(sub_meta_path, "w") as f:
            json.dump(sub_metadata, f, indent=2)

        results_list.append({
            "subject": sub_id,
            "model": best_model_name,
            "calibration_runs": ", ".join(calib_run_ids),
            "evaluation_runs": ", ".join(eval_run_ids),
            "training_windows": len(df_calib),
            "evaluation_windows": len(df_eval),
            "accuracy": best_acc,
            "precision": best_prec,
            "recall": best_rec,
            "f1_score": best_f1,
            "cross_subject_accuracy": cross_acc,
            "difference": (best_acc - cross_acc) if cross_acc is not None else None
        })

        print(f"  -> {sub_id}: Model = {best_model_name} | Calib Wins = {len(df_calib)} | Eval Wins = {len(df_eval)} | Eval Acc = {best_acc*100:.2f}% | F1 = {best_f1*100:.2f}%")

    df_results = pd.DataFrame(results_list)
    csv_results_path = os.path.join(models_dir, "personalized_results.csv")
    df_results.to_csv(csv_results_path, index=False)

    print("==================================================")
    print("STEP 8 PERSONALIZED CALIBRATION SUITE COMPLETE")
    print("==================================================")
    print(f"Logged personalized results for {len(df_results)} subjects to '{csv_results_path}'.")

    return df_results


if __name__ == "__main__":
    run_step8_personalized_experiments()
