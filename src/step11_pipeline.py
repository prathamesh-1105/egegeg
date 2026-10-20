import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import joblib
import datetime
import time
import numpy as np
import pandas as pd
from collections import Counter
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.step7_pipeline import (
    discover_available_subjects,
    EXPECTED_CLASSES
)
from src.step8_pipeline import find_subject_runs
from src.step9_pipeline import extract_raw_multichannel_windows
from src.dataset_eeg_source import DatasetEEGSource


def apply_causal_temporal_smoothing(past_predictions: list, window_size: int = 3) -> str:
    """
    Applies causal temporal smoothing (majority voting) over the past `window_size` predictions.
    Uses strictly current and past predictions [P_{t-k}, ..., P_t] with zero future data leakage.
    If a tie occurs, returns the most recent prediction P_t.
    """
    if not past_predictions:
        return "REST"

    recent = past_predictions[-window_size:]
    counts = Counter(recent)
    max_count = max(counts.values())

    # Find candidates with max count
    candidates = [pred for pred, cnt in counts.items() if cnt == max_count]
    if len(candidates) == 1:
        return candidates[0]
    else:
        # Tie-breaker: return the most recent prediction
        return recent[-1]


def train_online_personalized_model(
    subject_id: str,
    data_dir: str = "data",
    models_dir: str = "models"
) -> dict:
    """
    Trains a subject-specific Riemannian Tangent Space model using online calibration runs (R03 + R07).
    Evaluates strictly on unseen holdout run (R11).
    Saves artifacts to models/online_calibration/{subject_id}/.
    """
    found_runs = find_subject_runs(subject_id, data_dir=data_dir)
    calib_run_ids = [r for r in ["R03", "R07"] if r in found_runs]
    eval_run_ids = [r for r in ["R11"] if r in found_runs]

    if not calib_run_ids or not eval_run_ids:
        avail = sorted(list(found_runs.keys()))
        if len(avail) >= 2:
            calib_run_ids = avail[:-1]
            eval_run_ids = [avail[-1]]
        else:
            raise ValueError(f"Insufficient runs for subject {subject_id}")

    # 1. Load Calibration Epochs (R03 + R07)
    calib_wins = []
    calib_lbls = []
    for r_id in calib_run_ids:
        X_r, y_r, _ = extract_raw_multichannel_windows(found_runs[r_id], channel_group="Group B", window_sec=3.0, overlap=0.5)
        if len(X_r) > 0:
            calib_wins.append(X_r)
            calib_lbls.append(y_r)

    # 2. Load Evaluation Epochs (R11)
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
        raise ValueError(f"No windows extracted for subject {subject_id}")

    X_calib = np.concatenate(calib_wins, axis=0)  # Shape: (n_windows, 15_channels, n_samples)
    y_calib = np.concatenate(calib_lbls, axis=0)

    X_eval = np.concatenate(eval_wins, axis=0)
    y_eval = np.concatenate(eval_lbls, axis=0)

    # 3. Fit Covariance & Tangent Space strictly on calibration runs
    cov_est = Covariances(estimator='lwf')
    covs_calib = cov_est.fit_transform(X_calib)

    ts_trans = TangentSpace(metric='riemann')
    X_calib_ts = ts_trans.fit_transform(covs_calib)

    scaler = StandardScaler()
    X_calib_sc = scaler.fit_transform(X_calib_ts)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_calib_sc, y_calib)

    # 4. Transform & Predict unseen evaluation run R11 (zero refitting)
    t0_start = time.time()
    covs_eval = cov_est.transform(X_eval)
    X_eval_ts = ts_trans.transform(covs_eval)
    X_eval_sc = scaler.transform(X_eval_ts)
    y_eval_pred = model.predict(X_eval_sc)
    t0_end = time.time()

    avg_latency_ms = float((t0_end - t0_start) / len(X_eval) * 1000.0)

    # Apply Causal Temporal Smoothing across sequential evaluation stream
    smoothed_preds = []
    raw_history = []
    for pred in y_eval_pred:
        raw_history.append(pred)
        s_pred = apply_causal_temporal_smoothing(raw_history, window_size=3)
        smoothed_preds.append(s_pred)

    raw_acc = float(accuracy_score(y_eval, y_eval_pred))
    smooth_acc = float(accuracy_score(y_eval, smoothed_preds))
    f1_val = float(f1_score(y_eval, smoothed_preds, average="macro", zero_division=0))
    prec_val = float(precision_score(y_eval, smoothed_preds, average="macro", zero_division=0))
    rec_val = float(recall_score(y_eval, smoothed_preds, average="macro", zero_division=0))
    cm = confusion_matrix(y_eval, smoothed_preds, labels=EXPECTED_CLASSES).tolist()

    # Build Sample Predictions
    sample_preds = []
    for i in range(len(eval_infos)):
        info_i = eval_infos[i]
        sample_preds.append({
            "subject": subject_id,
            "run": str(info_i["run"]),
            "window_time": f"{info_i['window_start']}–{info_i['window_end']} sec",
            "actual_label": str(info_i["label"]),
            "raw_predicted_label": str(y_eval_pred[i]),
            "smoothed_predicted_label": str(smoothed_preds[i]),
            "status": "CORRECT" if str(info_i["label"]) == str(smoothed_preds[i]) else "INCORRECT"
        })

    # Save Online Calibration Artifacts under models/online_calibration/{subject_id}/
    out_dir = os.path.join(models_dir, "online_calibration", subject_id)
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(cov_est, os.path.join(out_dir, "cov.pkl"))
    joblib.dump(ts_trans, os.path.join(out_dir, "riemannian.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(model, os.path.join(out_dir, "model.pkl"))

    metadata = {
        "subject": subject_id,
        "calibration_runs": calib_run_ids,
        "evaluation_runs": eval_run_ids,
        "calibration_source": "Real PhysioNet EEG recording (Runs R03 & R07)",
        "training_windows": len(X_calib),
        "evaluation_windows": len(X_eval),
        "raw_accuracy": raw_acc,
        "smoothed_accuracy": smooth_acc,
        "precision": prec_val,
        "recall": rec_val,
        "f1_score": f1_val,
        "avg_latency_ms": avg_latency_ms,
        "temporal_smoothing": "3-window causal majority vote",
        "confusion_matrix": cm,
        "sample_predictions": sample_preds,
        "timestamp": datetime.datetime.now().isoformat()
    }

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def run_step11_suite(data_dir: str = "data", models_dir: str = "models") -> pd.DataFrame:
    """
    Executes Step 11 Online Personal Calibration & BCI simulation benchmark across all subjects.
    """
    print("==================================================")
    print("STARTING STEP 11 — ONLINE PERSONAL BCI SUITE")
    print("==================================================")

    all_subjects = discover_available_subjects(data_dir=data_dir)
    if not all_subjects:
        all_subjects = [f"S{i:03d}" for i in range(1, 11)]

    results_list = []

    for sub_id in all_subjects:
        try:
            meta = train_online_personalized_model(sub_id, data_dir=data_dir, models_dir=models_dir)
            results_list.append({
                "subject": sub_id,
                "calibration_runs": ", ".join(meta["calibration_runs"]),
                "evaluation_runs": ", ".join(meta["evaluation_runs"]),
                "training_windows": meta["training_windows"],
                "evaluation_windows": meta["evaluation_windows"],
                "raw_accuracy": meta["raw_accuracy"],
                "smoothed_accuracy": meta["smoothed_accuracy"],
                "precision": meta["precision"],
                "recall": meta["recall"],
                "f1_score": meta["f1_score"],
                "avg_latency_ms": meta["avg_latency_ms"]
            })
            print(f"  -> {sub_id}: Calib Wins = {meta['training_windows']} | Eval Wins = {meta['evaluation_windows']} | Raw Acc = {meta['raw_accuracy']*100:.2f}% | Smoothed Acc = {meta['smoothed_accuracy']*100:.2f}% | Latency = {meta['avg_latency_ms']:.2f} ms")
        except Exception as e:
            print(f"  -> {sub_id}: Failed ({e})")

    df_res = pd.DataFrame(results_list)
    csv_path = os.path.join(models_dir, "online_results.csv")
    df_res.to_csv(csv_path, index=False)

    print("==================================================")
    print("STEP 11 ONLINE BCI SUITE COMPLETE")
    print("==================================================")
    print(f"Logged online BCI results for {len(df_res)} subjects to '{csv_path}'.")

    return df_res


if __name__ == "__main__":
    run_step11_suite()
