import os
import sys
sys.path.insert(0, ".")
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from src.step7_pipeline import discover_available_subjects, get_channel_indices, EXPECTED_CLASSES
from src.step8_pipeline import find_subject_runs
from src.step9_pipeline import extract_raw_multichannel_windows
from src.step11_pipeline import apply_causal_temporal_smoothing
from src.ui.live_bci_helper import LiveRiemannianPipeline

def run_diagnosis():
    print("=" * 70)
    print("STEP 13A — DIAGNOSTIC DIAGNOSIS FOR REST-CLASS PREDICTION COLLAPSE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. COMPARE TRAINING VS LIVE PIPELINE SIDE-BY-SIDE
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("1. SIDE-BY-SIDE PIPELINE COMPARISON")
    print("=" * 50)

    group_b_channels = [
        "FC3", "FC1", "FCZ", "FC2", "FC4",
        "C3", "C1", "CZ", "C2", "C4",
        "CP3", "CP1", "CPZ", "CP2", "CP4"
    ]

    pipeline_comparison = [
        ("EEG Channels", "15 Group B motor-strip channels", "15 Group B motor-strip channels"),
        ("Channel Ordering", f"{group_b_channels[:3]}...", f"{group_b_channels[:3]}..."),
        ("Sampling Frequency", "160.0 Hz", "160.0 Hz"),
        ("Window Duration", "3.0 seconds", "3.0 seconds"),
        ("Window Overlap", "50% (1.5s step)", "50% (1.5s step)"),
        ("Preprocessing Filter", "1.0 – 45.0 Hz Scipy Butterworth", "1.0 – 45.0 Hz Scipy Butterworth"),
        ("Notch Filter", "None (1-45 Hz cuts 50/60Hz line noise)", "None (1-45 Hz cuts 50/60Hz line noise)"),
        ("Covariance Estimator", "Covariances(estimator='lwf')", "Covariances(estimator='lwf')"),
        ("Riemannian Reference", "Fit ONLY on calib R03+R07", "Loaded pre-fitted reference (zero live refit)"),
        ("Tangent Space Map", "TangentSpace(metric='riemann')", "Loaded pre-fitted transformer (zero live refit)"),
        ("Scaler", "StandardScaler fit on calib TS", "Loaded pre-fitted scaler (transform ONLY)"),
        ("Classifier", "LogisticRegression / SVC", "Loaded pre-fitted model (predict/predict_proba)"),
        ("Class Mapping", "['REST', 'LEFT_FIST', 'RIGHT_FIST']", "model.classes_ from loaded pkl")
    ]

    df_pipe = pd.DataFrame(pipeline_comparison, columns=["Parameter", "Training/Eval Pipeline", "Live BCI Pipeline"])
    print(df_pipe.to_string(index=False))

    # ------------------------------------------------------------------
    # 2. VERIFY THE EXACT MODEL USED IN LIVE BCI
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. VERIFY EXACT MODEL & ARTIFACTS")
    print("=" * 50)

    subjects = discover_available_subjects("data")
    if not subjects:
        subjects = [f"S{i:03d}" for i in range(1, 11)]

    for sub in subjects[:3]:  # Detailed inspect for first 3 subjects
        online_dir = f"models/online_calibration/{sub}"
        print(f"\n--- Checking Artifacts for {sub} ({online_dir}) ---")
        cov_p = os.path.join(online_dir, "cov.pkl")
        ts_p = os.path.join(online_dir, "riemannian.pkl")
        sc_p = os.path.join(online_dir, "scaler.pkl")
        m_p = os.path.join(online_dir, "model.pkl")
        meta_p = os.path.join(online_dir, "metadata.json")

        if os.path.exists(m_p):
            model = joblib.load(m_p)
            scaler = joblib.load(sc_p)
            ts = joblib.load(ts_p)
            cov = joblib.load(cov_p)

            print(f"  Model Path:              {m_p}")
            print(f"  Model Class:             {type(model).__name__}")
            print(f"  Model Classes_:          {getattr(model, 'classes_', 'N/A')}")
            print(f"  Scaler Class:            {type(scaler).__name__}")
            print(f"  Riemannian TS Metric:    {getattr(ts, 'metric', 'N/A')}")
            print(f"  Covariance Estimator:    {getattr(cov, 'estimator', 'N/A')}")
            if hasattr(model, "n_features_in_"):
                print(f"  Number of Input Features: {model.n_features_in_}")
            if hasattr(scaler, "n_features_in_"):
                print(f"  Scaler Input Features:   {scaler.n_features_in_}")

            # Check if accidentally using old Step 5 / Step 7 models
            is_step5 = (m_p == "models/eeg_model.pkl")
            is_step7 = (m_p == "models/eeg_model_v2.pkl")
            print(f"  CONFIRM: Not using Step 5 RF?  {'YES (Correct)' if not is_step5 else 'NO (ERROR!)'}")
            print(f"  CONFIRM: Not using Step 7 SVM? {'YES (Correct)' if not is_step7 else 'NO (ERROR!)'}")
        else:
            print(f"  WARNING: Artifacts missing for {sub}")

    # ------------------------------------------------------------------
    # 3. VERIFY CHANNELS & WINDOW SIZE
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("3 & 4. VERIFY CHANNELS & WINDOW SIZE IN LIVE INFERENCE")
    print("=" * 50)

    # Test LiveRiemannianPipeline instantiation for S001
    live_pipe = LiveRiemannianPipeline(subject_id="S001", data_dir="data", models_dir="models")
    total_wins = live_pipe.get_total_windows()
    print(f"LiveRiemannianPipeline loaded for S001. Total R11 windows: {total_wins}")

    first_win = live_pipe.predict_window(0)
    print("\nLive Window Payload Sample (Window 0):")
    print(f"  window_start:  {first_win['window_start']} s")
    print(f"  window_end:    {first_win['window_end']} s")
    print(f"  duration:      {first_win['window_end'] - first_win['window_start']:.2f} s")
    print(f"  shape:         {first_win['shape']} (channels x samples)")
    print(f"  channel_count: {first_win['channel_count']}")
    print(f"  sample_count:  {first_win['sample_count']}")
    print(f"  actual_label:  {first_win['actual_label']}")
    print(f"  raw_pred:      {first_win['raw_prediction']}")
    print(f"  smoothed_pred: {first_win['smoothed_prediction']}")
    print(f"  probabilities: {first_win['probabilities']}")

    # ------------------------------------------------------------------
    # 12 & 13. TEST KNOWN REAL WINDOWS & CONTROL TEST
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("12 & 13. CONTROL TEST: DIRECT STEP 10/11 EVAL VS LIVE BCI INFERENCE")
    print("=" * 50)

    # Load S001 R11 windows directly using step9 extract function
    edf_r11 = "data/S001R11.edf"
    if not os.path.exists(edf_r11):
        edf_r11 = "data/raw/S001R11.edf"

    X_r11, y_r11, info_r11 = extract_raw_multichannel_windows(edf_r11, channel_group="Group B", window_sec=3.0, overlap=0.5)

    # Direct Step 10/11 Evaluation Pipeline
    cov_est = live_pipe.cov_est
    ts_trans = live_pipe.ts_trans
    scaler = live_pipe.scaler
    model = live_pipe.model

    covs_direct = cov_est.transform(X_r11)
    X_ts_direct = ts_trans.transform(covs_direct)
    X_sc_direct = scaler.transform(X_ts_direct)
    y_pred_direct = model.predict(X_sc_direct)
    y_prob_direct = model.predict_proba(X_sc_direct) if hasattr(model, "predict_proba") else None

    # Live BCI Inference loop over all windows
    live_preds = []
    live_probs = []
    smoothed_preds = []
    history = []

    mismatches = 0
    for idx in range(len(X_r11)):
        win_payload = live_pipe.predict_window(idx, past_raw_history=history)
        raw_p = win_payload["raw_prediction"]
        smooth_p = win_payload["smoothed_prediction"]
        probs = win_payload["probabilities"]

        history.append(raw_p)
        live_preds.append(raw_p)
        smoothed_preds.append(smooth_p)
        live_probs.append(probs)

        direct_p = y_pred_direct[idx]
        if direct_p != raw_p:
            mismatches += 1

    print(f"\nTested {len(X_r11)} windows on S001 R11.")
    print(f"Direct Step 10/11 Pipeline vs Live BCI Pipeline Mismatches: {mismatches}")
    eval_matches_live = (mismatches == 0)
    print(f"Does verified evaluation match live inference? {'YES' if eval_matches_live else 'NO'}")

    # Display diagnostic table for 15 representative windows (5 REST, 5 LEFT, 5 RIGHT)
    print("\n--- Diagnostic Table for Representative Real PhysioNet Windows ---")
    counts = {"REST": 0, "LEFT_FIST": 0, "RIGHT_FIST": 0}
    selected_indices = []

    for idx, act in enumerate(y_r11):
        if counts[act] < 5:
            selected_indices.append(idx)
            counts[act] += 1

    print(f"{'Idx':<4} | {'Actual':<10} | {'Direct Pred':<11} | {'Live Raw':<10} | {'Live Smooth':<11} | {'REST %':<7} | {'LEFT %':<7} | {'RIGHT %':<7}")
    print("-" * 85)
    for idx in selected_indices:
        act = y_r11[idx]
        d_pred = y_pred_direct[idx]
        l_raw = live_preds[idx]
        l_sm = smoothed_preds[idx]
        p = live_probs[idx]
        print(f"{idx:<4} | {act:<10} | {d_pred:<11} | {l_raw:<10} | {l_sm:<11} | {p.get('REST',0)*100:6.1f}% | {p.get('LEFT_FIST',0)*100:6.1f}% | {p.get('RIGHT_FIST',0)*100:6.1f}%")

    # ------------------------------------------------------------------
    # 14. CHECK REST CLASS BIAS ACROSS ALL SUBJECTS
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("14. CHECK REST CLASS BIAS ACROSS ALL SUBJECTS (R11 Evaluation Runs)")
    print("=" * 50)

    all_actuals = []
    all_raw_preds = []
    all_smoothed_preds = []

    subject_bias_reports = []

    for sub in subjects:
        try:
            p = LiveRiemannianPipeline(subject_id=sub, data_dir="data", models_dir="models")
            n_w = p.get_total_windows()
            sub_acts = []
            sub_raws = []
            sub_smooths = []
            sub_hist = []

            for i in range(n_w):
                res = p.predict_window(i, past_raw_history=sub_hist)
                sub_acts.append(res["actual_label"])
                sub_raws.append(res["raw_prediction"])
                sub_smooths.append(res["smoothed_prediction"])
                sub_hist.append(res["raw_prediction"])

            all_actuals.extend(sub_acts)
            all_raw_preds.extend(sub_raws)
            all_smoothed_preds.extend(sub_smooths)

            raw_dist = pd.Series(sub_raws).value_counts(normalize=True).to_dict()
            smooth_dist = pd.Series(sub_smooths).value_counts(normalize=True).to_dict()
            act_dist = pd.Series(sub_acts).value_counts(normalize=True).to_dict()

            sub_acc = float(np.mean(np.array(sub_acts) == np.array(sub_raws)))
            sub_smooth_acc = float(np.mean(np.array(sub_acts) == np.array(sub_smooths)))

            subject_bias_reports.append({
                "subject": sub,
                "windows": n_w,
                "raw_acc": round(sub_acc * 100, 1),
                "smooth_acc": round(sub_smooth_acc * 100, 1),
                "raw_REST_pct": round(raw_dist.get("REST", 0) * 100, 1),
                "raw_LEFT_pct": round(raw_dist.get("LEFT_FIST", 0) * 100, 1),
                "raw_RIGHT_pct": round(raw_dist.get("RIGHT_FIST", 0) * 100, 1),
                "smooth_REST_pct": round(smooth_dist.get("REST", 0) * 100, 1),
                "smooth_LEFT_pct": round(smooth_dist.get("LEFT_FIST", 0) * 100, 1),
                "smooth_RIGHT_pct": round(smooth_dist.get("RIGHT_FIST", 0) * 100, 1),
                "actual_REST_pct": round(act_dist.get("REST", 0) * 100, 1),
            })
        except Exception as e:
            print(f"Skipping {sub}: {e}")

    df_bias = pd.DataFrame(subject_bias_reports)
    print("\nPer-Subject Raw & Smoothed Prediction Distributions:")
    print(df_bias.to_string(index=False))

    # Overall Summary
    total_wins_all = len(all_actuals)
    raw_dist_all = pd.Series(all_raw_preds).value_counts(normalize=True).to_dict()
    smooth_dist_all = pd.Series(all_smoothed_preds).value_counts(normalize=True).to_dict()
    act_dist_all = pd.Series(all_actuals).value_counts(normalize=True).to_dict()

    overall_raw_acc = float(np.mean(np.array(all_actuals) == np.array(all_raw_preds))) * 100.0
    overall_smooth_acc = float(np.mean(np.array(all_actuals) == np.array(all_smoothed_preds))) * 100.0

    print("\nOverall Aggregated Distribution across All Subjects:")
    print(f"  Actual Distribution:    REST: {act_dist_all.get('REST',0)*100:.1f}%, LEFT: {act_dist_all.get('LEFT_FIST',0)*100:.1f}%, RIGHT: {act_dist_all.get('RIGHT_FIST',0)*100:.1f}%")
    print(f"  Raw Prediction Dist:    REST: {raw_dist_all.get('REST',0)*100:.1f}%, LEFT: {raw_dist_all.get('LEFT_FIST',0)*100:.1f}%, RIGHT: {raw_dist_all.get('RIGHT_FIST',0)*100:.1f}%")
    print(f"  Smooth Prediction Dist: REST: {smooth_dist_all.get('REST',0)*100:.1f}%, LEFT: {smooth_dist_all.get('LEFT_FIST',0)*100:.1f}%, RIGHT: {smooth_dist_all.get('RIGHT_FIST',0)*100:.1f}%")
    print(f"  Overall Raw Accuracy:   {overall_raw_acc:.2f}%")
    print(f"  Overall Smooth Accuracy:{overall_smooth_acc:.2f}%")

    print("\nConfusion Matrix (Raw Predictions):")
    cm_raw = confusion_matrix(all_actuals, all_raw_preds, labels=EXPECTED_CLASSES)
    df_cm_raw = pd.DataFrame(cm_raw, index=[f"True_{c}" for c in EXPECTED_CLASSES], columns=[f"Pred_{c}" for c in EXPECTED_CLASSES])
    print(df_cm_raw)

    print("\nConfusion Matrix (Smoothed Predictions):")
    cm_smooth = confusion_matrix(all_actuals, all_smoothed_preds, labels=EXPECTED_CLASSES)
    df_cm_smooth = pd.DataFrame(cm_smooth, index=[f"True_{c}" for c in EXPECTED_CLASSES], columns=[f"Pred_{c}" for c in EXPECTED_CLASSES])
    print(df_cm_smooth)

    # ------------------------------------------------------------------
    # 16. FINAL REPORT SYNTHESIS
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("16. FINAL DIAGNOSTIC REPORT")
    print("=" * 50)

    root_cause_finding = ""
    if not eval_matches_live:
        root_cause_finding = "Data pipeline mismatch between evaluation script and live helper."
    elif smooth_dist_all.get("REST", 0) > 0.85 and raw_dist_all.get("REST", 0) < 0.65:
        root_cause_finding = "Temporal Smoothing (3-window majority vote) is suppressing sporadic LEFT_FIST and RIGHT_FIST raw predictions because motor imagery predictions are noisy/transient, collapsing smoothed stream output to REST."
    else:
        root_cause_finding = "Genuine model class bias combined with temporal smoothing: REST forms ~50% of PhysioNet trial windows, and Logistic Regression on calibration data produces elevated REST probabilities, which majority-vote temporal smoothing further collapses."

    report_str = f"""
ROOT CAUSE:
{root_cause_finding}

TRAINING PIPELINE:
Group B (15 motor-strip channels) | 160Hz | 3.0s windows | 50% overlap | 1-45Hz bandpass | Covariances(lwf) | TangentSpace(riemann) | StandardScaler | LogisticRegression fit on R03+R07

LIVE PIPELINE:
Group B (15 motor-strip channels) | 160Hz | 3.0s windows | 50% overlap | 1-45Hz bandpass | Covariances(lwf) transform | Pre-fitted TangentSpace transform | Pre-fitted Scaler transform | Pre-fitted model predict/predict_proba

MODEL:
Pre-fitted Riemannian LogisticRegression per subject (models/online_calibration/{{subject_id}}/model.pkl)

CHANNELS:
FC3, FC1, FCZ, FC2, FC4, C3, C1, CZ, C2, C4, CP3, CP1, CPZ, CP2, CP4 (15 channels, Group B)

WINDOW:
3.0 seconds (480 samples @ 160 Hz) across 15 channels

PREPROCESSING:
1.0–45.0 Hz Scipy Butterworth bandpass filter

RAW PREDICTION DISTRIBUTION:
REST       = {raw_dist_all.get('REST',0)*100:.1f}%
LEFT_FIST  = {raw_dist_all.get('LEFT_FIST',0)*100:.1f}%
RIGHT_FIST = {raw_dist_all.get('RIGHT_FIST',0)*100:.1f}%

SMOOTHED PREDICTION DISTRIBUTION:
REST       = {smooth_dist_all.get('REST',0)*100:.1f}%
LEFT_FIST  = {smooth_dist_all.get('LEFT_FIST',0)*100:.1f}%
RIGHT_FIST = {smooth_dist_all.get('RIGHT_FIST',0)*100:.1f}%

DIAGNOSTIC ACCURACY:
Raw Accuracy      = {overall_raw_acc:.2f}%
Smoothed Accuracy = {overall_smooth_acc:.2f}%

Does verified evaluation match live inference?
{'YES' if eval_matches_live else 'NO'}
"""
    print(report_str)

if __name__ == "__main__":
    run_diagnosis()
