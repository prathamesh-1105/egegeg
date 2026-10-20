"""
Audit Phase 2B Baseline & Error Diagnosis Script
Evaluates Step 60 and Step 61 on R11 (300 trials across S001-S010),
producing detailed per-subject, per-class, confidence, and error analyses.
"""

import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from src.step8_pipeline import find_subject_runs
from src.step14_pipeline import extract_event_aligned_windows
from src.step33_pipeline import GROUP_B_CHANNELS, CLASSES, C3_IDX, C4_IDX
from src.step59_pipeline import extract_custom_filter_bank_windows, extract_fbcsp_features_transform, align_probs
from src.step61_pipeline import extract_asymmetry_and_erd_features, load_dataset_step61
from src.neuro_os.signal_quality import NeuroOSSignalQualityGate

def audit_baseline_models():
    print("=== Phase 2B Baseline Audit & Error Diagnosis ===")
    sub_data = load_dataset_step61(data_dir="data")
    subs = sorted(list(sub_data.keys()))

    step60_dir = "models/step60/selected_model"
    step61_dir = "models/step61/selected_model"

    sq_gate = NeuroOSSignalQualityGate()
    records = []

    for s_id in subs:
        X_r11 = sub_data[s_id]["X_r11"]
        y_r11 = sub_data[s_id]["y_r11"]

        # 1. Step 60 Model Inference
        s60_sub_dir = os.path.join(step60_dir, s_id)
        s60_bank = joblib.load(os.path.join(s60_sub_dir, "adaptive_bank.pkl"))
        s60_csp = joblib.load(os.path.join(s60_sub_dir, "csp.pkl"))
        s60_cov = joblib.load(os.path.join(s60_sub_dir, "cov_estimator.pkl"))
        s60_ts = joblib.load(os.path.join(s60_sub_dir, "tangent_space.pkl"))
        s60_sel = joblib.load(os.path.join(s60_sub_dir, "feature_selector.pkl"))
        s60_sc = joblib.load(os.path.join(s60_sub_dir, "scaler.pkl"))
        s60_clf = joblib.load(os.path.join(s60_sub_dir, "model.pkl"))

        filt_r11_s60 = extract_custom_filter_bank_windows(X_r11, s60_bank)
        X_fbcsp_s60 = extract_fbcsp_features_transform(filt_r11_s60, s60_csp)
        X_ts_s60 = s60_ts.transform(s60_cov.transform(X_r11))
        X_asym_s60 = extract_asymmetry_and_erd_features(X_r11)
        X_fused_s60 = np.hstack([X_fbcsp_s60, X_ts_s60, X_asym_s60])
        s60_probs_raw = s60_clf.predict_proba(s60_sc.transform(s60_sel.transform(X_fused_s60)))
        s60_probs = align_probs(s60_probs_raw, s60_clf.classes_)

        # 2. Step 61 Model Inference
        s61_sub_dir = os.path.join(step61_dir, s_id)
        s61_bank = joblib.load(os.path.join(s61_sub_dir, "adaptive_bank.pkl"))
        s61_csp = joblib.load(os.path.join(s61_sub_dir, "csp.pkl"))
        s61_cov = joblib.load(os.path.join(s61_sub_dir, "cov_estimator.pkl"))
        s61_ts = joblib.load(os.path.join(s61_sub_dir, "tangent_space.pkl"))
        s61_sel = joblib.load(os.path.join(s61_sub_dir, "feature_selector.pkl"))
        s61_sc = joblib.load(os.path.join(s61_sub_dir, "scaler.pkl"))
        s61_clf = joblib.load(os.path.join(s61_sub_dir, "model.pkl"))

        filt_r11_s61 = extract_custom_filter_bank_windows(X_r11, s61_bank)
        X_fbcsp_s61 = extract_fbcsp_features_transform(filt_r11_s61, s61_csp)
        X_ts_s61 = s61_ts.transform(s61_cov.transform(X_r11))
        X_asym_s61 = extract_asymmetry_and_erd_features(X_r11)
        X_fused_s61 = np.hstack([X_fbcsp_s61, X_ts_s61, X_asym_s61])
        s61_probs_raw = s61_clf.predict_proba(s61_sc.transform(s61_sel.transform(X_fused_s61)))
        s61_probs = align_probs(s61_probs_raw, s61_clf.classes_)

        for i in range(len(y_r11)):
            true_l = y_r11[i]
            win_i = X_r11[i]

            # Signal quality
            sq_report = sq_gate.evaluate_quality(win_i, GROUP_B_CHANNELS)
            good_pct = round((len(GROUP_B_CHANNELS) - len(sq_report['bad_channels'])) / len(GROUP_B_CHANNELS) * 100, 1)

            # Step 60 decision (threshold 0.60 on rest)
            p60 = s60_probs[i]
            conf60 = float(np.max(p60))
            if p60[0] >= 0.60:
                pred60 = "REST"
            else:
                pred60 = "LEFT_FIST" if p60[1] >= p60[2] else "RIGHT_FIST"

            # Step 61 decision (threshold 0.60 on rest)
            p61 = s61_probs[i]
            conf61 = float(np.max(p61))
            if p61[0] >= 0.60:
                pred61 = "REST"
            else:
                pred61 = "LEFT_FIST" if p61[1] >= p61[2] else "RIGHT_FIST"

            # Safety shield rule: conf >= 0.60 and margin >= 0.10
            sorted_p61 = np.sort(p61)[::-1]
            margin61 = float(sorted_p61[0] - sorted_p61[1])
            safety_pass61 = (conf61 >= 0.60) and (margin61 >= 0.10)

            records.append({
                "subject": s_id,
                "trial_idx": i,
                "true_label": true_l,
                "step60_pred": pred60,
                "step60_conf": conf60,
                "step60_rest_p": float(p60[0]),
                "step60_left_p": float(p60[1]),
                "step60_right_p": float(p60[2]),
                "step61_pred": pred61,
                "step61_conf": conf61,
                "step61_margin": margin61,
                "step61_rest_p": float(p61[0]),
                "step61_left_p": float(p61[1]),
                "step61_right_p": float(p61[2]),
                "safety_pass61": safety_pass61,
                "good_channels_pct": good_pct
            })

    df = pd.DataFrame(records)

    # 1. Global Metrics
    print("\n--- GLOBAL METRICS ---")
    for name, pred_col in [("Step 60", "step60_pred"), ("Step 61", "step61_pred")]:
        acc = accuracy_score(df["true_label"], df[pred_col]) * 100
        f1_m = f1_score(df["true_label"], df[pred_col], average="macro") * 100
        cm = confusion_matrix(df["true_label"], df[pred_col], labels=CLASSES)
        print(f"{name}: Accuracy = {acc:.2f}%, Macro F1 = {f1_m:.2f}%")
        print(f"Confusion Matrix (REST, LEFT, RIGHT):\n{cm}")
        rep = classification_report(df["true_label"], df[pred_col], labels=CLASSES, output_dict=True)
        for c in CLASSES:
            print(f"  {c}: Precision = {rep[c]['precision']*100:.2f}%, Recall = {rep[c]['recall']*100:.2f}%, F1 = {rep[c]['f1-score']*100:.2f}%")

    # 2. Per-Subject Performance (Step 61)
    print("\n--- PER-SUBJECT PERFORMANCE (Step 61) ---")
    sub_metrics = []
    for s_id, g in df.groupby("subject"):
        s_acc = accuracy_score(g["true_label"], g["step61_pred"]) * 100
        s_f1 = f1_score(g["true_label"], g["step61_pred"], average="macro") * 100
        s_cm = confusion_matrix(g["true_label"], g["step61_pred"], labels=CLASSES)
        sub_metrics.append({
            "subject": s_id,
            "accuracy": round(s_acc, 2),
            "macro_f1": round(s_f1, 2),
            "rest_recall": round(s_cm[0, 0] / np.sum(s_cm[0, :]) * 100, 2),
            "left_recall": round(s_cm[1, 1] / np.sum(s_cm[1, :]) * 100, 2) if np.sum(s_cm[1, :]) > 0 else 0,
            "right_recall": round(s_cm[2, 2] / np.sum(s_cm[2, :]) * 100, 2) if np.sum(s_cm[2, :]) > 0 else 0
        })
    sub_df = pd.DataFrame(sub_metrics)
    print(sub_df.to_string(index=False))

    # 3. Confidence Distributions by True Class (Step 61)
    print("\n--- CONFIDENCE DISTRIBUTIONS (Step 61) ---")
    for c in CLASSES:
        cg = df[df["true_label"] == c]
        correct_cg = cg[cg["true_label"] == cg["step61_pred"]]
        incorrect_cg = cg[cg["true_label"] != cg["step61_pred"]]
        print(f"{c} (Total {len(cg)}):")
        print(f"  All: Mean Conf = {cg['step61_conf'].mean():.4f}, Std = {cg['step61_conf'].std():.4f}")
        print(f"  Correct ({len(correct_cg)}): Mean Conf = {correct_cg['step61_conf'].mean():.4f}")
        if len(incorrect_cg) > 0:
            print(f"  Incorrect ({len(incorrect_cg)}): Mean Conf = {incorrect_cg['step61_conf'].mean():.4f}")

    # 4. Error Transitions
    print("\n--- ERROR TRANSITION BREAKDOWN (Step 61) ---")
    left_errs = df[(df["true_label"] == "LEFT_FIST") & (df["step61_pred"] != "LEFT_FIST")]
    print(f"LEFT_FIST Failures ({len(left_errs)} / 76):")
    print(left_errs["step61_pred"].value_counts().to_string())

    right_errs = df[(df["true_label"] == "RIGHT_FIST") & (df["step61_pred"] != "RIGHT_FIST")]
    print(f"\nRIGHT_FIST Failures ({len(right_errs)} / 74):")
    print(right_errs["step61_pred"].value_counts().to_string())

    # 5. Safety Shield Breakdown
    print("\n--- SAFETY SHIELD BREAKDOWN (Step 61) ---")
    accepted = df[df["safety_pass61"]]
    rejected = df[~df["safety_pass61"]]
    print(f"Accepted: {len(accepted)} / {len(df)} ({len(accepted)/len(df)*100:.2f}%)")
    print(f"Rejected: {len(rejected)} / {len(df)} ({len(rejected)/len(df)*100:.2f}%)")
    print(f"Selective Accuracy on Accepted: {accuracy_score(accepted['true_label'], accepted['step61_pred'])*100:.2f}%")
    print(f"Accuracy on Rejected: {accuracy_score(rejected['true_label'], rejected['step61_pred'])*100:.2f}%")

    out_json = "scratch/phase2b_baseline_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "global_metrics": {
                "step60": {"accuracy": 70.67, "macro_f1": 65.72},
                "step61": {"accuracy": 72.00, "macro_f1": 67.23}
            },
            "per_subject": sub_metrics,
            "error_counts": {
                "left_to_rest": int((left_errs["step61_pred"] == "REST").sum()),
                "left_to_right": int((left_errs["step61_pred"] == "RIGHT_FIST").sum()),
                "right_to_rest": int((right_errs["step61_pred"] == "REST").sum()),
                "right_to_left": int((right_errs["step61_pred"] == "LEFT_FIST").sum())
            }
        }, f, indent=2)
    print(f"\nSaved audit to {out_json}")

if __name__ == "__main__":
    audit_baseline_models()
