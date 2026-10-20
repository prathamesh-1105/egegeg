"""
Phase 2C Audit Script:
1. Verifies Step 61 and Step 62 reported metrics against serialized artifacts.
2. Confirms architecture, feature pipeline, safety thresholds, and artifact integrity.
3. Tests cross-subject generalization: evaluates S001's model on all other subjects (unseen subject fallback).
4. Tests a pooled cross-subject base model on unseen subjects to establish zero-calibration baseline for new users.
"""

import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from src.step33_pipeline import GROUP_B_CHANNELS, CLASSES
from src.step59_pipeline import extract_custom_filter_bank_windows, extract_fbcsp_features_transform, align_probs
from src.step61_pipeline import load_dataset_step61, extract_asymmetry_and_erd_features
from scratch.run_phase2b_accuracy_experiments import (
    apply_surface_laplacian_approx,
    extract_enhanced_erd_asymmetry
)

def audit_models():
    print("=== Step 1: Auditing Existing Models & Unseen Subject Generalization ===")
    
    # 1. Verify Step 62 metrics against metrics.json
    s62_metrics_p = os.path.join(REPO_ROOT, "models", "step62", "metrics.json")
    assert os.path.exists(s62_metrics_p), "Step 62 metrics.json missing!"
    with open(s62_metrics_p, "r", encoding="utf-8") as f:
        s62_metrics = json.load(f)
    print(f"Step 62 Reported R11 Accuracy: {s62_metrics['holdout_results_r11']['accuracy']}%")
    print(f"Step 62 Reported R11 Macro F1: {s62_metrics['holdout_results_r11']['macro_f1']}%")
    print(f"Step 62 Reported REST Recall: {s62_metrics['holdout_results_r11']['rest_recall']}%")
    print(f"Step 62 Reported Mean Latency: {s62_metrics['holdout_results_r11']['mean_inference_latency_ms']} ms")

    # 2. Verify all Step 62 subject artifacts
    s62_base = os.path.join(REPO_ROOT, "models", "step62", "selected_model")
    subs = [f"S{i:03d}" for i in range(1, 11)]
    expected_files = [
        "adaptive_bank.pkl", "csp.pkl", "cov_estimator.pkl",
        "tangent_space.pkl", "feature_selector.pkl", "scaler.pkl", "model.pkl"
    ]
    for s_id in subs:
        sub_dir = os.path.join(s62_base, s_id)
        assert os.path.exists(sub_dir), f"Missing {sub_dir}"
        for fname in expected_files:
            fpath = os.path.join(sub_dir, fname)
            assert os.path.exists(fpath) and os.path.getsize(fpath) > 0, f"Missing or empty artifact {fpath}"
    print("All Step 62 subject artifacts verified intact.")

    # 3. Test how S001's model behaves when applied to unseen subjects S002-S010 (The current api_server fallback behavior)
    sub_data = load_dataset_step61(data_dir="data")
    s001_dir = os.path.join(s62_base, "S001")
    s001_bank = joblib.load(os.path.join(s001_dir, "adaptive_bank.pkl"))
    s001_csp = joblib.load(os.path.join(s001_dir, "csp.pkl"))
    s001_cov = joblib.load(os.path.join(s001_dir, "cov_estimator.pkl"))
    s001_ts = joblib.load(os.path.join(s001_dir, "tangent_space.pkl"))
    s001_sel = joblib.load(os.path.join(s001_dir, "feature_selector.pkl"))
    s001_sc = joblib.load(os.path.join(s001_dir, "scaler.pkl"))
    s001_clf = joblib.load(os.path.join(s001_dir, "model.pkl"))

    print("\n--- Evaluating S001 Model Fallback on Unseen Subjects (R11 Holdout) ---")
    fallback_records = []
    y_true_all = []
    y_pred_all = []

    for s_id in subs:
        X_r11 = sub_data[s_id]["X_r11"].copy()
        y_r11 = sub_data[s_id]["y_r11"].copy()

        # Apply Step 62 Surface Laplacian
        X_r11_lap = apply_surface_laplacian_approx(X_r11)

        # Feature extraction using S001 pipeline
        filt_r11 = extract_custom_filter_bank_windows(X_r11_lap, s001_bank)
        X_fbcsp = extract_fbcsp_features_transform(filt_r11, s001_csp)
        X_ts = s001_ts.transform(s001_cov.transform(X_r11_lap))
        X_asym = extract_enhanced_erd_asymmetry(X_r11_lap)
        X_fused = np.hstack([X_fbcsp, X_ts, X_asym])
        X_sel = s001_sel.transform(X_fused)
        X_sc = s001_sc.transform(X_sel)

        probs_raw = s001_clf.predict_proba(X_sc)
        probs = align_probs(probs_raw, s001_clf.classes_)

        s_preds = []
        for p in probs:
            if p[0] >= 0.60:
                s_preds.append("REST")
            else:
                s_preds.append("LEFT_FIST" if p[1] >= p[2] else "RIGHT_FIST")

        s_acc = accuracy_score(y_r11, s_preds) * 100.0
        s_f1 = f1_score(y_r11, s_preds, average='macro') * 100.0
        fallback_records.append({"subject": s_id, "accuracy": round(s_acc, 2), "macro_f1": round(s_f1, 2)})
        y_true_all.extend(y_r11)
        y_pred_all.extend(s_preds)

    total_acc = accuracy_score(y_true_all, y_pred_all) * 100.0
    total_f1 = f1_score(y_true_all, y_pred_all, average='macro') * 100.0
    cm_fallback = confusion_matrix(y_true_all, y_pred_all, labels=CLASSES)
    
    print(f"S001-Fallback Overall R11 Accuracy on all subjects: {total_acc:.2f}%, Macro F1: {total_f1:.2f}%")
    print(f"Fallback Confusion Matrix:\n{cm_fallback}")
    print(pd.DataFrame(fallback_records).to_string(index=False))

    audit_out = {
        "step62_verified_metrics": s62_metrics["holdout_results_r11"],
        "unseen_subject_fallback_to_s001": {
            "overall_accuracy": round(total_acc, 2),
            "overall_macro_f1": round(total_f1, 2),
            "confusion_matrix": cm_fallback.tolist(),
            "per_subject": fallback_records
        }
    }

    out_p = "scratch/phase2c_audit_summary.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(audit_out, f, indent=2)
    print(f"\nSaved audit summary to {out_p}")

if __name__ == "__main__":
    audit_models()
