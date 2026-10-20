"""
Evaluate Phase 2C Calibration Methods on Held-out R11 Test Set (300 trials across S001-S010).
Measures whether calibration on Development data (budgets 5, 10, 20, 30) actually improves
unseen generalization on R11 compared to the frozen Step 62 baseline.

Metrics:
- Accuracy, Macro F1, Recalls (REST, LEFT, RIGHT), Precisions
- Safety Shield Coverage, Selective Accuracy, Rejected Accuracy
- Incorrect Command Execution Rate
- Per-subject variance and inference latency
"""

import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.step33_pipeline import GROUP_B_CHANNELS, CLASSES
from src.step59_pipeline import extract_custom_filter_bank_windows, extract_fbcsp_features_transform, align_probs
from src.step61_pipeline import load_dataset_step61
from scratch.run_phase2b_accuracy_experiments import (
    apply_surface_laplacian_approx,
    extract_enhanced_erd_asymmetry
)
from scratch.run_phase2c_calibration_experiments import (
    TemperatureScaler,
    AdaptiveLinearHead,
    extract_event_grouped_windows,
    calculate_bci_safety_metrics
)

def evaluate_holdout_calibration():
    print("=== Phase 2C Holdout Test (R11) Calibration Evaluation ===")
    sub_data = load_dataset_step61(data_dir="data")
    subs = sorted(list(sub_data.keys()))
    step62_base_dir = os.path.join(REPO_ROOT, "models", "step62", "selected_model")

    budgets = [5, 10, 20, 30]
    methods = [
        "1_frozen_step62_no_calib",
        "2_subject_specific_normalization",
        "3_probability_calibration",
        "4_lightweight_classifier_adaptation",
        "5_confidence_rejection_tuning"
    ]

    holdout_records = []

    for budget in budgets:
        print(f"\n--- Testing Calibration Budget {budget} on R11 ---")
        for method in methods:
            t_start = time.time()
            all_r11_gt = []
            all_r11_pred = []
            all_r11_probs = []
            latencies = []
            sub_results = []

            for s_id in subs:
                # Load Development calibration partition to simulate calibration session
                X_dev_raw = sub_data[s_id]["X_calib"].copy()
                y_dev_raw = sub_data[s_id]["y_calib"].copy()
                X_wins, y_wins, event_ids = extract_event_grouped_windows(X_dev_raw, y_dev_raw)

                calib_indices = []
                for c_name in CLASSES:
                    c_win_indices = np.where(y_wins == c_name)[0]
                    c_cal_idx = c_win_indices[:min(budget, len(c_win_indices))]
                    calib_indices.extend(c_cal_idx)
                calib_indices = np.array(calib_indices)

                X_calib_w = X_wins[calib_indices]
                y_calib_w = y_wins[calib_indices]

                # Untouched R11 Holdout Data
                X_r11_raw = sub_data[s_id]["X_r11"].copy() # 30 trials (15 REST, ~7-8 LEFT, ~7-8 RIGHT)
                y_r11_raw = sub_data[s_id]["y_r11"].copy()

                # Load Frozen Step 62 Model
                s_dir = os.path.join(step62_base_dir, s_id)
                bank = joblib.load(os.path.join(s_dir, "adaptive_bank.pkl"))
                csp_d = joblib.load(os.path.join(s_dir, "csp.pkl"))
                cov_est = joblib.load(os.path.join(s_dir, "cov_estimator.pkl"))
                ts = joblib.load(os.path.join(s_dir, "tangent_space.pkl"))
                sel = joblib.load(os.path.join(s_dir, "feature_selector.pkl"))
                scaler = joblib.load(os.path.join(s_dir, "scaler.pkl"))
                base_clf = joblib.load(os.path.join(s_dir, "model.pkl"))

                # Preprocessing & Feature Extraction
                X_calib_lap = apply_surface_laplacian_approx(X_calib_w)
                X_r11_lap = apply_surface_laplacian_approx(X_r11_raw)

                filt_calib = extract_custom_filter_bank_windows(X_calib_lap, bank)
                filt_r11 = extract_custom_filter_bank_windows(X_r11_lap, bank)

                X_calib_fbcsp = extract_fbcsp_features_transform(filt_calib, csp_d)
                X_r11_fbcsp = extract_fbcsp_features_transform(filt_r11, csp_d)

                X_calib_ts = ts.transform(cov_est.transform(X_calib_lap))
                X_r11_ts = ts.transform(cov_est.transform(X_r11_lap))

                X_calib_asym = extract_enhanced_erd_asymmetry(X_calib_lap)
                X_r11_asym = extract_enhanced_erd_asymmetry(X_r11_lap)

                X_calib_fused = np.hstack([X_calib_fbcsp, X_calib_ts, X_calib_asym])
                X_r11_fused = np.hstack([X_r11_fbcsp, X_r11_ts, X_r11_asym])

                X_calib_sel = sel.transform(X_calib_fused)
                X_r11_sel = sel.transform(X_r11_fused)

                # --- METHOD EXECUTION ---
                if method == "1_frozen_step62_no_calib":
                    X_r11_sc = scaler.transform(X_r11_sel)
                    t0_inf = time.perf_counter()
                    probs_raw = base_clf.predict_proba(X_r11_sc)
                    t_inf = (time.perf_counter() - t0_inf) * 1000.0 / len(X_r11_sc)
                    latencies.append(t_inf)
                    probs = align_probs(probs_raw, base_clf.classes_)
                    rest_th = 0.60

                elif method == "2_subject_specific_normalization":
                    calib_sc = StandardScaler().fit(X_calib_sel)
                    X_r11_sc = calib_sc.transform(X_r11_sel)
                    t0_inf = time.perf_counter()
                    probs_raw = base_clf.predict_proba(X_r11_sc)
                    t_inf = (time.perf_counter() - t0_inf) * 1000.0 / len(X_r11_sc)
                    latencies.append(t_inf)
                    probs = align_probs(probs_raw, base_clf.classes_)
                    rest_th = 0.60

                elif method == "3_probability_calibration":
                    X_calib_sc = scaler.transform(X_calib_sel)
                    X_r11_sc = scaler.transform(X_r11_sel)
                    calib_probs = align_probs(base_clf.predict_proba(X_calib_sc), base_clf.classes_)
                    temp_scaler = TemperatureScaler().fit(calib_probs, y_calib_w)
                    
                    t0_inf = time.perf_counter()
                    r11_probs_raw = align_probs(base_clf.predict_proba(X_r11_sc), base_clf.classes_)
                    probs = temp_scaler.transform(r11_probs_raw)
                    t_inf = (time.perf_counter() - t0_inf) * 1000.0 / len(X_r11_sc)
                    latencies.append(t_inf)
                    rest_th = 0.60

                elif method == "4_lightweight_classifier_adaptation":
                    X_calib_sc = scaler.transform(X_calib_sel)
                    X_r11_sc = scaler.transform(X_r11_sel)
                    adaptive_head = AdaptiveLinearHead(base_clf, alpha=0.35).fit_adaptation(X_calib_sc, y_calib_w)
                    
                    t0_inf = time.perf_counter()
                    probs_raw = adaptive_head.predict_proba(X_r11_sc)
                    t_inf = (time.perf_counter() - t0_inf) * 1000.0 / len(X_r11_sc)
                    latencies.append(t_inf)
                    probs = align_probs(probs_raw, base_clf.classes_)
                    rest_th = 0.60

                elif method == "5_confidence_rejection_tuning":
                    X_calib_sc = scaler.transform(X_calib_sel)
                    X_r11_sc = scaler.transform(X_r11_sel)
                    cal_probs = align_probs(base_clf.predict_proba(X_calib_sc), base_clf.classes_)
                    
                    best_th = 0.60
                    best_f1 = -1.0
                    for th_cand in np.arange(0.50, 0.75, 0.05):
                        c_preds = ["REST" if p[0] >= th_cand else ("LEFT_FIST" if p[1] >= p[2] else "RIGHT_FIST") for p in cal_probs]
                        c_f1 = f1_score(y_calib_w, c_preds, average='macro', zero_division=0)
                        if c_f1 > best_f1:
                            best_f1 = c_f1
                            best_th = float(th_cand)

                    t0_inf = time.perf_counter()
                    probs_raw = base_clf.predict_proba(X_r11_sc)
                    t_inf = (time.perf_counter() - t0_inf) * 1000.0 / len(X_r11_sc)
                    latencies.append(t_inf)
                    probs = align_probs(probs_raw, base_clf.classes_)
                    rest_th = best_th

                s_preds = []
                for p in probs:
                    if p[0] >= rest_th:
                        s_preds.append("REST")
                    else:
                        s_preds.append("LEFT_FIST" if p[1] >= p[2] else "RIGHT_FIST")

                s_acc = accuracy_score(y_r11_raw, s_preds) * 100.0
                s_f1 = f1_score(y_r11_raw, s_preds, average='macro', zero_division=0) * 100.0
                sub_results.append({
                    "subject": s_id,
                    "accuracy": round(s_acc, 2),
                    "macro_f1": round(s_f1, 2)
                })

                all_r11_gt.extend(y_r11_raw)
                all_r11_pred.extend(s_preds)
                all_r11_probs.extend(probs)

            # Global Metrics on R11
            global_acc = float(accuracy_score(all_r11_gt, all_r11_pred) * 100.0)
            global_f1 = float(f1_score(all_r11_gt, all_r11_pred, average='macro', zero_division=0) * 100.0)
            cm = confusion_matrix(all_r11_gt, all_r11_pred, labels=CLASSES)
            
            rest_rec = float(cm[0, 0] / max(1, np.sum(cm[0, :])) * 100.0)
            left_rec = float(cm[1, 1] / max(1, np.sum(cm[1, :])) * 100.0)
            right_rec = float(cm[2, 2] / max(1, np.sum(cm[2, :])) * 100.0)

            rest_prec = float(cm[0, 0] / max(1, np.sum(cm[:, 0])) * 100.0)
            left_prec = float(cm[1, 1] / max(1, np.sum(cm[:, 1])) * 100.0)
            right_prec = float(cm[2, 2] / max(1, np.sum(cm[:, 2])) * 100.0)

            safety_metrics = calculate_bci_safety_metrics(all_r11_gt, all_r11_pred, all_r11_probs)
            sub_accs = [r["accuracy"] for r in sub_results]
            sub_mean = float(np.mean(sub_accs))
            sub_std = float(np.std(sub_accs))
            mean_lat = float(np.mean(latencies))
            dur = round(time.time() - t_start, 1)

            record = {
                "calibration_budget": budget,
                "method": method,
                "holdout_accuracy": round(global_acc, 2),
                "holdout_macro_f1": round(global_f1, 2),
                "rest_recall": round(rest_rec, 2),
                "left_recall": round(left_rec, 2),
                "right_recall": round(right_rec, 2),
                "rest_precision": round(rest_prec, 2),
                "left_precision": round(left_prec, 2),
                "right_precision": round(right_prec, 2),
                "sub_mean_acc": round(sub_mean, 2),
                "sub_std_acc": round(sub_std, 2),
                "coverage": safety_metrics["coverage"],
                "selective_accuracy": safety_metrics["selective_accuracy"],
                "rejected_accuracy": safety_metrics["rejected_accuracy"],
                "incorrect_command_rate": safety_metrics["incorrect_command_rate"],
                "mean_latency_ms": round(mean_lat, 2),
                "confusion_matrix": cm.tolist(),
                "duration_sec": dur,
                "per_subject": sub_results
            }
            holdout_records.append(record)

            print(f"[{method} | Budget {budget}] R11 Acc: {global_acc:.2f}%, F1: {global_f1:.2f}%, Rest Rec: {rest_rec:.2f}%, Left: {left_rec:.2f}%, Right: {right_rec:.2f}% | SelAcc: {safety_metrics['selective_accuracy']:.2f}%, IncCmd: {safety_metrics['incorrect_command_rate']:.2f}%")

    out_json = "scratch/phase2c_holdout_calibration_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(holdout_records, f, indent=2)

    df = pd.DataFrame(holdout_records).drop(columns=["per_subject", "confusion_matrix"])
    df.to_csv("scratch/phase2c_holdout_calibration_results.csv", index=False)
    print(f"\nSaved R11 holdout calibration results to {out_json} and scratch/phase2c_holdout_calibration_results.csv")

if __name__ == "__main__":
    evaluate_holdout_calibration()
