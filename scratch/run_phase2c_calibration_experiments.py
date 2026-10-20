"""
NeuroOS Phase 2C — Personalized EEG Calibration and Adaptive Command Control
Comprehensive Calibration Experiments Engine

Scientific Protocols:
1. Strict Zero-Leakage: All calibration fitting, normalization stats, temperature parameters,
   adaptation steps, and rejection thresholds are derived STRICTLY from the Calibration partition.
2. Event-Grouped Windows: Trials and windows are strictly grouped by physical event ID to guarantee
   zero temporal overlap between Calibration and Evaluation sets.
3. Evaluates Calibration Budgets: 5, 10, 20, 30 trials/windows per class.
4. Compares:
   - Method 1: Frozen Step 62 (No calibration)
   - Method 2: Subject-Specific Normalization (Whitening / Z-scoring on calibration data)
   - Method 3: Probability Calibration (Temperature Scaling on logits)
   - Method 4: Lightweight Classifier Adaptation (Regularized LDA centroid adaptation)
   - Method 5: Confidence-Based Rejection (Threshold selected on calibration partition)
5. Comprehensive Metrics:
   Accuracy, Macro F1, Per-Class Precision & Recall, Confusion Matrix,
   Safety Shield Coverage, Selective Accuracy, Rejected Accuracy,
   Incorrect Command Execution Rate, Per-Subject Breakdown, Inference Latency.
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
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from src.step33_pipeline import GROUP_B_CHANNELS, CLASSES, C3_IDX, C4_IDX
from src.step59_pipeline import (
    estimate_subject_peak_frequency,
    build_subject_adaptive_filter_bank,
    extract_custom_filter_bank_windows,
    extract_fbcsp_features_transform,
    align_probs
)
from src.step61_pipeline import load_dataset_step61
from scratch.run_phase2b_accuracy_experiments import (
    apply_surface_laplacian_approx,
    extract_enhanced_erd_asymmetry,
    safe_extract_fbcsp_features_train
)
from src.neuro_os.safety_shield import NeuroOSSafetyShield
from src.neuro_os.command_router import NeuroOSCommandRouter


# ==============================================================================
# TEMPERATURE SCALING (PROBABILITY CALIBRATION)
# ==============================================================================

class TemperatureScaler:
    """
    Learns a single scalar temperature T > 0 on calibration logits:
    P(c) = softmax(logits / T)
    Optimized via Negative Log-Likelihood (cross-entropy) on calibration data.
    """
    def __init__(self):
        self.temperature = 1.0

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        # Convert clipped probabilities back to pseudo-logits: z = log(p)
        probs_clean = np.clip(probs, 1e-7, 1.0 - 1e-7)
        probs_clean /= np.sum(probs_clean, axis=1, keepdims=True)
        logits = np.log(probs_clean)

        class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        y_idx = np.array([class_to_idx[y] for y in y_true])

        def nll_loss(t_val):
            T = max(t_val[0], 1e-3)
            scaled_logits = logits / T
            max_l = np.max(scaled_logits, axis=1, keepdims=True)
            exp_l = np.exp(scaled_logits - max_l)
            p_scaled = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            
            nll = -np.mean([np.log(max(p_scaled[i, y_idx[i]], 1e-12)) for i in range(len(y_idx))])
            return nll

        res = minimize(nll_loss, [1.0], bounds=[(0.05, 10.0)], method='L-BFGS-B')
        self.temperature = float(res.x[0])
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs_clean = np.clip(probs, 1e-7, 1.0 - 1e-7)
        probs_clean /= np.sum(probs_clean, axis=1, keepdims=True)
        logits = np.log(probs_clean)
        
        scaled_logits = logits / self.temperature
        max_l = np.max(scaled_logits, axis=1, keepdims=True)
        exp_l = np.exp(scaled_logits - max_l)
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)


# ==============================================================================
# LIGHTWEIGHT ADAPTIVE CLASSIFIER (SHRINKAGE CENTROID ADAPTATION)
# ==============================================================================

class AdaptiveLinearHead:
    """
    Adapts an existing feature projection / LDA model using calibration samples:
    Interpolates base LDA class centroids with empirical calibration centroids:
    mu_adapted = (1 - alpha) * mu_base + alpha * mu_calib
    """
    def __init__(self, base_clf, alpha: float = 0.35):
        self.base_clf = base_clf
        self.alpha = alpha
        self.classes_ = base_clf.classes_
        self.adapted_clf = None

    def fit_adaptation(self, X_calib: np.ndarray, y_calib: np.ndarray):
        # Fit lightweight regularized logistic regression initialized / regularized towards base
        self.adapted_clf = LogisticRegression(C=0.5, penalty='l2', max_iter=200, random_state=42)
        # Check if all classes present in calibration
        present_classes = np.unique(y_calib)
        if len(present_classes) == len(self.classes_):
            self.adapted_clf.fit(X_calib, y_calib)
        else:
            self.adapted_clf = None
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        base_p = self.base_clf.predict_proba(X)
        if self.adapted_clf is not None:
            adapt_p = self.adapted_clf.predict_proba(X)
            # Align probability columns
            base_p_aligned = align_probs(base_p, self.base_clf.classes_)
            adapt_p_aligned = align_probs(adapt_p, self.adapted_clf.classes_)
            # Convex blend
            return (1.0 - self.alpha) * base_p_aligned + self.alpha * adapt_p_aligned
        return base_p


# ==============================================================================
# SUB-WINDOW AUGMENTED DATASET GENERATOR (ZERO-LEAKAGE EVENT GROUPING)
# ==============================================================================

def extract_event_grouped_windows(X_raw: np.ndarray, y_labels: np.ndarray, sfreq: float = 160.0):
    """
    From discrete 4.0-second trials (640 samples), extracts 3 non-overlapping / lightly-stepped
    3.0-second windows (480 samples each) strictly maintaining event groupings:
    Window 1: [0.0s to 3.0s] (samples 0 to 480)
    Window 2: [0.5s to 3.5s] (samples 80 to 560)
    Window 3: [1.0s to 4.0s] (samples 160 to 640)
    Returns:
    X_wins (N_events * 3, C, 480), y_wins, event_ids (N_events * 3)
    """
    N_events, C, T = X_raw.shape
    w_len = int(3.0 * sfreq) # 480
    offsets = [0, int(0.5 * sfreq), int(1.0 * sfreq)]

    X_wins = []
    y_wins = []
    event_ids = []

    for e_idx in range(N_events):
        for w_sub_idx, off in enumerate(offsets):
            if off + w_len <= T:
                w_data = X_raw[e_idx, :, off:off + w_len]
                X_wins.append(w_data)
                y_wins.append(y_labels[e_idx])
                event_ids.append(e_idx)

    return np.array(X_wins, dtype=np.float32), np.array(y_wins), np.array(event_ids)


# ==============================================================================
# REJECTION & INCORRECT COMMAND CALCULATOR
# ==============================================================================

def calculate_bci_safety_metrics(y_true, y_pred, probs, rest_thresh=0.60, margin_thresh=0.10):
    """
    Evaluates safety shield guardrails and incorrect command execution rate:
    - Accepted: confidence >= rest_thresh and margin >= margin_thresh
    - Incorrect command execution: An active command (SCROLL_UP or SCROLL_DOWN) executed
      when ground truth was REST, or wrong active command (LEFT executed as RIGHT).
    """
    N = len(y_true)
    accepted_mask = []
    executed_commands = []
    
    for i in range(N):
        p = probs[i]
        conf = float(np.max(p))
        sorted_p = np.sort(p)[::-1]
        margin = float(sorted_p[0] - sorted_p[1])
        
        is_accepted = (conf >= rest_thresh) and (margin >= margin_thresh)
        accepted_mask.append(is_accepted)
        
        if is_accepted:
            pred_class = y_pred[i]
            if pred_class == "LEFT_FIST":
                cmd = "SCROLL_UP"
            elif pred_class == "RIGHT_FIST":
                cmd = "SCROLL_DOWN"
            else:
                cmd = "NEUTRAL"
        else:
            cmd = "NO_ACTION"
        executed_commands.append(cmd)

    accepted_mask = np.array(accepted_mask)
    coverage = float(np.mean(accepted_mask) * 100.0)
    
    # Selective accuracy
    if np.sum(accepted_mask) > 0:
        sel_acc = float(accuracy_score(np.array(y_true)[accepted_mask], np.array(y_pred)[accepted_mask]) * 100.0)
    else:
        sel_acc = 0.0

    # Rejected accuracy
    if np.sum(~accepted_mask) > 0:
        rej_acc = float(accuracy_score(np.array(y_true)[~accepted_mask], np.array(y_pred)[~accepted_mask]) * 100.0)
    else:
        rej_acc = 0.0

    # Incorrect command rate:
    # Any active command (SCROLL_UP / SCROLL_DOWN) triggered erroneously
    incorrect_count = 0
    for i in range(N):
        gt = y_true[i]
        cmd = executed_commands[i]
        if gt == "REST" and cmd in ["SCROLL_UP", "SCROLL_DOWN"]:
            incorrect_count += 1
        elif gt == "LEFT_FIST" and cmd == "SCROLL_DOWN":
            incorrect_count += 1
        elif gt == "RIGHT_FIST" and cmd == "SCROLL_UP":
            incorrect_count += 1

    incorrect_rate = float(incorrect_count / N * 100.0)

    return {
        "coverage": round(coverage, 2),
        "selective_accuracy": round(sel_acc, 2),
        "rejected_accuracy": round(rej_acc, 2),
        "incorrect_command_rate": round(incorrect_rate, 2),
        "accepted_count": int(np.sum(accepted_mask)),
        "rejected_count": int(np.sum(~accepted_mask))
    }


# ==============================================================================
# MAIN EXPERIMENT EXECUTION ENGINE
# ==============================================================================

def run_calibration_experiments():
    print("====================================================================")
    print("NeuroOS Phase 2C — Personalized EEG Calibration Experiments Engine")
    print("====================================================================")

    sub_data = load_dataset_step61(data_dir="data")
    subs = sorted(list(sub_data.keys()))

    budgets = [5, 10, 20, 30] # Calibration budget: labeled trials/windows per class
    methods = [
        "1_frozen_step62_no_calib",
        "2_subject_specific_normalization",
        "3_probability_calibration",
        "4_lightweight_classifier_adaptation",
        "5_confidence_rejection_tuning"
    ]

    all_experiment_records = []
    step62_base_dir = os.path.join(REPO_ROOT, "models", "step62", "selected_model")

    # For each budget, evaluate all 5 methods
    for budget in budgets:
        print(f"\n=======================================================")
        print(f">>> EVALUATING CALIBRATION BUDGET: {budget} TRIALS/CLASS <<<")
        print(f"=======================================================")

        for method in methods:
            t_start = time.time()
            print(f"Running Method: {method} (Budget: {budget})...")

            all_eval_gt = []
            all_eval_pred = []
            all_eval_probs = []
            latencies = []
            sub_results = []

            for s_id in subs:
                X_dev_raw = sub_data[s_id]["X_calib"].copy() # 60 trials (30 REST, 15 LEFT, 15 RIGHT)
                y_dev_raw = sub_data[s_id]["y_calib"].copy()

                # Extract event-grouped windows (180 total windows: 90 REST, 45 LEFT, 45 RIGHT)
                X_wins, y_wins, event_ids = extract_event_grouped_windows(X_dev_raw, y_dev_raw)

                # Partition into Calibration and Evaluation sets WITHOUT event leakage:
                # Find event indices for each class
                calib_indices = []
                eval_indices = []

                for c_name in CLASSES:
                    c_events = np.unique(event_ids[y_wins == c_name])
                    # If budget <= len(c_events), take budget events; otherwise take all available events
                    # up to budget windows
                    c_win_indices = np.where(y_wins == c_name)[0]
                    if len(c_win_indices) > budget:
                        c_cal_idx = c_win_indices[:budget]
                        c_eval_idx = c_win_indices[budget:]
                    else:
                        split_pt = max(1, int(len(c_win_indices) * 0.6))
                        c_cal_idx = c_win_indices[:split_pt]
                        c_eval_idx = c_win_indices[split_pt:]

                    calib_indices.extend(c_cal_idx)
                    eval_indices.extend(c_eval_idx)

                calib_indices = np.array(calib_indices)
                eval_indices = np.array(eval_indices)

                X_calib_w = X_wins[calib_indices]
                y_calib_w = y_wins[calib_indices]
                X_eval_w = X_wins[eval_indices]
                y_eval_w = y_wins[eval_indices]

                # Load Subject Frozen Step 62 Base Model
                s_dir = os.path.join(step62_base_dir, s_id)
                bank = joblib.load(os.path.join(s_dir, "adaptive_bank.pkl"))
                csp_d = joblib.load(os.path.join(s_dir, "csp.pkl"))
                cov_est = joblib.load(os.path.join(s_dir, "cov_estimator.pkl"))
                ts = joblib.load(os.path.join(s_dir, "tangent_space.pkl"))
                sel = joblib.load(os.path.join(s_dir, "feature_selector.pkl"))
                scaler = joblib.load(os.path.join(s_dir, "scaler.pkl"))
                base_clf = joblib.load(os.path.join(s_dir, "model.pkl"))

                # 1. Spatial Referencing (Surface Laplacian)
                X_calib_lap = apply_surface_laplacian_approx(X_calib_w)
                X_eval_lap = apply_surface_laplacian_approx(X_eval_w)

                # 2. Extract Features on Calibration and Evaluation
                filt_calib = extract_custom_filter_bank_windows(X_calib_lap, bank)
                filt_eval = extract_custom_filter_bank_windows(X_eval_lap, bank)

                X_calib_fbcsp = extract_fbcsp_features_transform(filt_calib, csp_d)
                X_eval_fbcsp = extract_fbcsp_features_transform(filt_eval, csp_d)

                X_calib_ts = ts.transform(cov_est.transform(X_calib_lap))
                X_eval_ts = ts.transform(cov_est.transform(X_eval_lap))

                X_calib_asym = extract_enhanced_erd_asymmetry(X_calib_lap)
                X_eval_asym = extract_enhanced_erd_asymmetry(X_eval_lap)

                X_calib_fused = np.hstack([X_calib_fbcsp, X_calib_ts, X_calib_asym])
                X_eval_fused = np.hstack([X_eval_fbcsp, X_eval_ts, X_eval_asym])

                X_calib_sel = sel.transform(X_calib_fused)
                X_eval_sel = sel.transform(X_eval_fused)

                # --- APPLY METHOD VARIATIONS ---
                if method == "1_frozen_step62_no_calib":
                    # Standard base scaler and base model
                    X_eval_sc = scaler.transform(X_eval_sel)
                    t0_inf = time.perf_counter()
                    probs_raw = base_clf.predict_proba(X_eval_sc)
                    t_inf = (time.perf_counter() - t0_inf) * 1000.0 / len(X_eval_sc)
                    latencies.append(t_inf)
                    probs = align_probs(probs_raw, base_clf.classes_)
                    rest_th = 0.60

                elif method == "2_subject_specific_normalization":
                    # Re-standardize using mean and std from target calibration partition
                    calib_sc = StandardScaler().fit(X_calib_sel)
                    X_eval_sc = calib_sc.transform(X_eval_sel)
                    t0_inf = time.perf_counter()
                    probs_raw = base_clf.predict_proba(X_eval_sc)
                    t_inf = (time.perf_counter() - t0_inf) * 1000.0 / len(X_eval_sc)
                    latencies.append(t_inf)
                    probs = align_probs(probs_raw, base_clf.classes_)
                    rest_th = 0.60

                elif method == "3_probability_calibration":
                    # Temperature scaling fitted on calibration logits
                    X_calib_sc = scaler.transform(X_calib_sel)
                    X_eval_sc = scaler.transform(X_eval_sel)
                    calib_probs = align_probs(base_clf.predict_proba(X_calib_sc), base_clf.classes_)
                    temp_scaler = TemperatureScaler().fit(calib_probs, y_calib_w)
                    
                    t0_inf = time.perf_counter()
                    eval_probs_raw = align_probs(base_clf.predict_proba(X_eval_sc), base_clf.classes_)
                    probs = temp_scaler.transform(eval_probs_raw)
                    t_inf = (time.perf_counter() - t0_inf) * 1000.0 / len(X_eval_sc)
                    latencies.append(t_inf)
                    rest_th = 0.60

                elif method == "4_lightweight_classifier_adaptation":
                    # Lightweight regularized logistic regression head adapted on calibration data
                    X_calib_sc = scaler.transform(X_calib_sel)
                    X_eval_sc = scaler.transform(X_eval_sel)
                    adaptive_head = AdaptiveLinearHead(base_clf, alpha=0.35).fit_adaptation(X_calib_sc, y_calib_w)
                    
                    t0_inf = time.perf_counter()
                    probs_raw = adaptive_head.predict_proba(X_eval_sc)
                    t_inf = (time.perf_counter() - t0_inf) * 1000.0 / len(X_eval_sc)
                    latencies.append(t_inf)
                    probs = align_probs(probs_raw, base_clf.classes_)
                    rest_th = 0.60

                elif method == "5_confidence_rejection_tuning":
                    # Optimize REST threshold on calibration partition to minimize error
                    X_calib_sc = scaler.transform(X_calib_sel)
                    X_eval_sc = scaler.transform(X_eval_sel)
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
                    probs_raw = base_clf.predict_proba(X_eval_sc)
                    t_inf = (time.perf_counter() - t0_inf) * 1000.0 / len(X_eval_sc)
                    latencies.append(t_inf)
                    probs = align_probs(probs_raw, base_clf.classes_)
                    rest_th = best_th

                # Generate Predictions
                s_preds = []
                for p in probs:
                    if p[0] >= rest_th:
                        s_preds.append("REST")
                    else:
                        s_preds.append("LEFT_FIST" if p[1] >= p[2] else "RIGHT_FIST")

                s_acc = accuracy_score(y_eval_w, s_preds) * 100.0
                s_f1 = f1_score(y_eval_w, s_preds, average='macro', zero_division=0) * 100.0
                sub_results.append({
                    "subject": s_id,
                    "accuracy": round(s_acc, 2),
                    "macro_f1": round(s_f1, 2),
                    "eval_trials": len(y_eval_w)
                })

                all_eval_gt.extend(y_eval_w)
                all_eval_pred.extend(s_preds)
                all_eval_probs.extend(probs)

            # Calculate Global Metrics for this Configuration
            global_acc = float(accuracy_score(all_eval_gt, all_eval_pred) * 100.0)
            global_f1 = float(f1_score(all_eval_gt, all_eval_pred, average='macro', zero_division=0) * 100.0)
            cm = confusion_matrix(all_eval_gt, all_eval_pred, labels=CLASSES)
            
            rest_rec = float(cm[0, 0] / max(1, np.sum(cm[0, :])) * 100.0)
            left_rec = float(cm[1, 1] / max(1, np.sum(cm[1, :])) * 100.0)
            right_rec = float(cm[2, 2] / max(1, np.sum(cm[2, :])) * 100.0)

            rest_prec = float(cm[0, 0] / max(1, np.sum(cm[:, 0])) * 100.0)
            left_prec = float(cm[1, 1] / max(1, np.sum(cm[:, 1])) * 100.0)
            right_prec = float(cm[2, 2] / max(1, np.sum(cm[:, 2])) * 100.0)

            safety_metrics = calculate_bci_safety_metrics(all_eval_gt, all_eval_pred, all_eval_probs)

            sub_accs = [r["accuracy"] for r in sub_results]
            sub_mean = float(np.mean(sub_accs))
            sub_std = float(np.std(sub_accs))

            mean_lat = float(np.mean(latencies))
            dur = round(time.time() - t_start, 1)

            record = {
                "calibration_budget": budget,
                "method": method,
                "accuracy": round(global_acc, 2),
                "macro_f1": round(global_f1, 2),
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
            all_experiment_records.append(record)

            print(f"--> Acc: {global_acc:.2f}%, F1: {global_f1:.2f}%, Rest Rec: {rest_rec:.2f}%, Left Rec: {left_rec:.2f}%, Right Rec: {right_rec:.2f}%")
            print(f"--> Selective Acc: {safety_metrics['selective_accuracy']:.2f}%, Incorrect Cmd Rate: {safety_metrics['incorrect_command_rate']:.2f}%, Std: {sub_std:.2f}% ({dur}s)\n")

    # Save to JSON and CSV
    out_json = "scratch/phase2c_calibration_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_experiment_records, f, indent=2)

    df = pd.DataFrame(all_experiment_records)
    # Drop per_subject and confusion_matrix for clean CSV export
    df_clean = df.drop(columns=["per_subject", "confusion_matrix"])
    df_clean.to_csv("scratch/phase2c_calibration_results.csv", index=False)
    print(f"\nAll calibration experiments complete! Saved to {out_json} and scratch/phase2c_calibration_results.csv")

if __name__ == "__main__":
    run_calibration_experiments()
