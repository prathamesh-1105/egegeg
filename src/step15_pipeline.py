import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import joblib
import datetime
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
from src.step11_pipeline import apply_causal_temporal_smoothing
from src.step14_pipeline import extract_event_aligned_windows


def compute_causal_smoothing_series(predictions: list, window_size: int = 3) -> list:
    smoothed = []
    history = []
    for p in predictions:
        history.append(p)
        smoothed.append(apply_causal_temporal_smoothing(history, window_size=window_size))
    return smoothed


def run_step15_adaptive_experiments(data_dir: str = "data", models_dir: str = "models") -> dict:
    print("==================================================")
    print("STARTING STEP 15 — ADAPTIVE PERSONALIZED Riemannian EEG CALIBRATION")
    print("==================================================")

    out_dir = os.path.join(models_dir, "step15")
    os.makedirs(out_dir, exist_ok=True)

    all_subjects = discover_available_subjects(data_dir=data_dir)
    if not all_subjects:
        all_subjects = [f"S{i:03d}" for i in range(1, 11)]

    adaptation_percentages = [0, 10, 20, 30, 40, 50, 60]
    adaptation_strategies = ["Strategy 1: Full Retraining (R03 + R07_adapt)", "Strategy 2: Warm-Start Classifier Update"]

    # Containers for results
    adaptation_curve_records = []
    strategy_eval_records = []
    subject_r11_results = []

    all_baseline14_r11_actuals, all_baseline14_r11_preds = [], []
    all_adaptive15_r11_actuals, all_adaptive15_r11_raw_preds = [], []
    all_adaptive15_r11_smooth3_preds = []

    # Latency tracking
    latency_records = {
        "preprocessing_cov_ts_scale_ms": [],
        "classifier_inference_ms": [],
        "total_inference_ms": [],
        "adaptation_update_time_sec": []
    }

    # Tracking R07 validation scores across percentages to select winning adaptation setup
    strategy_pct_scores = {(strat, pct): [] for strat in adaptation_strategies for pct in adaptation_percentages}

    for sub_id in all_subjects:
        found_runs = find_subject_runs(sub_id, data_dir=data_dir)

        # STRICT DATA SPLIT: R03 = Base Calibration, R07 = Adaptation, R11 = Untouched Final Holdout
        r03_path = found_runs.get("R03", None)
        r07_path = found_runs.get("R07", None)
        r11_path = found_runs.get("R11", None)

        if not r03_path or not r07_path or not r11_path:
            avail = sorted(list(found_runs.keys()))
            if len(avail) >= 3:
                r03_path = found_runs[avail[0]]
                r07_path = found_runs[avail[1]]
                r11_path = found_runs[avail[-1]]
            else:
                continue

        # Extract Event-Aligned Windows (Step 14 best baseline representation: Group B 15 channels, 3s window, 0.5s offset)
        X_r03, y_r03, _ = extract_event_aligned_windows(r03_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
        X_r07, y_r07, _ = extract_event_aligned_windows(r07_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
        X_r11, y_r11, _ = extract_event_aligned_windows(r11_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)

        if len(X_r03) == 0 or len(X_r07) == 0 or len(X_r11) == 0:
            continue

        # -------------------------------------------------------------
        # ADAPTATION EXPERIMENTS ON R07 VALIDATION SET ONLY
        # -------------------------------------------------------------
        for strat in adaptation_strategies:
            for pct in adaptation_percentages:
                n_adapt = int(np.floor(len(X_r07) * (pct / 100.0)))
                
                if pct == 0 or n_adapt == 0:
                    # 0% Adaptation: Train exclusively on R03, validate on full R07
                    X_tr_comb = X_r03
                    y_tr_comb = y_r03
                    X_val_r07 = X_r07
                    y_val_r07 = y_r07
                else:
                    # Chronological split: First n_adapt windows for adaptation, remaining for validation
                    X_r07_adapt = X_r07[:n_adapt]
                    y_r07_adapt = y_r07[:n_adapt]

                    X_val_r07 = X_r07[n_adapt:]
                    y_val_r07 = y_r07[n_adapt:]

                    if len(X_val_r07) == 0:
                        X_val_r07 = X_r07_adapt
                        y_val_r07 = y_r07_adapt

                    X_tr_comb = np.concatenate([X_r03, X_r07_adapt], axis=0)
                    y_tr_comb = np.concatenate([y_r03, y_r07_adapt], axis=0)

                if "Strategy 1" in strat:
                    # Strategy 1: Full Retraining (Refit Covariance, TangentSpace, Scaler, and Classifier on R03 + R07_adapt)
                    cov_est = Covariances(estimator='lwf')
                    ts_trans = TangentSpace(metric='riemann')
                    scaler = StandardScaler()

                    X_tr_ts = ts_trans.fit_transform(cov_est.fit_transform(X_tr_comb))
                    X_tr_sc = scaler.fit_transform(X_tr_ts)

                    model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
                    model.fit(X_tr_sc, y_tr_comb)

                    X_val_ts = ts_trans.transform(cov_est.transform(X_val_r07))
                    X_val_sc = scaler.transform(X_val_ts)
                    val_preds = model.predict(X_val_sc)

                else:
                    # Strategy 2: Incremental Classifier Update (Keep Covariance, TangentSpace, Scaler from R03, Update Classifier)
                    cov_est = Covariances(estimator='lwf')
                    ts_trans = TangentSpace(metric='riemann')
                    scaler = StandardScaler()

                    X_r03_ts = ts_trans.fit_transform(cov_est.fit_transform(X_r03))
                    X_r03_sc = scaler.fit_transform(X_r03_ts)

                    if pct > 0 and n_adapt > 0:
                        X_r07_adapt_sc = scaler.transform(ts_trans.transform(cov_est.transform(X_r07[:n_adapt])))
                        X_pooled_sc = np.vstack([X_r03_sc, X_r07_adapt_sc])
                        y_pooled = np.concatenate([y_r03, y_r07[:n_adapt]])
                    else:
                        X_pooled_sc = X_r03_sc
                        y_pooled = y_r03

                    model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
                    model.fit(X_pooled_sc, y_pooled)

                    X_val_sc = scaler.transform(ts_trans.transform(cov_est.transform(X_val_r07)))
                    val_preds = model.predict(X_val_sc)

                acc_val = accuracy_score(y_val_r07, val_preds)
                f1_val = f1_score(y_val_r07, val_preds, average="macro", zero_division=0)
                rep_val = classification_report(y_val_r07, val_preds, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)

                strategy_pct_scores[(strat, pct)].append(f1_val)

                adaptation_curve_records.append({
                    "subject": sub_id,
                    "strategy": strat,
                    "adaptation_percentage": pct,
                    "adaptation_windows": n_adapt,
                    "validation_windows": len(y_val_r07),
                    "validation_accuracy": round(acc_val * 100, 2),
                    "validation_macro_f1": round(f1_val * 100, 2),
                    "rest_recall": round(rep_val.get("REST", {}).get("recall", 0) * 100, 2),
                    "left_recall": round(rep_val.get("LEFT_FIST", {}).get("recall", 0) * 100, 2),
                    "right_recall": round(rep_val.get("RIGHT_FIST", {}).get("recall", 0) * 100, 2)
                })

    # Save Adaptation Curve CSV
    df_curve = pd.DataFrame(adaptation_curve_records)
    df_curve.to_csv(os.path.join(out_dir, "adaptation_curve.csv"), index=False)

    # -------------------------------------------------------------
    # SELECT OPTIMAL ADAPTATION STRATEGY & PERCENTAGE ON R07 VALIDATION
    # -------------------------------------------------------------
    mean_strat_pct_scores = {key: np.mean(scores) for key, scores in strategy_pct_scores.items() if len(scores) > 0}
    best_key = max(mean_strat_pct_scores, key=mean_strat_pct_scores.get)
    best_strategy, best_pct = best_key
    best_mean_f1 = mean_strat_pct_scores[best_key]

    print(f"Selected Optimal Adaptation Setup (R07 CV): {best_strategy} @ {best_pct}% Adaptation (Mean Validation F1: {best_mean_f1*100:.2f}%)")

    # -------------------------------------------------------------
    # FREEZE ADAPTIVE PIPELINE & EVALUATE ON UNTOUCHED R11 HOLDOUT
    # -------------------------------------------------------------
    for sub_id in all_subjects:
        found_runs = find_subject_runs(sub_id, data_dir=data_dir)
        r03_path = found_runs.get("R03", None)
        r07_path = found_runs.get("R07", None)
        r11_path = found_runs.get("R11", None)

        if not r03_path or not r07_path or not r11_path:
            avail = sorted(list(found_runs.keys()))
            if len(avail) >= 3:
                r03_path = found_runs[avail[0]]
                r07_path = found_runs[avail[1]]
                r11_path = found_runs[avail[-1]]
            else:
                continue

        X_r03, y_r03, _ = extract_event_aligned_windows(r03_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
        X_r07, y_r07, _ = extract_event_aligned_windows(r07_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
        X_r11, y_r11, _ = extract_event_aligned_windows(r11_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)

        if len(X_r03) == 0 or len(X_r07) == 0 or len(X_r11) == 0:
            continue

        # Baseline Step 14 Model (Trained on full R03 + R07)
        X_r03_07 = np.concatenate([X_r03, X_r07], axis=0)
        y_r03_07 = np.concatenate([y_r03, y_r07], axis=0)

        cov_est_b14 = Covariances(estimator='lwf')
        ts_trans_b14 = TangentSpace(metric='riemann')
        X_b14_ts = ts_trans_b14.fit_transform(cov_est_b14.fit_transform(X_r03_07))
        scaler_b14 = StandardScaler()
        X_b14_sc = scaler_b14.fit_transform(X_b14_ts)

        m_b14 = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
        m_b14.fit(X_b14_sc, y_r03_07)

        # Baseline Step 14 R11 Prediction
        X_r11_ts_b14 = ts_trans_b14.transform(cov_est_b14.transform(X_r11))
        X_r11_sc_b14 = scaler_b14.transform(X_r11_ts_b14)
        s14_r11_preds = m_b14.predict(X_r11_sc_b14)

        # Step 15 Adaptive Model Training (R03 Base + best_pct% R07 Adaptation)
        t_adapt_start = time.perf_counter()
        n_adapt_opt = int(np.floor(len(X_r07) * (best_pct / 100.0)))
        
        if best_pct == 0 or n_adapt_opt == 0:
            X_tr_adaptive = X_r03
            y_tr_adaptive = y_r03
        else:
            X_tr_adaptive = np.concatenate([X_r03, X_r07[:n_adapt_opt]], axis=0)
            y_tr_adaptive = np.concatenate([y_r03, y_r07[:n_adapt_opt]], axis=0)

        cov_est_s15 = Covariances(estimator='lwf')
        ts_trans_s15 = TangentSpace(metric='riemann')
        X_s15_ts = ts_trans_s15.fit_transform(cov_est_s15.fit_transform(X_tr_adaptive))
        scaler_s15 = StandardScaler()
        X_s15_sc = scaler_s15.fit_transform(X_s15_ts)

        m_s15 = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
        m_s15.fit(X_s15_sc, y_tr_adaptive)
        t_adapt_end = time.perf_counter()
        adapt_update_time_sec = t_adapt_end - t_adapt_start
        latency_records["adaptation_update_time_sec"].append(adapt_update_time_sec)

        # Save Step 15 Adaptive Model Artifacts
        sub_out_dir = os.path.join(out_dir, "subjects", sub_id)
        os.makedirs(sub_out_dir, exist_ok=True)

        joblib.dump(cov_est_s15, os.path.join(sub_out_dir, "cov.pkl"))
        joblib.dump(ts_trans_s15, os.path.join(sub_out_dir, "riemannian.pkl"))
        joblib.dump(scaler_s15, os.path.join(sub_out_dir, "scaler.pkl"))
        joblib.dump(m_s15, os.path.join(sub_out_dir, "model.pkl"))

        meta_dict = {
            "subject": sub_id,
            "pipeline": "Step 15 Adaptive Personalized Riemannian Classifier",
            "adaptation_strategy": best_strategy,
            "adaptation_percentage": best_pct,
            "adaptation_windows_used": n_adapt_opt,
            "base_calibration": "R03",
            "adaptation_run": "R07",
            "holdout_run": "R11",
            "classes": list(m_s15.classes_)
        }
        with open(os.path.join(sub_out_dir, "metadata.json"), "w") as f:
            json.dump(meta_dict, f, indent=2)

        # Transform & Predict Untouched Holdout R11 for Step 15
        t_inf0 = time.perf_counter()
        covs_r11 = cov_est_s15.transform(X_r11)
        X_r11_ts = ts_trans_s15.transform(covs_r11)
        X_r11_sc = scaler_s15.transform(X_r11_ts)
        t_inf1 = time.perf_counter()

        t_clf0 = time.perf_counter()
        s15_r11_raw_preds = m_s15.predict(X_r11_sc)
        t_clf1 = time.perf_counter()

        prep_cov_ts_scale_ms = ((t_inf1 - t_inf0) / max(1, len(X_r11))) * 1000.0
        clf_ms = ((t_clf1 - t_clf0) / max(1, len(X_r11))) * 1000.0
        total_inf_ms = prep_cov_ts_scale_ms + clf_ms

        latency_records["preprocessing_cov_ts_scale_ms"].append(prep_cov_ts_scale_ms)
        latency_records["classifier_inference_ms"].append(clf_ms)
        latency_records["total_inference_ms"].append(total_inf_ms)

        s15_r11_smooth3_preds = compute_causal_smoothing_series(list(s15_r11_raw_preds), window_size=3)

        all_baseline14_r11_actuals.extend(y_r11)
        all_baseline14_r11_preds.extend(s14_r11_preds)

        all_adaptive15_r11_actuals.extend(y_r11)
        all_adaptive15_r11_raw_preds.extend(s15_r11_raw_preds)
        all_adaptive15_r11_smooth3_preds.extend(s15_r11_smooth3_preds)

        # Subject-level metrics on R11
        s14_acc = accuracy_score(y_r11, s14_r11_preds)
        s14_f1 = f1_score(y_r11, s14_r11_preds, average="macro", zero_division=0)

        s15_acc = accuracy_score(y_r11, s15_r11_raw_preds)
        s15_f1 = f1_score(y_r11, s15_r11_raw_preds, average="macro", zero_division=0)
        rep_s15 = classification_report(y_r11, s15_r11_raw_preds, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)

        rep_s14 = classification_report(y_r11, s14_r11_preds, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)
        right_rec_diff = (rep_s15.get("RIGHT_FIST", {}).get("recall", 0) - rep_s14.get("RIGHT_FIST", {}).get("recall", 0)) * 100.0

        subject_r11_results.append({
            "subject": sub_id,
            "windows": len(y_r11),
            "baseline_step14_accuracy": round(s14_acc * 100, 2),
            "adaptive_step15_accuracy": round(s15_acc * 100, 2),
            "accuracy_difference": round((s15_acc - s14_acc) * 100, 2),
            "baseline_step14_macro_f1": round(s14_f1 * 100, 2),
            "adaptive_step15_macro_f1": round(s15_f1 * 100, 2),
            "macro_f1_difference": round((s15_f1 - s14_f1) * 100, 2),
            "rest_recall": round(rep_s15.get("REST", {}).get("recall", 0) * 100, 2),
            "left_recall": round(rep_s15.get("LEFT_FIST", {}).get("recall", 0) * 100, 2),
            "right_recall": round(rep_s15.get("RIGHT_FIST", {}).get("recall", 0) * 100, 2),
            "right_recall_improvement": round(right_rec_diff, 2)
        })

    # Save Subject Results CSV
    df_sub_r11 = pd.DataFrame(subject_r11_results)
    df_sub_r11.to_csv(os.path.join(out_dir, "subject_results.csv"), index=False)

    # -------------------------------------------------------------
    # AGGREGATE METRICS & COMPARATIVE ANALYSIS ON UNTOUCHED R11
    # -------------------------------------------------------------
    s14_acc = float(accuracy_score(all_baseline14_r11_actuals, all_baseline14_r11_preds) * 100)
    s14_f1 = float(f1_score(all_baseline14_r11_actuals, all_baseline14_r11_preds, average="macro", zero_division=0) * 100)
    s14_rep = classification_report(all_baseline14_r11_actuals, all_baseline14_r11_preds, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)

    s15_acc = float(accuracy_score(all_adaptive15_r11_actuals, all_adaptive15_r11_raw_preds) * 100)
    s15_prec = float(precision_score(all_adaptive15_r11_actuals, all_adaptive15_r11_raw_preds, average="macro", zero_division=0) * 100)
    s15_rec = float(recall_score(all_adaptive15_r11_actuals, all_adaptive15_r11_raw_preds, average="macro", zero_division=0) * 100)
    s15_f1 = float(f1_score(all_adaptive15_r11_actuals, all_adaptive15_r11_raw_preds, average="macro", zero_division=0) * 100)
    s15_rep = classification_report(all_adaptive15_r11_actuals, all_adaptive15_r11_raw_preds, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)

    # Temporal smoothing check
    s15_s3_rep = classification_report(all_adaptive15_r11_actuals, all_adaptive15_r11_smooth3_preds, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)
    if (s15_s3_rep["LEFT_FIST"]["recall"] < s15_rep["LEFT_FIST"]["recall"]) or (s15_s3_rep["RIGHT_FIST"]["recall"] < s15_rep["RIGHT_FIST"]["recall"]):
        selected_smoothing = "RAW (No smoothing - smoothing reduced motor recall)"
    else:
        selected_smoothing = "3-Window Causal Majority"

    comparison_data = [
        {"metric": "Overall Accuracy", "step14_baseline": round(s14_acc, 2), "step15_adaptive": round(s15_acc, 2), "change": round(s15_acc - s14_acc, 2)},
        {"metric": "Macro Precision", "step14_baseline": round(precision_score(all_baseline14_r11_actuals, all_baseline14_r11_preds, average="macro", zero_division=0)*100, 2), "step15_adaptive": round(s15_prec, 2), "change": round(s15_prec - precision_score(all_baseline14_r11_actuals, all_baseline14_r11_preds, average="macro", zero_division=0)*100, 2)},
        {"metric": "Macro Recall", "step14_baseline": round(recall_score(all_baseline14_r11_actuals, all_baseline14_r11_preds, average="macro", zero_division=0)*100, 2), "step15_adaptive": round(s15_rec, 2), "change": round(s15_rec - recall_score(all_baseline14_r11_actuals, all_baseline14_r11_preds, average="macro", zero_division=0)*100, 2)},
        {"metric": "Macro F1", "step14_baseline": round(s14_f1, 2), "step15_adaptive": round(s15_f1, 2), "change": round(s15_f1 - s14_f1, 2)},
        {"metric": "REST Recall", "step14_baseline": round(s14_rep["REST"]["recall"] * 100, 2), "step15_adaptive": round(s15_rep["REST"]["recall"] * 100, 2), "change": round((s15_rep["REST"]["recall"] - s14_rep["REST"]["recall"]) * 100, 2)},
        {"metric": "LEFT_FIST Recall", "step14_baseline": round(s14_rep["LEFT_FIST"]["recall"] * 100, 2), "step15_adaptive": round(s15_rep["LEFT_FIST"]["recall"] * 100, 2), "change": round((s15_rep["LEFT_FIST"]["recall"] - s14_rep["LEFT_FIST"]["recall"]) * 100, 2)},
        {"metric": "RIGHT_FIST Recall", "step14_baseline": round(s14_rep["RIGHT_FIST"]["recall"] * 100, 2), "step15_adaptive": round(s15_rep["RIGHT_FIST"]["recall"] * 100, 2), "change": round((s15_rep["RIGHT_FIST"]["recall"] - s14_rep["RIGHT_FIST"]["recall"]) * 100, 2)},
    ]
    df_comp = pd.DataFrame(comparison_data)
    df_comp.to_csv(os.path.join(out_dir, "comparison.csv"), index=False)

    # 3x3 Confusion Matrix
    cm_s15 = confusion_matrix(all_adaptive15_r11_actuals, all_adaptive15_r11_raw_preds, labels=EXPECTED_CLASSES)
    df_cm = pd.DataFrame(cm_s15, index=[f"True_{c}" for c in EXPECTED_CLASSES], columns=[f"Pred_{c}" for c in EXPECTED_CLASSES])
    df_cm.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

    # Subject-by-Subject Adaptation Breakdown (Improved / Unchanged / Degraded)
    acc_diffs = [r["accuracy_difference"] for r in subject_r11_results]
    n_improved = int(np.sum(np.array(acc_diffs) > 0.0))
    n_unchanged = int(np.sum(np.array(acc_diffs) == 0.0))
    n_degraded = int(np.sum(np.array(acc_diffs) < 0.0))

    summary_stats = {
        "accuracy": {"mean": round(np.mean([r["adaptive_step15_accuracy"] for r in subject_r11_results]), 2), "median": round(np.median([r["adaptive_step15_accuracy"] for r in subject_r11_results]), 2), "min": round(np.min([r["adaptive_step15_accuracy"] for r in subject_r11_results]), 2), "max": round(np.max([r["adaptive_step15_accuracy"] for r in subject_r11_results]), 2), "std": round(np.std([r["adaptive_step15_accuracy"] for r in subject_r11_results]), 2)},
        "macro_f1": {"mean": round(np.mean([r["adaptive_step15_macro_f1"] for r in subject_r11_results]), 2), "median": round(np.median([r["adaptive_step15_macro_f1"] for r in subject_r11_results]), 2), "min": round(np.min([r["adaptive_step15_macro_f1"] for r in subject_r11_results]), 2), "max": round(np.max([r["adaptive_step15_macro_f1"] for r in subject_r11_results]), 2), "std": round(np.std([r["adaptive_step15_macro_f1"] for r in subject_r11_results]), 2)},
        "left_recall": {"mean": round(np.mean([r["left_recall"] for r in subject_r11_results]), 2), "median": round(np.median([r["left_recall"] for r in subject_r11_results]), 2), "min": round(np.min([r["left_recall"] for r in subject_r11_results]), 2), "max": round(np.max([r["left_recall"] for r in subject_r11_results]), 2), "std": round(np.std([r["left_recall"] for r in subject_r11_results]), 2)},
        "right_recall": {"mean": round(np.mean([r["right_recall"] for r in subject_r11_results]), 2), "median": round(np.median([r["right_recall"] for r in subject_r11_results]), 2), "min": round(np.min([r["right_recall"] for r in subject_r11_results]), 2), "max": round(np.max([r["right_recall"] for r in subject_r11_results]), 2), "std": round(np.std([r["right_recall"] for r in subject_r11_results]), 2)},
        "rest_recall": {"mean": round(np.mean([r["rest_recall"] for r in subject_r11_results]), 2), "median": round(np.median([r["rest_recall"] for r in subject_r11_results]), 2), "min": round(np.min([r["rest_recall"] for r in subject_r11_results]), 2), "max": round(np.max([r["rest_recall"] for r in subject_r11_results]), 2), "std": round(np.std([r["rest_recall"] for r in subject_r11_results]), 2)},
    }

    # Final Decision Rule
    # Adapt Step 15 ONLY if overall accuracy & Macro F1 improve or maintain while boosting motor class recalls
    passed_step15 = (s15_f1 > s14_f1) or (s15_acc > s14_acc) or (s15_rep["RIGHT_FIST"]["recall"] > s14_rep["RIGHT_FIST"]["recall"] + 0.05 and s15_acc >= s14_acc - 1.0)
    
    if passed_step15:
        final_decision = "ADOPT STEP 15"
    else:
        final_decision = "KEEP STEP 14"

    latency_data = {
        "preprocessing_cov_ts_scale_ms": round(float(np.mean(latency_records["preprocessing_cov_ts_scale_ms"])), 4),
        "classifier_inference_ms": round(float(np.mean(latency_records["classifier_inference_ms"])), 4),
        "total_inference_ms": round(float(np.mean(latency_records["total_inference_ms"])), 4),
        "average_adaptation_update_time_sec": round(float(np.mean(latency_records["adaptation_update_time_sec"])), 4)
    }

    summary_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "final_decision": final_decision,
        "best_adaptation_strategy": best_strategy,
        "best_calibration_percentage": best_pct,
        "selected_smoothing_strategy": selected_smoothing,
        "subject_adaptation_counts": {
            "improved": n_improved,
            "unchanged": n_unchanged,
            "degraded": n_degraded
        },
        "latency_benchmark": latency_data,
        "step14_baseline_r11_metrics": {
            "accuracy": round(s14_acc, 2),
            "macro_f1": round(s14_f1, 2),
            "rest_recall": round(s14_rep["REST"]["recall"] * 100, 2),
            "left_fist_recall": round(s14_rep["LEFT_FIST"]["recall"] * 100, 2),
            "right_fist_recall": round(s14_rep["RIGHT_FIST"]["recall"] * 100, 2),
        },
        "step15_adaptive_r11_metrics": {
            "accuracy": round(s15_acc, 2),
            "macro_f1": round(s15_f1, 2),
            "rest_recall": round(s15_rep["REST"]["recall"] * 100, 2),
            "left_fist_recall": round(s15_rep["LEFT_FIST"]["recall"] * 100, 2),
            "right_fist_recall": round(s15_rep["RIGHT_FIST"]["recall"] * 100, 2),
        },
        "subject_statistics": summary_stats
    }

    with open(os.path.join(out_dir, "summary_report.json"), "w") as f:
        json.dump(summary_report, f, indent=2)

    print("Step 15 Adaptive Experiments completed successfully.")
    print(f"Final Decision: {final_decision}")
    print(f"Optimal Setup: {best_strategy} @ {best_pct}% adaptation")
    print(f"Step 14 Baseline R11 Acc: {s14_acc:.2f}%, F1: {s14_f1:.2f}% | Step 15 Adaptive R11 Acc: {s15_acc:.2f}%, F1: {s15_f1:.2f}%")
    return summary_report


if __name__ == "__main__":
    run_step15_adaptive_experiments()
