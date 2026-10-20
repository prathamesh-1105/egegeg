import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import joblib
import datetime
import numpy as np
import pandas as pd
from collections import Counter
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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

def compute_causal_smoothing_series(predictions: list, window_size: int = 3) -> list:
    smoothed = []
    history = []
    for p in predictions:
        history.append(p)
        smoothed.append(apply_causal_temporal_smoothing(history, window_size=window_size))
    return smoothed

def compute_custom_class_weights(y: np.ndarray) -> dict:
    classes = np.unique(y)
    total_samples = len(y)
    n_classes = len(classes)
    weights = {}
    counts = Counter(y)
    for c in classes:
        weights[c] = total_samples / (n_classes * max(1, counts[c]))
    return weights

def run_step13b_class_balanced_experiments(data_dir: str = "data", models_dir: str = "models") -> dict:
    print("==================================================")
    print("STARTING STEP 13B — CLASS-BALANCED RIEMANNIAN OPTIMIZATION")
    print("==================================================")

    out_dir = os.path.join(models_dir, "step13b")
    os.makedirs(out_dir, exist_ok=True)

    all_subjects = discover_available_subjects(data_dir=data_dir)
    if not all_subjects:
        all_subjects = [f"S{i:03d}" for i in range(1, 11)]

    # Containers for experiments
    exp_results = []
    prob_analysis_records = []
    subject_r11_results = []
    
    all_baseline_r11_actuals = []
    all_baseline_r11_raw_preds = []
    all_baseline_r11_smooth_preds = []

    all_new_r11_actuals = []
    all_new_r11_raw_preds = []
    all_new_r11_smooth_3_preds = []
    all_new_r11_smooth_5_preds = []

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

        # 1. Load Calibration Epochs (R03 + R07)
        calib_wins, calib_lbls = [], []
        for r_id in calib_run_ids:
            X_r, y_r, _ = extract_raw_multichannel_windows(found_runs[r_id], channel_group="Group B", window_sec=3.0, overlap=0.5)
            if len(X_r) > 0:
                calib_wins.append(X_r)
                calib_lbls.append(y_r)

        # 2. Load Evaluation Epochs (R11)
        eval_wins, eval_lbls, eval_infos = [], [], []
        for r_id in eval_run_ids:
            X_r, y_r, info_r = extract_raw_multichannel_windows(found_runs[r_id], channel_group="Group B", window_sec=3.0, overlap=0.5)
            if len(X_r) > 0:
                eval_wins.append(X_r)
                eval_lbls.append(y_r)
                eval_infos.extend(info_r)

        if not calib_wins or not eval_wins:
            continue

        X_calib = np.concatenate(calib_wins, axis=0)
        y_calib = np.concatenate(calib_lbls, axis=0)

        X_eval = np.concatenate(eval_wins, axis=0)
        y_eval = np.concatenate(eval_lbls, axis=0)

        # Fit Covariance & Tangent Space on R03 + R07 calibration data ONLY
        cov_est = Covariances(estimator='lwf')
        covs_calib = cov_est.fit_transform(X_calib)

        ts_trans = TangentSpace(metric='riemann')
        X_calib_ts = ts_trans.fit_transform(covs_calib)

        scaler = StandardScaler()
        X_calib_sc = scaler.fit_transform(X_calib_ts)

        # -------------------------------------------------------------
        # EXPERIMENTS A, B, C, D, E on CALIBRATION DATA ONLY (R03 + R07)
        # Using 5-Fold Stratified Cross-Validation on Calibration Data
        # -------------------------------------------------------------
        custom_weights = compute_custom_class_weights(y_calib)
        candidate_configs = {
            "Baseline (class_weight=None)": LogisticRegression(max_iter=1000, C=1.0, class_weight=None, random_state=42),
            "Class-Balanced (class_weight='balanced')": LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42),
            "Custom Weighted": LogisticRegression(max_iter=1000, C=1.0, class_weight=custom_weights, random_state=42),
            "Class-Balanced C=0.1": LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced", random_state=42),
            "Class-Balanced C=10.0": LogisticRegression(max_iter=1000, C=10.0, class_weight="balanced", random_state=42),
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_cv_f1 = -1.0
        best_cfg_name = None

        for cfg_name, model_inst in candidate_configs.items():
            cv_raw_preds = np.empty(len(y_calib), dtype=object)
            cv_probs_list = []
            cv_indices_list = []

            for train_idx, val_idx in skf.split(X_calib_sc, y_calib):
                X_tr, y_tr = X_calib_sc[train_idx], y_calib[train_idx]
                X_va, y_va = X_calib_sc[val_idx], y_calib[val_idx]

                m_clone = LogisticRegression(
                    max_iter=model_inst.max_iter,
                    C=model_inst.C,
                    class_weight=model_inst.class_weight,
                    random_state=model_inst.random_state
                )
                m_clone.fit(X_tr, y_tr)
                preds_va = m_clone.predict(X_va)
                probs_va = m_clone.predict_proba(X_va) if hasattr(m_clone, "predict_proba") else None

                cv_raw_preds[val_idx] = preds_va
                if probs_va is not None:
                    for i_local, idx_global in enumerate(val_idx):
                        prob_dict = {cls: float(p) for cls, p in zip(m_clone.classes_, probs_va[i_local])}
                        cv_probs_list.append((idx_global, prob_dict))

            # Temporal smoothing evaluation on calibration out-of-fold predictions
            cv_smooth_3 = compute_causal_smoothing_series(list(cv_raw_preds), window_size=3)
            cv_smooth_5 = compute_causal_smoothing_series(list(cv_raw_preds), window_size=5)

            # Metrics calculation for CV
            cv_acc = accuracy_score(y_calib, cv_raw_preds)
            cv_prec = precision_score(y_calib, cv_raw_preds, average="macro", zero_division=0)
            cv_rec = recall_score(y_calib, cv_raw_preds, average="macro", zero_division=0)
            cv_f1 = f1_score(y_calib, cv_raw_preds, average="macro", zero_division=0)

            rep = classification_report(y_calib, cv_raw_preds, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)

            exp_results.append({
                "subject": sub_id,
                "configuration": cfg_name,
                "cv_accuracy": round(cv_acc * 100, 2),
                "cv_macro_prec": round(cv_prec * 100, 2),
                "cv_macro_rec": round(cv_rec * 100, 2),
                "cv_macro_f1": round(cv_f1 * 100, 2),
                "rest_recall": round(rep.get("REST", {}).get("recall", 0) * 100, 2),
                "left_recall": round(rep.get("LEFT_FIST", {}).get("recall", 0) * 100, 2),
                "right_recall": round(rep.get("RIGHT_FIST", {}).get("recall", 0) * 100, 2),
            })

            # Record probability analysis for "Class-Balanced (class_weight='balanced')"
            if cfg_name == "Class-Balanced (class_weight='balanced')":
                for idx_g, p_dict in cv_probs_list:
                    prob_analysis_records.append({
                        "subject": sub_id,
                        "window_idx": idx_g,
                        "actual_label": y_calib[idx_g],
                        "prob_REST": p_dict.get("REST", 0),
                        "prob_LEFT_FIST": p_dict.get("LEFT_FIST", 0),
                        "prob_RIGHT_FIST": p_dict.get("RIGHT_FIST", 0)
                    })

            # Model Selection Criterion: Maximize CV Macro F1 on Calibration Data
            if cv_f1 > best_cv_f1:
                best_cv_f1 = cv_f1
                best_cfg_name = cfg_name

        # -------------------------------------------------------------
        # FREEZE SELECTED BEST MODEL CONFIGURATION & EVALUATE ON UNTOUCHED R11
        # -------------------------------------------------------------
        # Baseline Model Fit
        model_baseline = LogisticRegression(max_iter=1000, C=1.0, class_weight=None, random_state=42)
        model_baseline.fit(X_calib_sc, y_calib)

        # Selected Class-Balanced Model Fit (class_weight="balanced")
        model_selected = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
        model_selected.fit(X_calib_sc, y_calib)

        # Transform & Predict UNTOUCHED R11 Evaluation Data
        covs_eval = cov_est.transform(X_eval)
        X_eval_ts = ts_trans.transform(covs_eval)
        X_eval_sc = scaler.transform(X_eval_ts)

        # Baseline R11 Predictions
        base_r11_raw = model_baseline.predict(X_eval_sc)
        base_r11_smooth3 = compute_causal_smoothing_series(list(base_r11_raw), window_size=3)

        # New Class-Balanced R11 Predictions
        new_r11_raw = model_selected.predict(X_eval_sc)
        new_r11_smooth3 = compute_causal_smoothing_series(list(new_r11_raw), window_size=3)
        new_r11_smooth5 = compute_causal_smoothing_series(list(new_r11_raw), window_size=5)

        # Accumulate R11 outputs
        all_baseline_r11_actuals.extend(y_eval)
        all_baseline_r11_raw_preds.extend(base_r11_raw)
        all_baseline_r11_smooth_preds.extend(base_r11_smooth3)

        all_new_r11_actuals.extend(y_eval)
        all_new_r11_raw_preds.extend(new_r11_raw)
        all_new_r11_smooth_3_preds.extend(new_r11_smooth3)
        all_new_r11_smooth_5_preds.extend(new_r11_smooth5)

        # Calculate per-subject metrics on R11 for report
        base_acc = accuracy_score(y_eval, base_r11_raw)
        new_acc = accuracy_score(y_eval, new_r11_raw)
        new_f1 = f1_score(y_eval, new_r11_raw, average="macro", zero_division=0)
        
        rep_new = classification_report(y_eval, new_r11_raw, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)
        raw_dist_new = pd.Series(new_r11_raw).value_counts(normalize=True).to_dict()
        raw_dist_base = pd.Series(base_r11_raw).value_counts(normalize=True).to_dict()

        subject_r11_results.append({
            "subject": sub_id,
            "windows": len(y_eval),
            "baseline_accuracy": round(base_acc * 100, 2),
            "new_accuracy": round(new_acc * 100, 2),
            "accuracy_change": round((new_acc - base_acc) * 100, 2),
            "baseline_rest_pct": round(raw_dist_base.get("REST", 0) * 100, 2),
            "new_rest_pct": round(raw_dist_new.get("REST", 0) * 100, 2),
            "new_left_pct": round(raw_dist_new.get("LEFT_FIST", 0) * 100, 2),
            "new_right_pct": round(raw_dist_new.get("RIGHT_FIST", 0) * 100, 2),
            "rest_recall": round(rep_new.get("REST", {}).get("recall", 0) * 100, 2),
            "left_recall": round(rep_new.get("LEFT_FIST", {}).get("recall", 0) * 100, 2),
            "right_recall": round(rep_new.get("RIGHT_FIST", {}).get("recall", 0) * 100, 2),
            "macro_f1": round(new_f1 * 100, 2)
        })

        # Save Selected Class-Balanced Model Artifacts in models/step13b/selected_model/{sub_id}/
        sub_out_dir = os.path.join(out_dir, "selected_model", sub_id)
        os.makedirs(sub_out_dir, exist_ok=True)

        joblib.dump(cov_est, os.path.join(sub_out_dir, "cov.pkl"))
        joblib.dump(ts_trans, os.path.join(sub_out_dir, "riemannian.pkl"))
        joblib.dump(scaler, os.path.join(sub_out_dir, "scaler.pkl"))
        joblib.dump(model_selected, os.path.join(sub_out_dir, "model.pkl"))

        meta_dict = {
            "subject": sub_id,
            "pipeline": "Class-Balanced Riemannian Tangent Space LogisticRegression",
            "class_weight": "balanced",
            "channel_group": "Group B (15 channels)",
            "window_sec": 3.0,
            "r11_raw_accuracy": round(new_acc * 100, 2),
            "r11_macro_f1": round(new_f1 * 100, 2),
            "classes": list(model_selected.classes_)
        }
        with open(os.path.join(sub_out_dir, "metadata.json"), "w") as f:
            json.dump(meta_dict, f, indent=2)

    # -------------------------------------------------------------
    # OVERALL AGGREGATED METRICS & CSV PERSISTENCE
    # -------------------------------------------------------------
    df_exp = pd.DataFrame(exp_results)
    df_exp.to_csv(os.path.join(out_dir, "experiment_results.csv"), index=False)

    df_prob = pd.DataFrame(prob_analysis_records)
    df_prob.to_csv(os.path.join(out_dir, "probability_analysis.csv"), index=False)

    df_sub = pd.DataFrame(subject_r11_results)
    df_sub.to_csv(os.path.join(out_dir, "subject_results.csv"), index=False)

    # Overall Baseline vs New Model metrics on R11
    base_r11_acc = float(accuracy_score(all_baseline_r11_actuals, all_baseline_r11_raw_preds) * 100)
    base_r11_prec = float(precision_score(all_baseline_r11_actuals, all_baseline_r11_raw_preds, average="macro", zero_division=0) * 100)
    base_r11_rec = float(recall_score(all_baseline_r11_actuals, all_baseline_r11_raw_preds, average="macro", zero_division=0) * 100)
    base_r11_f1 = float(f1_score(all_baseline_r11_actuals, all_baseline_r11_raw_preds, average="macro", zero_division=0) * 100)
    base_rep = classification_report(all_baseline_r11_actuals, all_baseline_r11_raw_preds, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)
    base_dist = pd.Series(all_baseline_r11_raw_preds).value_counts(normalize=True).to_dict()

    new_r11_acc = float(accuracy_score(all_new_r11_actuals, all_new_r11_raw_preds) * 100)
    new_r11_prec = float(precision_score(all_new_r11_actuals, all_new_r11_raw_preds, average="macro", zero_division=0) * 100)
    new_r11_rec = float(recall_score(all_new_r11_actuals, all_new_r11_raw_preds, average="macro", zero_division=0) * 100)
    new_r11_f1 = float(f1_score(all_new_r11_actuals, all_new_r11_raw_preds, average="macro", zero_division=0) * 100)
    new_rep = classification_report(all_new_r11_actuals, all_new_r11_raw_preds, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)
    new_dist = pd.Series(all_new_r11_raw_preds).value_counts(normalize=True).to_dict()
    new_smooth3_dist = pd.Series(all_new_r11_smooth_3_preds).value_counts(normalize=True).to_dict()

    comparison_data = [
        {"metric": "Overall Accuracy", "baseline_r11": round(base_r11_acc, 2), "class_balanced_r11": round(new_r11_acc, 2), "change": round(new_r11_acc - base_r11_acc, 2)},
        {"metric": "Macro Precision", "baseline_r11": round(base_r11_prec, 2), "class_balanced_r11": round(new_r11_prec, 2), "change": round(new_r11_prec - base_r11_prec, 2)},
        {"metric": "Macro Recall", "baseline_r11": round(base_r11_rec, 2), "class_balanced_r11": round(new_r11_rec, 2), "change": round(new_r11_rec - base_r11_rec, 2)},
        {"metric": "Macro F1", "baseline_r11": round(base_r11_f1, 2), "class_balanced_r11": round(new_r11_f1, 2), "change": round(new_r11_f1 - base_r11_f1, 2)},
        {"metric": "REST Recall", "baseline_r11": round(base_rep["REST"]["recall"] * 100, 2), "class_balanced_r11": round(new_rep["REST"]["recall"] * 100, 2), "change": round((new_rep["REST"]["recall"] - base_rep["REST"]["recall"]) * 100, 2)},
        {"metric": "LEFT_FIST Recall", "baseline_r11": round(base_rep["LEFT_FIST"]["recall"] * 100, 2), "class_balanced_r11": round(new_rep["LEFT_FIST"]["recall"] * 100, 2), "change": round((new_rep["LEFT_FIST"]["recall"] - base_rep["LEFT_FIST"]["recall"]) * 100, 2)},
        {"metric": "RIGHT_FIST Recall", "baseline_r11": round(base_rep["RIGHT_FIST"]["recall"] * 100, 2), "class_balanced_r11": round(new_rep["RIGHT_FIST"]["recall"] * 100, 2), "change": round((new_rep["RIGHT_FIST"]["recall"] - base_rep["RIGHT_FIST"]["recall"]) * 100, 2)},
        {"metric": "REST Prediction Rate", "baseline_r11": round(base_dist.get("REST", 0) * 100, 2), "class_balanced_r11": round(new_dist.get("REST", 0) * 100, 2), "change": round((new_dist.get("REST", 0) - base_dist.get("REST", 0)) * 100, 2)},
    ]
    df_comp = pd.DataFrame(comparison_data)
    df_comp.to_csv(os.path.join(out_dir, "baseline_comparison.csv"), index=False)

    cm_new = confusion_matrix(all_new_r11_actuals, all_new_r11_raw_preds, labels=EXPECTED_CLASSES)
    df_cm = pd.DataFrame(cm_new, index=[f"True_{c}" for c in EXPECTED_CLASSES], columns=[f"Pred_{c}" for c in EXPECTED_CLASSES])
    df_cm.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

    # Subject statistics summary across S001-S010
    accs = [r["new_accuracy"] for r in subject_r11_results]
    f1s = [r["macro_f1"] for r in subject_r11_results]
    left_recs = [r["left_recall"] for r in subject_r11_results]
    right_recs = [r["right_recall"] for r in subject_r11_results]
    rest_recs = [r["rest_recall"] for r in subject_r11_results]

    summary_stats = {
        "accuracy": {"mean": round(np.mean(accs), 2), "median": round(np.median(accs), 2), "min": round(np.min(accs), 2), "max": round(np.max(accs), 2), "std": round(np.std(accs), 2)},
        "macro_f1": {"mean": round(np.mean(f1s), 2), "median": round(np.median(f1s), 2), "min": round(np.min(f1s), 2), "max": round(np.max(f1s), 2), "std": round(np.std(f1s), 2)},
        "left_recall": {"mean": round(np.mean(left_recs), 2), "median": round(np.median(left_recs), 2), "min": round(np.min(left_recs), 2), "max": round(np.max(left_recs), 2), "std": round(np.std(left_recs), 2)},
        "right_recall": {"mean": round(np.mean(right_recs), 2), "median": round(np.median(right_recs), 2), "min": round(np.min(right_recs), 2), "max": round(np.max(right_recs), 2), "std": round(np.std(right_recs), 2)},
        "rest_recall": {"mean": round(np.mean(rest_recs), 2), "median": round(np.median(rest_recs), 2), "min": round(np.min(rest_recs), 2), "max": round(np.max(rest_recs), 2), "std": round(np.std(rest_recs), 2)},
    }

    # Final Decision Calculation
    # Prefer higher Macro F1, better LEFT/RIGHT recall, reasonable REST rate while preserving accuracy
    passed_criteria = (new_r11_f1 >= base_r11_f1) and (new_rep["LEFT_FIST"]["recall"] >= base_rep["LEFT_FIST"]["recall"]) and (new_rep["RIGHT_FIST"]["recall"] >= base_rep["RIGHT_FIST"]["recall"])
    final_decision = "ADOPT CLASS-BALANCED MODEL" if passed_criteria else "KEEP BASELINE"

    summary_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "final_decision": final_decision,
        "baseline_r11_metrics": {
            "accuracy": round(base_r11_acc, 2),
            "macro_f1": round(base_r11_f1, 2),
            "rest_prediction_pct": round(base_dist.get("REST", 0) * 100, 2),
            "left_fist_recall": round(base_rep["LEFT_FIST"]["recall"] * 100, 2),
            "right_fist_recall": round(base_rep["RIGHT_FIST"]["recall"] * 100, 2),
            "rest_recall": round(base_rep["REST"]["recall"] * 100, 2),
        },
        "new_class_balanced_r11_metrics": {
            "accuracy": round(new_r11_acc, 2),
            "macro_f1": round(new_r11_f1, 2),
            "rest_prediction_pct": round(new_dist.get("REST", 0) * 100, 2),
            "left_fist_recall": round(new_rep["LEFT_FIST"]["recall"] * 100, 2),
            "right_fist_recall": round(new_rep["RIGHT_FIST"]["recall"] * 100, 2),
            "rest_recall": round(new_rep["REST"]["recall"] * 100, 2),
        },
        "improvements": {
            "accuracy_change": round(new_r11_acc - base_r11_acc, 2),
            "macro_f1_change": round(new_r11_f1 - base_r11_f1, 2),
            "left_fist_recall_change": round((new_rep["LEFT_FIST"]["recall"] - base_rep["LEFT_FIST"]["recall"]) * 100, 2),
            "right_fist_recall_change": round((new_rep["RIGHT_FIST"]["recall"] - base_rep["RIGHT_FIST"]["recall"]) * 100, 2),
            "rest_bias_reduction_pct": round((base_dist.get("REST", 0) - new_dist.get("REST", 0)) * 100, 2),
        },
        "raw_prediction_distribution": {
            "REST": round(new_dist.get("REST", 0) * 100, 2),
            "LEFT_FIST": round(new_dist.get("LEFT_FIST", 0) * 100, 2),
            "RIGHT_FIST": round(new_dist.get("RIGHT_FIST", 0) * 100, 2),
        },
        "smoothed_prediction_distribution_3win": {
            "REST": round(new_smooth3_dist.get("REST", 0) * 100, 2),
            "LEFT_FIST": round(new_smooth3_dist.get("LEFT_FIST", 0) * 100, 2),
            "RIGHT_FIST": round(new_smooth3_dist.get("RIGHT_FIST", 0) * 100, 2),
        },
        "subject_statistics": summary_stats
    }

    with open(os.path.join(out_dir, "summary_report.json"), "w") as f:
        json.dump(summary_report, f, indent=2)

    print("Step 13B Experiments completed successfully.")
    print(f"Final Decision: {final_decision}")
    print(f"Baseline R11 Acc: {base_r11_acc:.2f}%, F1: {base_r11_f1:.2f}% | New R11 Acc: {new_r11_acc:.2f}%, F1: {new_r11_f1:.2f}%")
    return summary_report

if __name__ == "__main__":
    run_step13b_class_balanced_experiments()
