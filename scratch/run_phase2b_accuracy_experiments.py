"""
NeuroOS Phase 2B — Motor Imagery Accuracy Research Experiments
Author: Lead EEG Machine Learning Researcher, NeuroOS Core Team
Strict Zero-Leakage Protocol:
- All hyperparameter tuning, feature selection, and model comparisons are evaluated
  using 5-Fold Stratified Cross-Validation on Development Calibration data (R03 + R07).
- Untouched holdout (R11) is evaluated ONLY on the frozen baselines (Step 60, Step 61)
  and validated candidate champions to verify final held-out generalization.
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
from scipy import signal
from collections import Counter
import mne
mne.set_log_level('WARNING')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from mne.decoding import CSP

from src.step8_pipeline import find_subject_runs
from src.step14_pipeline import extract_event_aligned_windows
from src.step33_pipeline import GROUP_B_CHANNELS, CLASSES, C3_IDX, C4_IDX, CZ_IDX, FC3_IDX, FC4_IDX
from src.step59_pipeline import (
    estimate_subject_peak_frequency,
    build_subject_adaptive_filter_bank,
    extract_custom_filter_bank_windows,
    extract_fbcsp_features_transform,
    align_probs
)
from src.step61_pipeline import extract_asymmetry_and_erd_features, load_dataset_step61


# ==============================================================================
# SPATIAL FILTERING & PREPROCESSING UTILITIES
# ==============================================================================

def apply_common_average_reference(X: np.ndarray) -> np.ndarray:
    """
    Applies Common Average Referencing (CAR) across channels:
    X_car[c, t] = X[c, t] - mean_k(X[k, t])
    """
    mean_ch = np.mean(X, axis=1, keepdims=True)
    return X - mean_ch


def apply_surface_laplacian_approx(X: np.ndarray) -> np.ndarray:
    """
    Applies local spatial derivation around key motor channels (C3, C4, Cz).
    Group B: FC5, FC3, FC1, FCz, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, C6, CPz
    """
    X_out = X.copy()
    c3_neighbors = [1, 7, 9]    # FC3, C5, C1
    c4_neighbors = [5, 11, 13]   # FC4, C2, C6
    cz_neighbors = [3, 9, 11]    # FCz, C1, C2

    X_out[:, C3_IDX, :] = X[:, C3_IDX, :] - np.mean(X[:, c3_neighbors, :], axis=1)
    X_out[:, C4_IDX, :] = X[:, C4_IDX, :] - np.mean(X[:, c4_neighbors, :], axis=1)
    X_out[:, CZ_IDX, :] = X[:, CZ_IDX, :] - np.mean(X[:, cz_neighbors, :], axis=1)
    return X_out


def safe_extract_fbcsp_features_train(filt_dict: dict, y_tr: np.ndarray, n_components: int = 4) -> tuple:
    """
    Fits CSP per band using Ledoit-Wolf regularization to prevent rank-deficiency
    singularities when spatial referencing (CAR / Laplacian) reduces channel degrees of freedom.
    """
    fb_tr = []
    csp_dict = {}
    for b_name, X_tr_b in filt_dict.items():
        try:
            csp = CSP(n_components=n_components, reg=0.01, log=True, norm_trace=False)
            X_tr_csp = csp.fit_transform(X_tr_b, y_tr)
        except Exception:
            csp = CSP(n_components=n_components, reg=0.05, log=True, norm_trace=False)
            X_tr_csp = csp.fit_transform(X_tr_b, y_tr)
        fb_tr.append(X_tr_csp)
        csp_dict[b_name] = csp
    return np.hstack(fb_tr), csp_dict


def extract_enhanced_erd_asymmetry(X: np.ndarray, sfreq: float = 160.0) -> np.ndarray:
    """
    Extracts frequency-resolved Mu (8-13 Hz) and Beta (13-30 Hz) ERD asymmetry
    specifically contrasting contralateral C3 vs C4 power ratios.
    """
    N, C, T = X.shape
    feats = []
    
    nyq = sfreq / 2.0
    b_mu, a_mu = signal.butter(3, [8.0 / nyq, 13.0 / nyq], btype='band')
    b_beta, a_beta = signal.butter(3, [13.0 / nyq, 30.0 / nyq], btype='band')
    
    X_mu = signal.filtfilt(b_mu, a_mu, X, axis=-1)
    X_beta = signal.filtfilt(b_beta, a_beta, X, axis=-1)

    for i in range(N):
        win_raw = X[i]
        win_mu = X_mu[i]
        win_beta = X_beta[i]

        v_raw = np.var(win_raw, axis=-1)
        v_mu = np.var(win_mu, axis=-1)
        v_beta = np.var(win_beta, axis=-1)

        c3_raw, c4_raw = v_raw[C3_IDX], v_raw[C4_IDX]
        c3_mu, c4_mu = v_mu[C3_IDX], v_mu[C4_IDX]
        c3_beta, c4_beta = v_beta[C3_IDX], v_beta[C4_IDX]

        fc3_raw, fc4_raw = v_raw[FC3_IDX], v_raw[FC4_IDX]

        asym_motor_raw = np.log(c4_raw + 1e-8) - np.log(c3_raw + 1e-8)
        asym_premotor_raw = np.log(fc4_raw + 1e-8) - np.log(fc3_raw + 1e-8)
        
        asym_mu = (c4_mu - c3_mu) / (c4_mu + c3_mu + 1e-8)
        asym_beta = (c4_beta - c3_beta) / (c4_beta + c3_beta + 1e-8)
        
        mu_beta_c3 = c3_mu / (c3_beta + 1e-8)
        mu_beta_c4 = c4_mu / (c4_beta + 1e-8)

        row = list(v_raw) + [
            asym_motor_raw, asym_premotor_raw,
            asym_mu, asym_beta,
            mu_beta_c3, mu_beta_c4
        ]
        feats.append(row)

    return np.array(feats, dtype=np.float32)


# ==============================================================================
# CROSS-VALIDATION PIPELINE ENGINE (DEVELOPMENT SET ONLY)
# ==============================================================================

def run_dev_cross_validation(
    sub_data: dict,
    spatial_prep: str = "raw",  # "raw", "car", "laplacian"
    feature_family: str = "fused",  # "fbcsp", "riemannian", "asym", "fused", "enhanced_fused"
    classifier_type: str = "reg_lda",  # "reg_lda", "lr_balanced", "ridge_balanced", "svm_linear", "svm_rbf", "ensemble_balanced"
    k_features: int = 24,
    fbcsp_components: int = 4,
    use_calibrated_threshold: bool = True
) -> dict:
    """
    Executes 5-Fold Stratified Cross-Validation on Development Set (R03 + R07)
    strictly within each subject and aggregates performance.
    """
    subs = sorted(list(sub_data.keys()))
    sub_cv_results = []
    all_y_true = []
    all_y_pred = []
    all_conf = []

    for s_id in subs:
        X_cal = sub_data[s_id]["X_calib"].copy()
        y_cal = sub_data[s_id]["y_calib"].copy()

        # 1. Spatial Preprocessing
        if spatial_prep == "car":
            X_cal = apply_common_average_reference(X_cal)
        elif spatial_prep == "laplacian":
            X_cal = apply_surface_laplacian_approx(X_cal)

        # Estimate subject peak frequencies from calibration data
        freq_info = estimate_subject_peak_frequency(X_cal)
        bank_s = build_subject_adaptive_filter_bank(freq_info["mu_peak_hz"], freq_info["beta_peak_hz"])

        # Precompute band-filtered windows and asymmetry for efficiency
        # (Filtering does not depend on fold labels; saves 95% compute)
        filt_cal_dict = extract_custom_filter_bank_windows(X_cal, bank_s)
        
        if feature_family == "enhanced_fused":
            X_asym_all = extract_enhanced_erd_asymmetry(X_cal)
        elif feature_family in ["asym", "fused"]:
            X_asym_all = extract_asymmetry_and_erd_features(X_cal)
        else:
            X_asym_all = None

        # 5-Fold Stratified CV on Development Data
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        s_preds = []
        s_trues = []
        s_confs = []

        for train_idx, val_idx in skf.split(X_cal, y_cal):
            X_tr, y_tr = X_cal[train_idx], y_cal[train_idx]
            X_va, y_va = X_cal[val_idx], y_cal[val_idx]

            feats_tr_list = []
            feats_va_list = []

            # A. FBCSP (Spatial filters learned ONLY on fold train)
            if feature_family in ["fbcsp", "fused", "enhanced_fused"]:
                filt_tr = {b: filt_cal_dict[b][train_idx] for b in filt_cal_dict}
                filt_va = {b: filt_cal_dict[b][val_idx] for b in filt_cal_dict}
                X_fbcsp_tr, csp_d = safe_extract_fbcsp_features_train(filt_tr, y_tr, n_components=fbcsp_components)
                X_fbcsp_va = extract_fbcsp_features_transform(filt_va, csp_d)
                feats_tr_list.append(X_fbcsp_tr)
                feats_va_list.append(X_fbcsp_va)

            # B. Riemannian Tangent Space (Fréchet mean learned ONLY on fold train)
            if feature_family in ["riemannian", "fused", "enhanced_fused"]:
                cov_est = Covariances(estimator='lwf')
                ts_trans = TangentSpace(metric='riemann')
                cov_tr = cov_est.fit_transform(X_tr)
                cov_va = cov_est.transform(X_va)
                X_ts_tr = ts_trans.fit_transform(cov_tr)
                X_ts_va = ts_trans.transform(cov_va)
                feats_tr_list.append(X_ts_tr)
                feats_va_list.append(X_ts_va)

            # C. Asymmetry Features
            if X_asym_all is not None:
                feats_tr_list.append(X_asym_all[train_idx])
                feats_va_list.append(X_asym_all[val_idx])

            X_fused_tr = np.hstack(feats_tr_list)
            X_fused_va = np.hstack(feats_va_list)

            # Feature Selection (Selected ONLY on fold train)
            k_sel = min(k_features, X_fused_tr.shape[1])
            sel = SelectKBest(f_classif, k=k_sel).fit(X_fused_tr, y_tr)
            X_sel_tr = sel.transform(X_fused_tr)
            X_sel_va = sel.transform(X_fused_va)

            # Scaling (Standardized ONLY on fold train)
            sc = StandardScaler().fit(X_sel_tr)
            X_sc_tr = sc.transform(X_sel_tr)
            X_sc_va = sc.transform(X_sel_va)

            # Classifier Fitting
            if classifier_type == "reg_lda":
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            elif classifier_type == "lr_balanced":
                clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=500, random_state=42)
            elif classifier_type == "ridge_balanced":
                clf = RidgeClassifier(alpha=1.0, class_weight='balanced', random_state=42)
            elif classifier_type == "svm_linear":
                clf = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42)
            elif classifier_type == "svm_rbf":
                clf = SVC(kernel='rbf', C=1.8, gamma='scale', class_weight='balanced', probability=True, random_state=42)
            elif classifier_type == "ensemble_balanced":
                lda_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                lr_clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, random_state=42)
                svm_clf = SVC(kernel='rbf', C=1.8, gamma='scale', class_weight='balanced', probability=True, random_state=42)
                clf = VotingClassifier(
                    estimators=[('lda', lda_clf), ('lr', lr_clf), ('svm', svm_clf)],
                    voting='soft',
                    weights=[1.5, 1.2, 1.0]
                )
            else:
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

            clf.fit(X_sc_tr, y_tr)

            # Prediction & Confidence
            if hasattr(clf, "predict_proba"):
                probs_raw = clf.predict_proba(X_sc_va)
                probs = align_probs(probs_raw, clf.classes_)
                
                for p_row in probs:
                    conf = float(np.max(p_row))
                    if use_calibrated_threshold and p_row[0] >= 0.58:
                        pred_l = "REST"
                    else:
                        pred_l = CLASSES[int(np.argmax(p_row))]
                    s_preds.append(pred_l)
                    s_confs.append(conf)
            else:
                preds_raw = clf.predict(X_sc_va)
                s_preds.extend(preds_raw)
                s_confs.extend([0.8] * len(preds_raw))

            s_trues.extend(y_va)

        acc_s = accuracy_score(s_trues, s_preds) * 100.0
        f1_s = f1_score(s_trues, s_preds, average='macro') * 100.0
        sub_cv_results.append({"subject": s_id, "acc": acc_s, "f1": f1_s})
        all_y_true.extend(s_trues)
        all_y_pred.extend(s_preds)
        all_conf.extend(s_confs)

    total_acc = accuracy_score(all_y_true, all_y_pred) * 100.0
    total_f1 = f1_score(all_y_true, all_y_pred, average='macro') * 100.0
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASSES)
    
    rest_rec = cm[0, 0] / np.sum(cm[0, :]) * 100.0 if np.sum(cm[0, :]) > 0 else 0
    left_rec = cm[1, 1] / np.sum(cm[1, :]) * 100.0 if np.sum(cm[1, :]) > 0 else 0
    right_rec = cm[2, 2] / np.sum(cm[2, :]) * 100.0 if np.sum(cm[2, :]) > 0 else 0

    return {
        "spatial_prep": spatial_prep,
        "feature_family": feature_family,
        "classifier_type": classifier_type,
        "k_features": k_features,
        "fbcsp_components": fbcsp_components,
        "cv_accuracy": round(total_acc, 2),
        "cv_macro_f1": round(total_f1, 2),
        "rest_recall": round(rest_rec, 2),
        "left_recall": round(left_rec, 2),
        "right_recall": round(right_rec, 2),
        "sub_mean_acc": round(float(np.mean([x['acc'] for x in sub_cv_results])), 2),
        "sub_std_acc": round(float(np.std([x['acc'] for x in sub_cv_results])), 2),
        "confusion_matrix": cm.tolist()
    }


# ==============================================================================
# MAIN CONTROLLED EXPERIMENT ORCHESTRATION
# ==============================================================================

def run_all_experiments():
    print("=" * 75, flush=True)
    print("NEUROOS PHASE 2B: CONTROLLED MOTOR-IMAGERY ACCURACY EXPERIMENTS", flush=True)
    print("Strict Zero-Leakage: Development CV on R03+R07 (N=600)", flush=True)
    print("=" * 75, flush=True)

    sub_data = load_dataset_step61(data_dir="data")
    experiment_results = []

    # --------------------------------------------------------------------------
    # GROUP A: SPATIAL FILTERING ABLATION
    # --------------------------------------------------------------------------
    print("\n--- GROUP A: SPATIAL FILTERING ABLATION (5-Fold Dev CV) ---", flush=True)
    for prep in ["raw", "car", "laplacian"]:
        t0 = time.time()
        res = run_dev_cross_validation(
            sub_data,
            spatial_prep=prep,
            feature_family="fused",
            classifier_type="reg_lda",
            k_features=24
        )
        dur = round(time.time() - t0, 1)
        res["group"] = "Group A: Spatial Referencing"
        res["duration_sec"] = dur
        experiment_results.append(res)
        print(f"[{prep.upper()}] CV Acc: {res['cv_accuracy']:.2f}%, F1: {res['cv_macro_f1']:.2f}% | Rest: {res['rest_recall']:.2f}%, Left: {res['left_recall']:.2f}%, Right: {res['right_recall']:.2f}% ({dur}s)", flush=True)

    # --------------------------------------------------------------------------
    # GROUP B: FEATURE FAMILY CONTRIBUTIONS & ENHANCED ERD ASYMMETRY
    # --------------------------------------------------------------------------
    print("\n--- GROUP B: FEATURE FAMILY ABLATIONS & ASYMMETRY (5-Fold Dev CV) ---", flush=True)
    best_spatial = "car"  # evaluate feature families on CAR
    for feats in ["fbcsp", "riemannian", "asym", "fused", "enhanced_fused"]:
        t0 = time.time()
        res = run_dev_cross_validation(
            sub_data,
            spatial_prep=best_spatial,
            feature_family=feats,
            classifier_type="reg_lda",
            k_features=24
        )
        dur = round(time.time() - t0, 1)
        res["group"] = "Group B: Feature Architecture"
        res["duration_sec"] = dur
        experiment_results.append(res)
        print(f"[{feats}] CV Acc: {res['cv_accuracy']:.2f}%, F1: {res['cv_macro_f1']:.2f}% | Rest: {res['rest_recall']:.2f}%, Left: {res['left_recall']:.2f}%, Right: {res['right_recall']:.2f}% ({dur}s)", flush=True)

    # --------------------------------------------------------------------------
    # GROUP C: CLASSIFIER ARCHITECTURES & BALANCED CLASS WEIGHTING
    # --------------------------------------------------------------------------
    print("\n--- GROUP C: CLASSIFIER ARCHITECTURES & BALANCED WEIGHTING (5-Fold Dev CV) ---", flush=True)
    for clf_name in ["reg_lda", "lr_balanced", "svm_linear", "svm_rbf", "ensemble_balanced"]:
        t0 = time.time()
        res = run_dev_cross_validation(
            sub_data,
            spatial_prep=best_spatial,
            feature_family="enhanced_fused",
            classifier_type=clf_name,
            k_features=24
        )
        dur = round(time.time() - t0, 1)
        res["group"] = "Group C: Classifier Selection"
        res["duration_sec"] = dur
        experiment_results.append(res)
        print(f"[{clf_name}] CV Acc: {res['cv_accuracy']:.2f}%, F1: {res['cv_macro_f1']:.2f}% | Rest: {res['rest_recall']:.2f}%, Left: {res['left_recall']:.2f}%, Right: {res['right_recall']:.2f}% ({dur}s)", flush=True)

    # --------------------------------------------------------------------------
    # GROUP D: FEATURE DIMENSIONALITY SWEEP
    # --------------------------------------------------------------------------
    print("\n--- GROUP D: FEATURE SELECTION SWEEP K (5-Fold Dev CV) ---", flush=True)
    for k_val in [16, 20, 24, 28, 36]:
        t0 = time.time()
        res = run_dev_cross_validation(
            sub_data,
            spatial_prep=best_spatial,
            feature_family="enhanced_fused",
            classifier_type="ensemble_balanced",
            k_features=k_val
        )
        dur = round(time.time() - t0, 1)
        res["group"] = "Group D: K-Feature Selection"
        res["duration_sec"] = dur
        experiment_results.append(res)
        print(f"[K={k_val}] CV Acc: {res['cv_accuracy']:.2f}%, F1: {res['cv_macro_f1']:.2f}% | Rest: {res['rest_recall']:.2f}%, Left: {res['left_recall']:.2f}%, Right: {res['right_recall']:.2f}% ({dur}s)", flush=True)

    # Save Development Results
    df_res = pd.DataFrame(experiment_results)
    csv_path = "scratch/phase2b_experiments.csv"
    json_path = "scratch/phase2b_experiments.json"
    df_res.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nAll development experiments completed. Saved to {csv_path} and {json_path}", flush=True)
    return df_res

if __name__ == "__main__":
    run_all_experiments()
