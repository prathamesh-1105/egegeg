"""
NeuroOS Phase 6 — Large-Scale EEG Dataset Expansion & Left/Right Accuracy Improvement
Research & Experimentation Suite

Comprehensive, scientifically defensible evaluation:
1. Exact audit and reproduction of baseline on Motor Execution (R11).
2. Domain Shift Evaluation: Phase 5 champion tested zero-shot on Pure Motor Imagery (R12).
3. Strategy A: Pure Motor Imagery Within-Task Training (R04+R08 -> R12).
   - 50-fold Stratified CV on Development (N=600 trials).
   - Holdout evaluation on R12 (N=300 trials).
   - Candidates:
     * P6-Cand-01: Baseline Exp 08 Anatomical Laplacian (k=28)
     * P6-Cand-02: Advanced Motor Features (k=32)
     * P6-Cand-03: Contralateral Asymmetry + Bilateral Differentials (k=36)
     * P6-Cand-04: Calibrated Two-Stage Hierarchical BCI
     * P6-Cand-05: Multi-Task Joint Training (Execution R03+R07 + Imagery R04+R08, N=1,200 trials)
4. Strategy B: Cross-Task / Domain Transfer (Execution R03+R07 -> Imagery R12).
5. Strategy C: Few-shot Subject Calibration curve on Motor Imagery (0, 5, 10, 15, 20 trials/class).
6. Serializes all metrics into docs/phase6_model_comparison.json.
"""

import os
import sys
import time
import json
import joblib
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy.signal as signal
from collections import Counter
from typing import Dict, List, Tuple, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from src.step14_pipeline import extract_event_aligned_windows
from src.step8_pipeline import find_subject_runs
from src.neuro_os.dataset_pipeline import (
    TARGET_MONTAGE,
    TARGET_SFREQ,
    TARGET_WINDOW_SEC,
    apply_anatomical_surface_laplacian,
    extract_advanced_motor_features as extract_p6_motor_features
)

TRUE_FC3_IDX = 0
TRUE_FC1_IDX = 1
TRUE_FCZ_IDX = 2
TRUE_FC2_IDX = 3
TRUE_FC4_IDX = 4
TRUE_C3_IDX  = 5
TRUE_C1_IDX  = 6
TRUE_CZ_IDX  = 7
TRUE_C2_IDX  = 8
TRUE_C4_IDX  = 9
TRUE_CP3_IDX = 10
TRUE_CP1_IDX = 11
TRUE_CPZ_IDX = 12
TRUE_CP2_IDX = 13
TRUE_CP4_IDX = 14

CLASSES = ["REST", "LEFT_FIST", "RIGHT_FIST"]
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ==============================================================================
# 1. Feature Extraction Routines
# ==============================================================================

def extract_lateralized_spectral_features(X: np.ndarray, sfreq: float = 160.0) -> np.ndarray:
    """
    Extracts high-resolution contralateral motor imagery features:
    - Raw channel log-variances (15 channels).
    - Mu-band (8-13 Hz) log-variances and contralateral asymmetry: log(C4) - log(C3), log(FC4) - log(FC3).
    - Beta-band (13-30 Hz) log-variances and contralateral asymmetry: log(C4) - log(C3), log(CP4) - log(CP3).
    - Normalized ERD asymmetry: (C4 - C3) / (C4 + C3) in Mu and Beta.
    - Contralateral Bipolar Differentials: (C3 - C1) vs (C4 - C2), (FC3 - CP3) vs (FC4 - CP4).
    - Vertex suppression ratio: (C3 + C4) / (2 * Cz).
    """
    N, C, T = X.shape
    feats = []

    nyq = sfreq / 2.0
    b_mu, a_mu = signal.butter(3, [8.0 / nyq, 13.0 / nyq], btype='band')
    b_beta, a_beta = signal.butter(3, [13.0 / nyq, 30.0 / nyq], btype='band')

    X_mu = signal.filtfilt(b_mu, a_mu, X, axis=-1)
    X_beta = signal.filtfilt(b_beta, a_beta, X, axis=-1)

    for i in range(N):
        v_raw = np.var(X[i], axis=-1)
        v_mu = np.var(X_mu[i], axis=-1)
        v_beta = np.var(X_beta[i], axis=-1)

        c3_r, c4_r = v_raw[TRUE_C3_IDX], v_raw[TRUE_C4_IDX]
        c3_mu, c4_mu = v_mu[TRUE_C3_IDX], v_mu[TRUE_C4_IDX]
        c3_beta, c4_beta = v_beta[TRUE_C3_IDX], v_beta[TRUE_C4_IDX]

        fc3_r, fc4_r = v_raw[TRUE_FC3_IDX], v_raw[TRUE_FC4_IDX]
        fc3_mu, fc4_mu = v_mu[TRUE_FC3_IDX], v_mu[TRUE_FC4_IDX]
        cp3_beta, cp4_beta = v_beta[TRUE_CP3_IDX], v_beta[TRUE_CP4_IDX]

        c1_r, c2_r = v_raw[TRUE_C1_IDX], v_raw[TRUE_C2_IDX]
        cz_r = v_raw[TRUE_CZ_IDX]

        # Contralateral Log Asymmetry
        asym_motor_raw = np.log(c4_r + 1e-8) - np.log(c3_r + 1e-8)
        asym_motor_mu = np.log(c4_mu + 1e-8) - np.log(c3_mu + 1e-8)
        asym_motor_beta = np.log(c4_beta + 1e-8) - np.log(c3_beta + 1e-8)
        asym_premotor_mu = np.log(fc4_mu + 1e-8) - np.log(fc3_mu + 1e-8)
        asym_cp_beta = np.log(cp4_beta + 1e-8) - np.log(cp3_beta + 1e-8)

        # Normalized ERD
        norm_mu = (c4_mu - c3_mu) / (c4_mu + c3_mu + 1e-8)
        norm_beta = (c4_beta - c3_beta) / (c4_beta + c3_beta + 1e-8)

        # Contralateral Bipolar Differentials
        bipolar_motor_left = c3_r - c1_r
        bipolar_motor_right = c4_r - c2_r
        bipolar_motor_diff = bipolar_motor_right - bipolar_motor_left

        bipolar_ap_left = fc3_r - v_raw[TRUE_CP3_IDX]
        bipolar_ap_right = fc4_r - v_raw[TRUE_CP4_IDX]
        bipolar_ap_diff = bipolar_ap_right - bipolar_ap_left

        # Central suppression
        central_supp = (c3_r + c4_r) / (2.0 * cz_r + 1e-8)

        row = list(v_raw) + [
            asym_motor_raw,
            asym_motor_mu,
            asym_motor_beta,
            asym_premotor_mu,
            asym_cp_beta,
            norm_mu,
            norm_beta,
            bipolar_motor_diff,
            bipolar_ap_diff,
            central_supp
        ]
        feats.append(row)

    return np.array(feats, dtype=np.float32)


# ==============================================================================
# 2. Calibrated Hierarchical Classifier
# ==============================================================================

class CalibratedHierarchicalBCI:
    """
    Two-Stage Hierarchical BCI Classifier:
    - Stage 1: Binary REST vs. ACTIVE intent detector (Tuned out-of-fold for >= 90% REST specificity).
    - Stage 2: Pure Binary LEFT vs. RIGHT classifier trained strictly on active motor trials.
    """
    def __init__(self, target_rest_recall: float = 0.90):
        self.stage1_clf = VotingClassifier(
            estimators=[
                ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
                ('lr', LogisticRegression(C=1.0, max_iter=500, class_weight='balanced', random_state=42)),
                ('svm', SVC(kernel='rbf', C=1.5, gamma='scale', probability=True, class_weight='balanced', random_state=42))
            ],
            voting='soft'
        )
        self.stage2_clf = VotingClassifier(
            estimators=[
                ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
                ('lr', LogisticRegression(C=1.5, max_iter=500, class_weight='balanced', random_state=42)),
                ('svm', SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, class_weight='balanced', random_state=42))
            ],
            voting='soft'
        )
        self.target_rest_recall = target_rest_recall
        self.calibrated_rest_threshold = 0.55
        self.classes_ = np.array(["REST", "LEFT_FIST", "RIGHT_FIST"])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, oof_tune: bool = True):
        y_binary = np.array(["REST" if label == "REST" else "ACTIVE" for label in y_train])
        self.stage1_clf.fit(X_train, y_binary)

        active_mask = (y_train != "REST")
        if np.sum(active_mask) > 0:
            X_active = X_train[active_mask]
            y_active = y_train[active_mask]
            self.stage2_clf.fit(X_active, y_active)

        if oof_tune and np.sum(y_binary == "REST") >= 10:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            oof_p_rest = []
            oof_y_true = []
            for tr_idx, val_idx in skf.split(X_train, y_binary):
                fold_clf = VotingClassifier(
                    estimators=[
                        ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
                        ('lr', LogisticRegression(C=1.0, max_iter=500, class_weight='balanced', random_state=42)),
                        ('svm', SVC(kernel='rbf', C=1.5, gamma='scale', probability=True, class_weight='balanced', random_state=42))
                    ],
                    voting='soft'
                )
                fold_clf.fit(X_train[tr_idx], y_binary[tr_idx])
                probs = fold_clf.predict_proba(X_train[val_idx])
                rest_idx = list(fold_clf.classes_).index("REST")
                oof_p_rest.extend(probs[:, rest_idx])
                oof_y_true.extend(y_binary[val_idx])

            oof_p_rest = np.array(oof_p_rest)
            oof_y_true = np.array(oof_y_true)
            rest_mask = (oof_y_true == "REST")
            if np.sum(rest_mask) > 0:
                rest_scores = oof_p_rest[rest_mask]
                self.calibrated_rest_threshold = float(np.percentile(rest_scores, (1.0 - self.target_rest_recall) * 100))
                self.calibrated_rest_threshold = max(0.40, min(0.75, self.calibrated_rest_threshold))

        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        s1_probs = self.stage1_clf.predict_proba(X_test)
        s1_classes = list(self.stage1_clf.classes_)
        rest_col = s1_classes.index("REST")
        p_rest_raw = s1_probs[:, rest_col]

        s2_probs = self.stage2_clf.predict_proba(X_test)
        s2_classes = list(self.stage2_clf.classes_)
        left_col = s2_classes.index("LEFT_FIST")
        right_col = s2_classes.index("RIGHT_FIST")
        p_left_given_active = s2_probs[:, left_col]
        p_right_given_active = s2_probs[:, right_col]

        combined_probs = np.zeros((len(X_test), 3), dtype=np.float32)
        for i in range(len(X_test)):
            pr = p_rest_raw[i]
            if pr >= self.calibrated_rest_threshold:
                p_rest_scaled = 0.50 + 0.50 * ((pr - self.calibrated_rest_threshold) / (1.0 - self.calibrated_rest_threshold + 1e-6))
                rem = max(0.0, 1.0 - p_rest_scaled)
                combined_probs[i, 0] = p_rest_scaled
                combined_probs[i, 1] = rem * p_left_given_active[i]
                combined_probs[i, 2] = rem * p_right_given_active[i]
            else:
                p_active_scaled = 0.50 + 0.50 * ((self.calibrated_rest_threshold - pr) / (self.calibrated_rest_threshold + 1e-6))
                p_rest_scaled = max(0.0, 1.0 - p_active_scaled)
                combined_probs[i, 0] = p_rest_scaled
                combined_probs[i, 1] = p_active_scaled * p_left_given_active[i]
                combined_probs[i, 2] = p_active_scaled * p_right_given_active[i]

            s = np.sum(combined_probs[i])
            if s > 0:
                combined_probs[i] /= s

        return combined_probs

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_test)
        preds_idx = np.argmax(probs, axis=1)
        return self.classes_[preds_idx]


# ==============================================================================
# 3. Helper Functions: Metrics & Pipelines
# ==============================================================================

def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    p_per = precision_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    r_per = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    f_per = f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

    # Lateralization confusion: Left predicted as Right + Right predicted as Left
    lr_confusion = int(cm[1, 2] + cm[2, 1])
    total_motor_trials = int(np.sum(cm[1, :]) + np.sum(cm[2, :]))
    lr_confusion_rate = round(lr_confusion / total_motor_trials, 4) if total_motor_trials > 0 else 0.0

    # REST False Actuation Rate: REST trials predicted as Left or Right
    total_rest = int(np.sum(cm[0, :]))
    rest_false_actuations = int(cm[0, 1] + cm[0, 2])
    rest_far = round(rest_false_actuations / total_rest, 4) if total_rest > 0 else 0.0

    return {
        "accuracy": round(float(acc) * 100, 2),
        "balanced_accuracy": round(float(bal_acc) * 100, 2),
        "macro_f1": round(float(macro_f1) * 100, 2),
        "rest_recall": round(float(r_per[0]) * 100, 2),
        "left_recall": round(float(r_per[1]) * 100, 2),
        "right_recall": round(float(r_per[2]) * 100, 2),
        "rest_precision": round(float(p_per[0]) * 100, 2),
        "left_precision": round(float(p_per[1]) * 100, 2),
        "right_precision": round(float(p_per[2]) * 100, 2),
        "rest_f1": round(float(f_per[0]) * 100, 2),
        "left_f1": round(float(f_per[1]) * 100, 2),
        "right_f1": round(float(f_per[2]) * 100, 2),
        "lr_confusion": lr_confusion,
        "lr_confusion_rate": lr_confusion_rate,
        "rest_false_actuations": rest_false_actuations,
        "rest_false_actuation_rate": rest_far,
        "confusion_matrix": cm.tolist()
    }


def fit_and_evaluate_fold(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    k_features: int = 32,
    use_hierarchical: bool = False,
    feature_type: str = "advanced"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fits full preprocessing, spatial filtering, feature selection, and ensemble strictly on training fold.
    Evaluates on test fold with zero data leakage.
    """
    # 1. Anatomical Surface Laplacian
    X_tr_lap = apply_anatomical_surface_laplacian(X_train_raw)
    X_te_lap = apply_anatomical_surface_laplacian(X_test_raw)

    # 2. Feature Extraction
    if feature_type == "lateralized":
        feats_tr_eng = extract_lateralized_spectral_features(X_tr_lap)
        feats_te_eng = extract_lateralized_spectral_features(X_te_lap)
    else:
        feats_tr_eng = extract_p6_motor_features(X_tr_lap)
        feats_te_eng = extract_p6_motor_features(X_te_lap)

    # 3. Covariance Tangent Space
    cov_est = Covariances(estimator='lwf')
    cov_tr = cov_est.fit_transform(X_tr_lap)
    cov_te = cov_est.transform(X_te_lap)

    ts = TangentSpace(metric='riemann')
    feats_tr_ts = ts.fit_transform(cov_tr)
    feats_te_ts = ts.transform(cov_te)

    # 4. Multiclass CSP
    csp = CSP(n_components=6, reg='ledoit_wolf', log=True, cov_est='epoch')
    feats_tr_csp = csp.fit_transform(X_tr_lap, y_train)
    feats_te_csp = csp.transform(X_te_lap)

    # 5. Concatenate Multimodal Features
    F_tr = np.hstack([feats_tr_eng, feats_tr_ts, feats_tr_csp])
    F_te = np.hstack([feats_te_eng, feats_te_ts, feats_te_csp])

    # 6. Feature Selection
    sel = SelectKBest(f_classif, k=min(k_features, F_tr.shape[1]))
    F_tr_sel = sel.fit_transform(F_tr, y_train)
    F_te_sel = sel.transform(F_te)

    # 7. Scaler
    scaler = StandardScaler()
    F_tr_scaled = scaler.fit_transform(F_tr_sel)
    F_te_scaled = scaler.transform(F_te_sel)

    # 8. Model Training
    if use_hierarchical:
        clf = CalibratedHierarchicalBCI(target_rest_recall=0.90)
        clf.fit(F_tr_scaled, y_train, oof_tune=True)
        probs = clf.predict_proba(F_te_scaled)
        preds = clf.predict(F_te_scaled)
    else:
        clf = VotingClassifier(
            estimators=[
                ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
                ('lr', LogisticRegression(C=1.2, max_iter=500, class_weight='balanced', random_state=42)),
                ('svm', SVC(kernel='rbf', C=1.5, gamma='scale', probability=True, class_weight='balanced', random_state=42))
            ],
            voting='soft'
        )
        clf.fit(F_tr_scaled, y_train)
        probs = clf.predict_proba(F_te_scaled)
        # Align class probabilities
        c_order = list(clf.classes_)
        aligned = np.zeros((len(F_te_scaled), 3), dtype=np.float32)
        for idx, c in enumerate(CLASSES):
            if c in c_order:
                aligned[:, idx] = probs[:, c_order.index(c)]
        probs = aligned
        preds = np.array([CLASSES[i] for i in np.argmax(probs, axis=1)])

    return preds, probs, {
        "scaler": scaler,
        "sel": sel,
        "csp": csp,
        "cov_est": cov_est,
        "ts": ts,
        "clf": clf
    }


# ==============================================================================
# 4. Main Experiment Suite
# ==============================================================================

def run_phase6_experiments():
    print("=" * 80)
    print("NEUROOS PHASE 6: LARGE-SCALE DATASET EXPANSION & ACCURACY SUITE")
    print("=" * 80)

    subjects = [f"S{i:03d}" for i in range(1, 11)]
    data_dir = "data"

    # --------------------------------------------------------------------------
    # Step 1: Load Datasets
    # --------------------------------------------------------------------------
    print("\n[Step 1] Loading Motor Execution (Task 1) and Pure Motor Imagery (Task 2) runs...")
    
    # Motor Execution: R03+R07 (Dev), R11 (Holdout)
    exec_dev_X, exec_dev_y, exec_dev_subs = [], [], []
    exec_eval_X, exec_eval_y, exec_eval_subs = [], [], []

    # Pure Motor Imagery: R04+R08 (Dev), R12 (Holdout)
    imag_dev_X, imag_dev_y, imag_dev_subs = [], [], []
    imag_eval_X, imag_eval_y, imag_eval_subs = [], [], []

    for sub in subjects:
        runs = find_subject_runs(sub, data_dir=data_dir)
        
        # Execution runs
        for r_id in ["R03", "R07"]:
            if r_id in runs:
                X_r, y_r, _ = extract_event_aligned_windows(runs[r_id], channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
                exec_dev_X.append(X_r)
                exec_dev_y.append(y_r)
                exec_dev_subs.extend([sub] * len(y_r))
        if "R11" in runs:
            X_r, y_r, _ = extract_event_aligned_windows(runs["R11"], channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
            exec_eval_X.append(X_r)
            exec_eval_y.append(y_r)
            exec_eval_subs.extend([sub] * len(y_r))

        # Pure Motor Imagery runs
        for r_id in ["R04", "R08"]:
            if r_id in runs:
                X_r, y_r, _ = extract_event_aligned_windows(runs[r_id], channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
                imag_dev_X.append(X_r)
                imag_dev_y.append(y_r)
                imag_dev_subs.extend([sub] * len(y_r))
        if "R12" in runs:
            X_r, y_r, _ = extract_event_aligned_windows(runs["R12"], channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
            imag_eval_X.append(X_r)
            imag_eval_y.append(y_r)
            imag_eval_subs.extend([sub] * len(y_r))

    X_exec_dev = np.concatenate(exec_dev_X, axis=0)
    y_exec_dev = np.concatenate(exec_dev_y, axis=0)
    X_exec_eval = np.concatenate(exec_eval_X, axis=0)
    y_exec_eval = np.concatenate(exec_eval_y, axis=0)

    X_imag_dev = np.concatenate(imag_dev_X, axis=0)
    y_imag_dev = np.concatenate(imag_dev_y, axis=0)
    X_imag_eval = np.concatenate(imag_eval_X, axis=0)
    y_imag_eval = np.concatenate(imag_eval_y, axis=0)

    print(f"  Motor Execution Dev (R03+R07):   {X_exec_dev.shape[0]} trials {Counter(y_exec_dev)}")
    print(f"  Motor Execution Holdout (R11):   {X_exec_eval.shape[0]} trials {Counter(y_exec_eval)}")
    print(f"  Pure Motor Imagery Dev (R04+R08): {X_imag_dev.shape[0]} trials {Counter(y_imag_dev)}")
    print(f"  Pure Motor Imagery Holdout (R12): {X_imag_eval.shape[0]} trials {Counter(y_imag_eval)}")
    print(f"  Total Grand Corpus: {len(y_exec_dev) + len(y_exec_eval) + len(y_imag_dev) + len(y_imag_eval)} real trials across 10 subjects.")

    results = {
        "metadata": {
            "execution_dev_trials": int(len(y_exec_dev)),
            "execution_eval_trials": int(len(y_exec_eval)),
            "imagery_dev_trials": int(len(y_imag_dev)),
            "imagery_eval_trials": int(len(y_imag_eval)),
            "total_corpus_trials": int(len(y_exec_dev) + len(y_exec_eval) + len(y_imag_dev) + len(y_imag_eval)),
            "subjects": subjects,
            "classes": CLASSES
        },
        "experiments": {}
    }

    # --------------------------------------------------------------------------
    # Step 2: Audit Phase 5 Baseline Reproduction
    # --------------------------------------------------------------------------
    print("\n[Step 2] Auditing Phase 5 Champion Baseline...")
    preds_p5_r11, _, _ = fit_and_evaluate_fold(
        X_exec_dev, y_exec_dev, X_exec_eval, y_exec_eval,
        k_features=32, use_hierarchical=False, feature_type="advanced"
    )
    p5_r11_metrics = compute_detailed_metrics(y_exec_eval, preds_p5_r11)
    print(f"  Phase 5 Champion on Execution Holdout (R11): Acc = {p5_r11_metrics['accuracy']}%, Macro F1 = {p5_r11_metrics['macro_f1']}%, Rest Recall = {p5_r11_metrics['rest_recall']}%")
    results["experiments"]["phase5_execution_baseline_r11"] = p5_r11_metrics

    # --------------------------------------------------------------------------
    # Step 3: Domain Shift — Phase 5 Model Tested Zero-Shot on Pure Motor Imagery (R12)
    # --------------------------------------------------------------------------
    print("\n[Step 3] Evaluating Cross-Task Domain Shift (Trained on Execution, Tested on Imagery R12)...")
    preds_p5_r12, _, _ = fit_and_evaluate_fold(
        X_exec_dev, y_exec_dev, X_imag_eval, y_imag_eval,
        k_features=32, use_hierarchical=False, feature_type="advanced"
    )
    p5_domain_metrics = compute_detailed_metrics(y_imag_eval, preds_p5_r12)
    print(f"  Execution -> Imagery Zero-Shot (R12): Acc = {p5_domain_metrics['accuracy']}%, Macro F1 = {p5_domain_metrics['macro_f1']}%, Left Recall = {p5_domain_metrics['left_recall']}%, Right Recall = {p5_domain_metrics['right_recall']}%")
    results["experiments"]["strategy_b_cross_task_execution_to_imagery_r12"] = p5_domain_metrics

    # --------------------------------------------------------------------------
    # Step 4: Strategy A — Pure Motor Imagery Model Development & Evaluation
    # --------------------------------------------------------------------------
    print("\n[Step 4] Strategy A: Evaluating Pure Motor Imagery (R04+R08 -> R12)...")
    
    candidates = [
        ("P6-Cand-01 (Baseline Exp08 k=28)", 28, False, "standard"),
        ("P6-Cand-02 (Advanced Motor k=32)", 32, False, "advanced"),
        ("P6-Cand-03 (Contralateral Asymmetry k=36)", 36, False, "lateralized"),
        ("P6-Cand-04 (Calibrated Hierarchical BCI)", 36, True, "lateralized")
    ]

    skf = StratifiedKFold(n_splits=50, shuffle=True, random_state=RANDOM_SEED)

    for cand_name, k_feat, is_hier, f_type in candidates:
        print(f"\n  Evaluating {cand_name} on Pure Motor Imagery...")
        
        # 50-fold Stratified CV on Development Imagery (N=600)
        cv_preds = np.zeros(len(y_imag_dev), dtype=object)
        for tr_idx, val_idx in skf.split(X_imag_dev, y_imag_dev):
            f_preds, _, _ = fit_and_evaluate_fold(
                X_imag_dev[tr_idx], y_imag_dev[tr_idx],
                X_imag_dev[val_idx], y_imag_dev[val_idx],
                k_features=k_feat, use_hierarchical=is_hier, feature_type=f_type
            )
            cv_preds[val_idx] = f_preds

        cv_metrics = compute_detailed_metrics(y_imag_dev, cv_preds)

        # Single-pass evaluation on frozen Pure Motor Imagery Holdout (R12, N=300)
        t_start = time.perf_counter()
        holdout_preds, holdout_probs, _ = fit_and_evaluate_fold(
            X_imag_dev, y_imag_dev, X_imag_eval, y_imag_eval,
            k_features=k_feat, use_hierarchical=is_hier, feature_type=f_type
        )
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        per_sample_latency = round(t_elapsed_ms / len(y_imag_eval), 3)

        holdout_metrics = compute_detailed_metrics(y_imag_eval, holdout_preds)
        holdout_metrics["per_sample_inference_latency_ms"] = per_sample_latency

        print(f"    50-Fold Dev CV: Acc = {cv_metrics['accuracy']}%, Macro F1 = {cv_metrics['macro_f1']}%, Left Recall = {cv_metrics['left_recall']}%, Right Recall = {cv_metrics['right_recall']}%")
        print(f"    Holdout R12:    Acc = {holdout_metrics['accuracy']}%, Macro F1 = {holdout_metrics['macro_f1']}%, Left Recall = {holdout_metrics['left_recall']}%, Right Recall = {holdout_metrics['right_recall']}%, Rest FAR = {holdout_metrics['rest_false_actuation_rate']}")

        key_slug = cand_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("=", "")
        results["experiments"][f"strategy_a_{key_slug}"] = {
            "candidate": cand_name,
            "dev_cv_50fold": cv_metrics,
            "holdout_r12": holdout_metrics
        }

    # --------------------------------------------------------------------------
    # Step 5: Candidate 5 — Multi-Task Joint Training (Execution + Imagery Dev -> Imagery R12)
    # --------------------------------------------------------------------------
    print("\n[Step 5] Evaluating Multi-Task Joint Training (Execution R03+R07 + Imagery R04+R08, N=1200 -> Imagery R12)...")
    X_multitask_dev = np.concatenate([X_exec_dev, X_imag_dev], axis=0)
    y_multitask_dev = np.concatenate([y_exec_dev, y_imag_dev], axis=0)

    preds_multitask, _, _ = fit_and_evaluate_fold(
        X_multitask_dev, y_multitask_dev, X_imag_eval, y_imag_eval,
        k_features=36, use_hierarchical=True, feature_type="lateralized"
    )
    multitask_metrics = compute_detailed_metrics(y_imag_eval, preds_multitask)
    print(f"  Multi-Task Joint Model on Imagery R12: Acc = {multitask_metrics['accuracy']}%, Macro F1 = {multitask_metrics['macro_f1']}%, Left Recall = {multitask_metrics['left_recall']}%, Right Recall = {multitask_metrics['right_recall']}%, Rest FAR = {multitask_metrics['rest_false_actuation_rate']}")
    results["experiments"]["strategy_a_multitask_joint_training_r12"] = multitask_metrics

    # --------------------------------------------------------------------------
    # Step 6: Strategy C — Few-Shot Subject Calibration Curve on Motor Imagery
    # --------------------------------------------------------------------------
    print("\n[Step 6] Strategy C: Evaluating Subject Calibration Curve on Motor Imagery...")
    budgets = [0, 5, 10, 15, 20]
    calib_results = []

    for budget in budgets:
        sub_accs = []
        sub_f1s = []
        sub_rest_recalls = []

        for sub in subjects:
            # Mask out this subject from development to form background pool
            sub_dev_mask = np.array([s == sub for s in imag_dev_subs])
            sub_eval_mask = np.array([s == sub for s in imag_eval_subs])

            X_pool = X_imag_dev[~sub_dev_mask]
            y_pool = y_imag_dev[~sub_dev_mask]

            X_sub_dev = X_imag_dev[sub_dev_mask]
            y_sub_dev = y_imag_dev[sub_dev_mask]

            X_sub_eval = X_imag_eval[sub_eval_mask]
            y_sub_eval = y_imag_eval[sub_eval_mask]

            if budget == 0:
                # Pure zero-shot LOSO
                X_tr = X_pool
                y_tr = y_pool
            else:
                # Sample few-shot calibration trials from subject's own dev runs
                calib_idx = []
                for c in CLASSES:
                    c_idx = np.where(y_sub_dev == c)[0]
                    chosen = c_idx[:min(budget, len(c_idx))]
                    calib_idx.extend(chosen)
                
                X_calib = X_sub_dev[calib_idx]
                y_calib = y_sub_dev[calib_idx]

                X_tr = np.concatenate([X_pool, X_calib], axis=0)
                y_tr = np.concatenate([y_pool, y_calib], axis=0)

            # Fit fold on combined training
            f_preds, _, _ = fit_and_evaluate_fold(
                X_tr, y_tr, X_sub_eval, y_sub_eval,
                k_features=32, use_hierarchical=False, feature_type="lateralized"
            )
            sub_metrics = compute_detailed_metrics(y_sub_eval, f_preds)
            sub_accs.append(sub_metrics["accuracy"])
            sub_f1s.append(sub_metrics["macro_f1"])
            sub_rest_recalls.append(sub_metrics["rest_recall"])

        mean_acc = round(float(np.mean(sub_accs)), 2)
        mean_f1 = round(float(np.mean(sub_f1s)), 2)
        mean_rest = round(float(np.mean(sub_rest_recalls)), 2)
        print(f"  Budget = {budget:2d} trials/class: Mean Acc = {mean_acc}%, Mean Macro F1 = {mean_f1}%, REST Recall = {mean_rest}%")
        calib_results.append({
            "trials_per_class": budget,
            "total_calibration_trials": budget * 3,
            "mean_accuracy": mean_acc,
            "mean_macro_f1": mean_f1,
            "mean_rest_recall": mean_rest
        })

    results["experiments"]["strategy_c_calibration_curve"] = calib_results

    # --------------------------------------------------------------------------
    # Step 7: Subject-Wise Performance on Pure Motor Imagery Holdout (R12)
    # --------------------------------------------------------------------------
    print("\n[Step 7] Computing Subject-Wise Breakdown on Pure Motor Imagery Holdout (R12)...")
    sub_breakdown = {}
    for sub in subjects:
        sub_mask = np.array([s == sub for s in imag_eval_subs])
        X_sub_eval = X_imag_eval[sub_mask]
        y_sub_eval = y_imag_eval[sub_mask]

        preds_sub, _, _ = fit_and_evaluate_fold(
            X_imag_dev, y_imag_dev, X_sub_eval, y_sub_eval,
            k_features=36, use_hierarchical=True, feature_type="lateralized"
        )
        sub_metrics = compute_detailed_metrics(y_sub_eval, preds_sub)
        sub_breakdown[sub] = sub_metrics
        print(f"  {sub}: Acc = {sub_metrics['accuracy']}%, F1 = {sub_metrics['macro_f1']}%, Left Recall = {sub_metrics['left_recall']}%, Right Recall = {sub_metrics['right_recall']}%, Rest FAR = {sub_metrics['rest_false_actuation_rate']}")

    results["experiments"]["subject_wise_breakdown_p6_hierarchical"] = sub_breakdown

    # Save to JSON
    out_path = os.path.join(REPO_ROOT, "docs", "phase6_model_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Done] Experiment results successfully written to {out_path}")
    return results


if __name__ == "__main__":
    run_phase6_experiments()
