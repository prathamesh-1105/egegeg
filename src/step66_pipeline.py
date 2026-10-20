import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy import signal
import mne
mne.set_log_level('WARNING')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from src.step8_pipeline import find_subject_runs
from src.step14_pipeline import extract_event_aligned_windows
from src.step33_pipeline import (
    GROUP_B_CHANNELS,
    CLASSES,
    C3_IDX,
    C4_IDX,
    CZ_IDX,
    FC3_IDX,
    FC4_IDX
)
from src.step59_pipeline import (
    estimate_subject_peak_frequency,
    build_subject_adaptive_filter_bank,
    extract_custom_filter_bank_windows,
    extract_fbcsp_features_train,
    extract_fbcsp_features_transform
)
from src.step61_pipeline import extract_asymmetry_and_erd_features, load_dataset_step61

OUT_DIR = "models/step66"
SEL_MODEL_DIR = os.path.join(OUT_DIR, "selected_model")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SEL_MODEL_DIR, exist_ok=True)


def optimize_subject_pipeline_hyperparams(X_fused_cal: np.ndarray, y_cal: np.ndarray):
    """
    Performs 5-fold cross-validation grid search on calibration data
    to optimize k_features, SVM C parameter, and LDA shrinkage per subject.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_score = -1.0
    best_params = {
        "k_features": 24,
        "svm_c": 1.8,
        "lda_shrinkage": "auto",
        "rest_threshold": 0.60
    }

    k_candidates = [20, 24, 28]
    c_candidates = [1.5, 1.8, 2.2]
    th_candidates = [0.58, 0.60, 0.62]

    for k_feat in k_candidates:
        if k_feat > X_fused_cal.shape[1]:
            continue
        sel = SelectKBest(f_classif, k=k_feat).fit(X_fused_cal, y_cal)
        X_sel = sel.transform(X_fused_cal)

        sc = StandardScaler().fit(X_sel)
        X_scaled = sc.transform(X_sel)

        for c_val in c_candidates:
            lda_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            svm_clf = SVC(kernel='rbf', C=c_val, gamma='scale', probability=True, random_state=42)
            ens_clf = VotingClassifier(
                estimators=[('lda', lda_clf), ('svm', svm_clf)],
                voting='soft',
                weights=[1.5, 1.0]
            )

            cv_scores = []
            for train_idx, val_idx in cv.split(X_scaled, y_cal):
                X_tr, y_tr = X_scaled[train_idx], y_cal[train_idx]
                X_va, y_va = X_scaled[val_idx], y_cal[val_idx]

                ens_clf.fit(X_tr, y_tr)
                probs_va = ens_clf.predict_proba(X_va)
                classes_clf = list(ens_clf.classes_)
                rest_i = classes_clf.index("REST") if "REST" in classes_clf else 0

                for th in th_candidates:
                    preds = []
                    for row_p in probs_va:
                        if row_p[rest_i] >= th:
                            preds.append("REST")
                        else:
                            preds.append(classes_clf[np.argmax(row_p)])
                    
                    score = f1_score(y_va, preds, average='macro')
                    cv_scores.append((score, th))

            # Aggregate cross-validation score across folds
            avg_score = float(np.mean([s[0] for s in cv_scores]))
            if avg_score > best_score:
                best_score = avg_score
                best_params = {
                    "k_features": k_feat,
                    "svm_c": c_val,
                    "lda_shrinkage": "auto",
                    "rest_threshold": 0.60
                }

    return best_params


def train_and_evaluate_step66(data_dir: str = "data"):
    print("=== Launching Step 66 Hyper-Tuned Per-Subject Adaptive Ensemble Pipeline ===")
    sub_data = load_dataset_step61(data_dir=data_dir)
    subs = sorted(list(sub_data.keys()))

    all_r11_preds = []
    all_r11_gt = []
    latencies = []
    subject_hyperparams = {}

    for s_id in subs:
        X_cal_s = sub_data[s_id]["X_calib"]
        y_cal_s = sub_data[s_id]["y_calib"]
        X_r11_s = sub_data[s_id]["X_r11"]
        y_r11_s = sub_data[s_id]["y_r11"]

        s_out = os.path.join(SEL_MODEL_DIR, s_id)
        os.makedirs(s_out, exist_ok=True)

        # 1. Adaptive Filter Bank & FBCSP
        freq_info = estimate_subject_peak_frequency(X_cal_s)
        bank_s = build_subject_adaptive_filter_bank(freq_info["mu_peak_hz"], freq_info["beta_peak_hz"])
        filt_cal_s = extract_custom_filter_bank_windows(X_cal_s, bank_s)
        X_fbcsp_s, csp_d_s = extract_fbcsp_features_train(filt_cal_s, y_cal_s, n_components=4)

        # 2. Riemannian Tangent Space Features (LWF Covariance)
        cov_est = Covariances(estimator='lwf')
        ts_trans = TangentSpace(metric='riemann')
        cov_cal_s = cov_est.fit_transform(X_cal_s)
        X_ts_s = ts_trans.fit_transform(cov_cal_s)

        # 3. Contralateral Asymmetry Features
        X_asym_s = extract_asymmetry_and_erd_features(X_cal_s)

        # 4. Multimodal Feature Fusion
        X_fused_s = np.hstack([X_fbcsp_s, X_ts_s, X_asym_s])

        # 5. Per-Subject Hyperparameter Optimization on Calibration Set
        best_p = optimize_subject_pipeline_hyperparams(X_fused_s, y_cal_s)
        subject_hyperparams[s_id] = best_p

        # 6. Feature Selection & Scaling using tuned params
        k_feat = best_p["k_features"]
        sel_s = SelectKBest(f_classif, k=k_feat).fit(X_fused_s, y_cal_s)
        X_sel_s = sel_s.transform(X_fused_s)

        sc_s = StandardScaler().fit(X_sel_s)
        X_scaled_s = sc_s.transform(X_sel_s)

        # 7. Tuned Ensemble
        lda_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=best_p["lda_shrinkage"])
        svm_clf = SVC(kernel='rbf', C=best_p["svm_c"], gamma='scale', probability=True, random_state=42)

        ensemble_clf = VotingClassifier(
            estimators=[('lda', lda_clf), ('svm', svm_clf)],
            voting='soft',
            weights=[1.5, 1.0]
        )
        ensemble_clf.fit(X_scaled_s, y_cal_s)

        # Save artifacts
        joblib.dump(bank_s, os.path.join(s_out, "adaptive_bank.pkl"))
        joblib.dump(csp_d_s, os.path.join(s_out, "csp.pkl"))
        joblib.dump(cov_est, os.path.join(s_out, "cov_estimator.pkl"))
        joblib.dump(ts_trans, os.path.join(s_out, "tangent_space.pkl"))
        joblib.dump(sel_s, os.path.join(s_out, "feature_selector.pkl"))
        joblib.dump(sc_s, os.path.join(s_out, "scaler.pkl"))
        joblib.dump(ensemble_clf, os.path.join(s_out, "model.pkl"))

        # 8. Single-Pass Evaluation on R11 Holdout Data
        t0 = time.time()
        filt_r11_s = extract_custom_filter_bank_windows(X_r11_s, bank_s)
        X_fbcsp_r11 = extract_fbcsp_features_transform(filt_r11_s, csp_d_s)
        
        cov_r11 = cov_est.transform(X_r11_s)
        X_ts_r11 = ts_trans.transform(cov_r11)

        X_asym_r11 = extract_asymmetry_and_erd_features(X_r11_s)
        X_fused_r11 = np.hstack([X_fbcsp_r11, X_ts_r11, X_asym_r11])

        X_sel_r11 = sel_s.transform(X_fused_r11)
        X_scaled_r11 = sc_s.transform(X_sel_r11)

        # Probability Prediction + REST Threshold Gating
        probs_r11 = ensemble_clf.predict_proba(X_scaled_r11)
        classes_clf = list(ensemble_clf.classes_)

        preds_r11 = []
        rest_i = classes_clf.index("REST") if "REST" in classes_clf else 0
        th_val = best_p["rest_threshold"]
        for row_p in probs_r11:
            rest_p = row_p[rest_i]
            if rest_p >= th_val:
                preds_r11.append("REST")
            else:
                preds_r11.append(classes_clf[np.argmax(row_p)])

        lat_ms = (time.time() - t0) * 1000.0 / len(X_r11_s)
        latencies.append(lat_ms)

        all_r11_preds.extend(preds_r11)
        all_r11_gt.extend(y_r11_s)

    # Calculate Aggregate Performance
    all_r11_gt = np.array(all_r11_gt)
    all_r11_preds = np.array(all_r11_preds)

    acc = float(accuracy_score(all_r11_gt, all_r11_preds) * 100.0)
    macro_f1 = float(f1_score(all_r11_gt, all_r11_preds, average='macro') * 100.0)
    avg_lat = float(np.mean(latencies))

    cm = confusion_matrix(all_r11_gt, all_r11_preds, labels=CLASSES)
    rest_rec = float(cm[0, 0] / np.sum(cm[0, :]) * 100.0)
    left_rec = float(cm[1, 1] / np.sum(cm[1, :]) * 100.0)
    right_rec = float(cm[2, 2] / np.sum(cm[2, :]) * 100.0)

    print("\n=======================================================")
    print("STEP 66 HYPER-TUNED PER-SUBJECT ENSEMBLE R11 HOLDOUT RESULTS:")
    print(f"   Accuracy        : {acc:.2f}%")
    print(f"   Macro F1 Score  : {macro_f1:.2f}%")
    print(f"   REST Recall     : {rest_rec:.2f}%")
    print(f"   LEFT FIST Recall: {left_rec:.2f}%")
    print(f"   RIGHT FIST Rec  : {right_rec:.2f}%")
    print(f"   Avg Latency     : {avg_lat:.2f} ms")
    print("=======================================================\n")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "Step 66 Hyper-Tuned Per-Subject Ensemble Engine",
        "step61_baseline_r11_accuracy": 72.0,
        "step66_authentic_r11_accuracy": round(acc, 2),
        "step66_authentic_r11_macro_f1": round(macro_f1, 2),
        "rest_recall": round(rest_rec, 2),
        "left_fist_recall": round(left_rec, 2),
        "right_fist_recall": round(right_rec, 2),
        "inference_latency_ms": round(avg_lat, 2),
        "confusion_matrix": cm.tolist(),
        "subject_hyperparameters": subject_hyperparams,
        "classes": CLASSES
    }

    with open(os.path.join(OUT_DIR, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    train_and_evaluate_step66()
