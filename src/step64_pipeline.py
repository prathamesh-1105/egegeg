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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
from src.step63_pipeline import (
    extract_advanced_spectral_and_hilbert_features,
    load_dataset_step63,
    optimize_subject_rest_threshold
)

OUT_DIR = "models/step64"
SEL_MODEL_DIR = os.path.join(OUT_DIR, "selected_model")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SEL_MODEL_DIR, exist_ok=True)


def apply_surface_laplacian(X: np.ndarray) -> np.ndarray:
    """
    Applies a 9-point spatial surface Laplacian filter centered around
    motor and premotor channels (C3, C4, Cz, FC3, FC4, CP3, CP4).
    """
    N, C, T = X.shape
    X_lap = X.copy()
    
    # Motor channel indices within Group B (15 channels)
    # Group B: FC3, FC1, FCz, FC2, FC4, C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4
    c3_i, c4_i, cz_i = C3_IDX, C4_IDX, CZ_IDX
    fc3_i, fc4_i = FC3_IDX, FC4_IDX
    
    # Define spatial neighbor maps for Laplacian computation
    neighbors = {
        c3_i: [0, 1, 6, 10],   # FC3, FC1, C1, CP3
        c4_i: [3, 4, 8, 14],   # FC2, FC4, C2, CP4
        cz_i: [1, 3, 6, 8],    # FC1, FC2, C1, C2
        fc3_i: [0, 1, 5, 10],  # FC3, FC1, C3, CP3
        fc4_i: [3, 4, 9, 14]   # FC2, FC4, C4, CP4
    }
    
    for ch_idx, nbr_list in neighbors.items():
        nbr_mean = np.mean(X[:, nbr_list, :], axis=1)
        X_lap[:, ch_idx, :] = X[:, ch_idx, :] - nbr_mean

    return X_lap


def train_and_evaluate_step64(data_dir: str = "data"):
    print("=== Launching Step 64 Surface Laplacian & Quad-Ensemble Optimization Pipeline ===")
    sub_data = load_dataset_step63(data_dir=data_dir)
    subs = sorted(list(sub_data.keys()))

    all_r11_preds = []
    all_r11_gt = []
    latencies = []
    opt_thresholds = {}

    for s_id in subs:
        X_cal_raw = sub_data[s_id]["X_calib"]
        y_cal_s = sub_data[s_id]["y_calib"]
        X_r11_raw = sub_data[s_id]["X_r11"]
        y_r11_s = sub_data[s_id]["y_r11"]

        s_out = os.path.join(SEL_MODEL_DIR, s_id)
        os.makedirs(s_out, exist_ok=True)

        # 1. Apply Surface Laplacian Filtering
        X_cal_s = apply_surface_laplacian(X_cal_raw)
        X_r11_s = apply_surface_laplacian(X_r11_raw)

        # 2. Subject Peak ERD Frequency & Adaptive Filter Bank
        freq_info = estimate_subject_peak_frequency(X_cal_s)
        bank_s = build_subject_adaptive_filter_bank(freq_info["mu_peak_hz"], freq_info["beta_peak_hz"])

        # 3. Filter Bank CSP Features
        filt_cal_s = extract_custom_filter_bank_windows(X_cal_s, bank_s)
        X_fbcsp_s, csp_d_s = extract_fbcsp_features_train(filt_cal_s, y_cal_s, n_components=4)

        # 4. Riemannian Tangent Space Features (LWF Covariance)
        cov_est = Covariances(estimator='lwf')
        ts_trans = TangentSpace(metric='riemann')
        cov_cal_s = cov_est.fit_transform(X_cal_s)
        X_ts_s = ts_trans.fit_transform(cov_cal_s)

        # 5. Advanced Spectral & Hilbert Features
        X_spec_s = extract_advanced_spectral_and_hilbert_features(X_cal_s)

        # 6. Multimodal Feature Fusion
        X_fused_s = np.hstack([X_fbcsp_s, X_ts_s, X_spec_s])

        # 7. Feature Selection & Scaling
        k_features = min(26, X_fused_s.shape[1])
        sel_s = SelectKBest(f_classif, k=k_features).fit(X_fused_s, y_cal_s)
        X_sel_s = sel_s.transform(X_fused_s)

        sc_s = StandardScaler().fit(X_sel_s)
        X_scaled_s = sc_s.transform(X_sel_s)

        # 8. Quad-Model Ensemble (RegLDA + RBF-SVM + LogisticRegression + Calibrated Ridge)
        lda_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        svm_clf = SVC(kernel='rbf', C=1.8, gamma='scale', probability=True, random_state=42)
        lr_clf = LogisticRegression(C=1.2, penalty='l2', max_iter=1000, random_state=42)
        ridge_base = RidgeClassifier(alpha=1.0)
        ridge_clf = CalibratedClassifierCV(ridge_base, cv=3)

        ensemble_clf = VotingClassifier(
            estimators=[('lda', lda_clf), ('svm', svm_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
            voting='soft',
            weights=[1.5, 1.2, 1.0, 0.8]
        )
        ensemble_clf.fit(X_scaled_s, y_cal_s)

        # 9. Dynamic Per-Subject REST Threshold Calibration
        opt_th = optimize_subject_rest_threshold(ensemble_clf, X_scaled_s, y_cal_s)
        opt_thresholds[s_id] = round(opt_th, 3)

        # Save artifacts
        joblib.dump(bank_s, os.path.join(s_out, "adaptive_bank.pkl"))
        joblib.dump(csp_d_s, os.path.join(s_out, "csp.pkl"))
        joblib.dump(cov_est, os.path.join(s_out, "cov_estimator.pkl"))
        joblib.dump(ts_trans, os.path.join(s_out, "tangent_space.pkl"))
        joblib.dump(sel_s, os.path.join(s_out, "feature_selector.pkl"))
        joblib.dump(sc_s, os.path.join(s_out, "scaler.pkl"))
        joblib.dump(ensemble_clf, os.path.join(s_out, "model.pkl"))

        # 10. Single-Pass Evaluation on R11 Holdout Data
        t0 = time.time()
        filt_r11_s = extract_custom_filter_bank_windows(X_r11_s, bank_s)
        X_fbcsp_r11 = extract_fbcsp_features_transform(filt_r11_s, csp_d_s)
        
        cov_r11 = cov_est.transform(X_r11_s)
        X_ts_r11 = ts_trans.transform(cov_r11)

        X_spec_r11 = extract_advanced_spectral_and_hilbert_features(X_r11_s)
        X_fused_r11 = np.hstack([X_fbcsp_r11, X_ts_r11, X_spec_r11])

        X_sel_r11 = sel_s.transform(X_fused_r11)
        X_scaled_r11 = sc_s.transform(X_sel_r11)

        # Probability Prediction + Dynamic REST Threshold Gating
        probs_r11 = ensemble_clf.predict_proba(X_scaled_r11)
        classes_clf = list(ensemble_clf.classes_)

        preds_r11 = []
        rest_i = classes_clf.index("REST") if "REST" in classes_clf else 0
        for row_p in probs_r11:
            rest_p = row_p[rest_i]
            if rest_p >= opt_th:
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
    print("STEP 64 SURFACE LAPLACIAN & QUAD-ENSEMBLE R11 HOLDOUT RESULTS:")
    print(f"   Accuracy        : {acc:.2f}%")
    print(f"   Macro F1 Score  : {macro_f1:.2f}%")
    print(f"   REST Recall     : {rest_rec:.2f}%")
    print(f"   LEFT FIST Recall: {left_rec:.2f}%")
    print(f"   RIGHT FIST Rec  : {right_rec:.2f}%")
    print(f"   Avg Latency     : {avg_lat:.2f} ms")
    print("=======================================================\n")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "Step 64 Surface Laplacian & Quad-Ensemble Engine",
        "step61_baseline_r11_accuracy": 72.0,
        "step64_authentic_r11_accuracy": round(acc, 2),
        "step64_authentic_r11_macro_f1": round(macro_f1, 2),
        "rest_recall": round(rest_rec, 2),
        "left_fist_recall": round(left_rec, 2),
        "right_fist_recall": round(right_rec, 2),
        "inference_latency_ms": round(avg_lat, 2),
        "confusion_matrix": cm.tolist(),
        "per_subject_thresholds": opt_thresholds,
        "classes": CLASSES
    }

    with open(os.path.join(OUT_DIR, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    train_and_evaluate_step64()
