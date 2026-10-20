import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy import signal, stats
import mne
mne.set_log_level('WARNING')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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

OUT_DIR = "models/step63"
SEL_MODEL_DIR = os.path.join(OUT_DIR, "selected_model")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SEL_MODEL_DIR, exist_ok=True)


def extract_advanced_spectral_and_hilbert_features(X: np.ndarray, sfreq: float = 160.0) -> np.ndarray:
    """
    Extracts multi-channel variance, contralateral motor cortex asymmetry indices,
    Hilbert envelope ratios, and band-power relative features.
    """
    N, C, T = X.shape
    features = []
    
    nyq = sfreq / 2.0
    b_mu, a_mu = signal.butter(3, [8.0 / nyq, 13.0 / nyq], btype='bandpass')
    b_beta, a_beta = signal.butter(3, [14.0 / nyq, 30.0 / nyq], btype='bandpass')
    b_theta, a_theta = signal.butter(3, [4.0 / nyq, 8.0 / nyq], btype='bandpass')

    for i in range(N):
        win = X[i]
        vars_ch = np.var(win, axis=-1)
        
        c3_sig = win[C3_IDX]
        c4_sig = win[C4_IDX]
        fc3_sig = win[FC3_IDX]
        fc4_sig = win[FC4_IDX]

        c3_v = vars_ch[C3_IDX]
        c4_v = vars_ch[C4_IDX]
        fc3_v = vars_ch[FC3_IDX]
        fc4_v = vars_ch[FC4_IDX]

        # Log variance asymmetry & motor ratios
        log_asym_motor = np.log(c4_v + 1e-8) - np.log(c3_v + 1e-8)
        log_asym_premotor = np.log(fc4_v + 1e-8) - np.log(fc3_v + 1e-8)
        ratio_motor = (c3_v - c4_v) / (c3_v + c4_v + 1e-6)

        # Hilbert envelope amplitude features
        c3_mu_env = np.mean(np.abs(signal.hilbert(signal.filtfilt(b_mu, a_mu, c3_sig))))
        c4_mu_env = np.mean(np.abs(signal.hilbert(signal.filtfilt(b_mu, a_mu, c4_sig))))
        c3_beta_env = np.mean(np.abs(signal.hilbert(signal.filtfilt(b_beta, a_beta, c3_sig))))
        c4_beta_env = np.mean(np.abs(signal.hilbert(signal.filtfilt(b_beta, a_beta, c4_sig))))

        env_asym_mu = (c4_mu_env - c3_mu_env) / (c4_mu_env + c3_mu_env + 1e-6)
        env_asym_beta = (c4_beta_env - c3_beta_env) / (c4_beta_env + c3_beta_env + 1e-6)

        # Band-power ratios (Theta vs Mu, Mu vs Beta)
        c3_theta_pow = np.var(signal.filtfilt(b_theta, a_theta, c3_sig))
        c4_theta_pow = np.var(signal.filtfilt(b_theta, a_theta, c4_sig))
        theta_mu_ratio_c3 = c3_theta_pow / (c3_mu_env**2 + 1e-6)
        theta_mu_ratio_c4 = c4_theta_pow / (c4_mu_env**2 + 1e-6)

        row_feats = list(vars_ch) + [
            log_asym_motor, log_asym_premotor, ratio_motor,
            env_asym_mu, env_asym_beta, theta_mu_ratio_c3, theta_mu_ratio_c4
        ]
        features.append(row_feats)
        
    return np.array(features, dtype=np.float32)


def load_dataset_step63(data_dir: str = "data") -> dict:
    """Loads genuine EEG windows for subjects S001-S010 across R03, R07, and R11."""
    subs = [f"S{i:03d}" for i in range(1, 11)]
    sub_data = {}

    for sub_id in subs:
        found_runs = find_subject_runs(sub_id, data_dir=data_dir)
        r03_path = found_runs.get("R03")
        r07_path = found_runs.get("R07")
        r11_path = found_runs.get("R11")

        X_r03, y_r03, info_r03 = extract_event_aligned_windows(r03_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
        X_r07, y_r07, info_r07 = extract_event_aligned_windows(r07_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
        X_r11, y_r11, info_r11 = extract_event_aligned_windows(r11_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)

        X_calib = np.concatenate([X_r03, X_r07], axis=0)
        y_calib = np.concatenate([y_r03, y_r07], axis=0)

        sub_data[sub_id] = {
            "X_r03": X_r03,
            "y_r03": y_r03,
            "X_r07": X_r07,
            "y_r07": y_r07,
            "X_calib": X_calib,
            "y_calib": y_calib,
            "X_r11": X_r11,
            "y_r11": y_r11
        }

    return sub_data


def optimize_subject_rest_threshold(ensemble_clf, X_scaled_calib: np.ndarray, y_calib: np.ndarray) -> float:
    """Finds the optimal REST threshold th in [0.45, 0.70] that maximizes Macro F1 on calibration data."""
    probs_cal = ensemble_clf.predict_proba(X_scaled_calib)
    classes_clf = list(ensemble_clf.classes_)
    rest_i = classes_clf.index("REST") if "REST" in classes_clf else 0

    best_th = 0.60
    best_f1 = -1.0

    for th in np.arange(0.45, 0.71, 0.025):
        preds = []
        for row_p in probs_cal:
            if row_p[rest_i] >= th:
                preds.append("REST")
            else:
                preds.append(classes_clf[np.argmax(row_p)])
        
        f1_val = f1_score(y_calib, preds, average='macro')
        if f1_val > best_f1:
            best_f1 = f1_val
            best_th = float(th)

    return best_th


def train_and_evaluate_step63(data_dir: str = "data"):
    print("=== Launching Step 63 Multi-Band Riemannian-FBCSP & Dynamic Threshold Optimization Pipeline ===")
    sub_data = load_dataset_step63(data_dir=data_dir)
    subs = sorted(list(sub_data.keys()))

    all_r11_preds = []
    all_r11_gt = []
    latencies = []
    opt_thresholds = {}

    for s_id in subs:
        X_cal_s = sub_data[s_id]["X_calib"]
        y_cal_s = sub_data[s_id]["y_calib"]
        X_r11_s = sub_data[s_id]["X_r11"]
        y_r11_s = sub_data[s_id]["y_r11"]

        s_out = os.path.join(SEL_MODEL_DIR, s_id)
        os.makedirs(s_out, exist_ok=True)

        # 1. Subject Peak ERD Frequency Estimation & Adaptive Filter Bank
        freq_info = estimate_subject_peak_frequency(X_cal_s)
        bank_s = build_subject_adaptive_filter_bank(freq_info["mu_peak_hz"], freq_info["beta_peak_hz"])

        # 2. Filter Bank CSP Features
        filt_cal_s = extract_custom_filter_bank_windows(X_cal_s, bank_s)
        X_fbcsp_s, csp_d_s = extract_fbcsp_features_train(filt_cal_s, y_cal_s, n_components=4)

        # 3. Riemannian Tangent Space Features (using OAS covariance estimation)
        cov_est = Covariances(estimator='oas')
        ts_trans = TangentSpace(metric='riemann')
        cov_cal_s = cov_est.fit_transform(X_cal_s)
        X_ts_s = ts_trans.fit_transform(cov_cal_s)

        # 4. Advanced Spectral & Hilbert Features
        X_spec_s = extract_advanced_spectral_and_hilbert_features(X_cal_s)

        # 5. Multimodal Feature Fusion
        X_fused_s = np.hstack([X_fbcsp_s, X_ts_s, X_spec_s])

        # 6. Feature Selection & Scaling
        k_features = min(28, X_fused_s.shape[1])
        sel_s = SelectKBest(f_classif, k=k_features).fit(X_fused_s, y_cal_s)
        X_sel_s = sel_s.transform(X_fused_s)

        sc_s = StandardScaler().fit(X_sel_s)
        X_scaled_s = sc_s.transform(X_sel_s)

        # 7. Tri-Model Ensemble (RegLDA + RBF-SVM + LogisticRegression)
        lda_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        svm_clf = SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=42)
        lr_clf = LogisticRegression(C=1.5, penalty='l2', max_iter=1000, random_state=42)

        ensemble_clf = VotingClassifier(
            estimators=[('lda', lda_clf), ('svm', svm_clf), ('lr', lr_clf)],
            voting='soft',
            weights=[1.5, 1.2, 1.0]
        )
        ensemble_clf.fit(X_scaled_s, y_cal_s)

        # 8. Dynamic Per-Subject REST Threshold Calibration on Calibration Data
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
        with open(os.path.join(s_out, "threshold.json"), "w", encoding="utf-8") as f:
            json.dump({"rest_threshold": opt_th}, f)

        # 9. Single-Pass Evaluation on R11 Holdout Data
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
    print("STEP 63 MULTI-BAND RIEMANNIAN & THRESHOLD OPTIMIZATION R11 HOLDOUT RESULTS:")
    print(f"   Accuracy        : {acc:.2f}%")
    print(f"   Macro F1 Score  : {macro_f1:.2f}%")
    print(f"   REST Recall     : {rest_rec:.2f}%")
    print(f"   LEFT FIST Recall: {left_rec:.2f}%")
    print(f"   RIGHT FIST Rec  : {right_rec:.2f}%")
    print(f"   Avg Latency     : {avg_lat:.2f} ms")
    print("=======================================================\n")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "Step 63 Multi-Band Riemannian-FBCSP & Dynamic Threshold Engine",
        "step61_baseline_r11_accuracy": 72.0,
        "step63_authentic_r11_accuracy": round(acc, 2),
        "step63_authentic_r11_macro_f1": round(macro_f1, 2),
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
    train_and_evaluate_step63()
