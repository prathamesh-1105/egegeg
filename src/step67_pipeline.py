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
from src.step61_pipeline import load_dataset_step61

OUT_DIR = "models/step67"
SEL_MODEL_DIR = os.path.join(OUT_DIR, "selected_model")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SEL_MODEL_DIR, exist_ok=True)


def extract_bipolar_and_erd_asymmetry(X: np.ndarray, sfreq: float = 160.0) -> np.ndarray:
    """
    Extracts explicit bipolar differential channels (C3-Cz, C4-Cz, FC3-Cz, FC4-Cz)
    along with narrow-band Mu/Beta contralateral desynchronization ratios.
    """
    N, C, T = X.shape
    features = []
    
    nyq = sfreq / 2.0
    b_mu, a_mu = signal.butter(3, [8.0 / nyq, 13.0 / nyq], btype='bandpass')
    b_beta, a_beta = signal.butter(3, [14.0 / nyq, 30.0 / nyq], btype='bandpass')

    for i in range(N):
        win = X[i]
        vars_ch = np.var(win, axis=-1)
        
        # Original channel signals
        c3_sig = win[C3_IDX]
        c4_sig = win[C4_IDX]
        cz_sig = win[CZ_IDX]
        fc3_sig = win[FC3_IDX]
        fc4_sig = win[FC4_IDX]

        # Bipolar differential signals
        bip_c3_cz = c3_sig - cz_sig
        bip_c4_cz = c4_sig - cz_sig
        bip_fc3_cz = fc3_sig - cz_sig
        bip_fc4_cz = fc4_sig - cz_sig

        bip_c3_v = np.var(bip_c3_cz)
        bip_c4_v = np.var(bip_c4_cz)
        bip_fc3_v = np.var(bip_fc3_cz)
        bip_fc4_v = np.var(bip_fc4_cz)

        # Bipolar Log Variance Asymmetry
        log_bip_motor_asym = np.log(bip_c4_v + 1e-8) - np.log(bip_c3_v + 1e-8)
        log_bip_premotor_asym = np.log(bip_fc4_v + 1e-8) - np.log(bip_fc3_v + 1e-8)

        # Mu & Beta band filtered desynchronization power
        c3_mu_p = np.var(signal.filtfilt(b_mu, a_mu, c3_sig))
        c4_mu_p = np.var(signal.filtfilt(b_mu, a_mu, c4_sig))
        c3_beta_p = np.var(signal.filtfilt(b_beta, a_beta, c3_sig))
        c4_beta_p = np.var(signal.filtfilt(b_beta, a_beta, c4_sig))

        mu_erd_ratio = (c3_mu_p - c4_mu_p) / (c3_mu_p + c4_mu_p + 1e-6)
        beta_erd_ratio = (c3_beta_p - c4_beta_p) / (c3_beta_p + c4_beta_p + 1e-6)

        row_feats = list(vars_ch) + [
            bip_c3_v, bip_c4_v, bip_fc3_v, bip_fc4_v,
            log_bip_motor_asym, log_bip_premotor_asym,
            mu_erd_ratio, beta_erd_ratio
        ]
        features.append(row_feats)
        
    return np.array(features, dtype=np.float32)


def train_and_evaluate_step67(data_dir: str = "data"):
    print("=== Launching Step 67 Bipolar Differencing & Class-Balanced Riemannian Engine ===")
    sub_data = load_dataset_step61(data_dir=data_dir)
    subs = sorted(list(sub_data.keys()))

    all_r11_preds = []
    all_r11_gt = []
    latencies = []

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

        # 2. Riemannian Tangent Space Features
        cov_est = Covariances(estimator='lwf')
        ts_trans = TangentSpace(metric='riemann')
        cov_cal_s = cov_est.fit_transform(X_cal_s)
        X_ts_s = ts_trans.fit_transform(cov_cal_s)

        # 3. Bipolar & Contralateral ERD Asymmetry Features
        X_bip_s = extract_bipolar_and_erd_asymmetry(X_cal_s)

        # 4. Multimodal Feature Fusion
        X_fused_s = np.hstack([X_fbcsp_s, X_ts_s, X_bip_s])

        # 5. Feature Selection & Scaling
        k_features = min(26, X_fused_s.shape[1])
        sel_s = SelectKBest(f_classif, k=k_features).fit(X_fused_s, y_cal_s)
        X_sel_s = sel_s.transform(X_fused_s)

        sc_s = StandardScaler().fit(X_sel_s)
        X_scaled_s = sc_s.transform(X_sel_s)

        # 6. Class-Balanced Soft Voting Ensemble (RegLDA + Balanced RBF-SVM)
        lda_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        svm_clf = SVC(kernel='rbf', C=2.0, class_weight='balanced', gamma='scale', probability=True, random_state=42)

        ensemble_clf = VotingClassifier(
            estimators=[('lda', lda_clf), ('svm', svm_clf)],
            voting='soft',
            weights=[1.6, 1.0]
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

        # 7. Single-Pass Evaluation on R11 Holdout Data
        t0 = time.time()
        filt_r11_s = extract_custom_filter_bank_windows(X_r11_s, bank_s)
        X_fbcsp_r11 = extract_fbcsp_features_transform(filt_r11_s, csp_d_s)
        
        cov_r11 = cov_est.transform(X_r11_s)
        X_ts_r11 = ts_trans.transform(cov_r11)

        X_bip_r11 = extract_bipolar_and_erd_asymmetry(X_r11_s)
        X_fused_r11 = np.hstack([X_fbcsp_r11, X_ts_r11, X_bip_r11])

        X_sel_r11 = sel_s.transform(X_fused_r11)
        X_scaled_r11 = sc_s.transform(X_sel_r11)

        # Probability Prediction + REST Threshold Gating (th=0.58)
        probs_r11 = ensemble_clf.predict_proba(X_scaled_r11)
        classes_clf = list(ensemble_clf.classes_)

        preds_r11 = []
        rest_i = classes_clf.index("REST") if "REST" in classes_clf else 0
        for row_p in probs_r11:
            rest_p = row_p[rest_i]
            if rest_p >= 0.58:
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
    print("STEP 67 BIPOLAR DIFFERENCING & CLASS-BALANCED R11 HOLDOUT RESULTS:")
    print(f"   Accuracy        : {acc:.2f}%")
    print(f"   Macro F1 Score  : {macro_f1:.2f}%")
    print(f"   REST Recall     : {rest_rec:.2f}%")
    print(f"   LEFT FIST Recall: {left_rec:.2f}%")
    print(f"   RIGHT FIST Rec  : {right_rec:.2f}%")
    print(f"   Avg Latency     : {avg_lat:.2f} ms")
    print("=======================================================\n")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "Step 67 Bipolar Differencing & Class-Balanced Engine",
        "step61_baseline_r11_accuracy": 72.0,
        "step67_authentic_r11_accuracy": round(acc, 2),
        "step67_authentic_r11_macro_f1": round(macro_f1, 2),
        "rest_recall": round(rest_rec, 2),
        "left_fist_recall": round(left_rec, 2),
        "right_fist_recall": round(right_rec, 2),
        "inference_latency_ms": round(avg_lat, 2),
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES
    }

    with open(os.path.join(OUT_DIR, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    train_and_evaluate_step67()
