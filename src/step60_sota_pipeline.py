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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

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

OUT_DIR = "models/step60"
SEL_MODEL_DIR = os.path.join(OUT_DIR, "selected_model")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SEL_MODEL_DIR, exist_ok=True)


def align_probs(probs: np.ndarray, classes_from_clf) -> np.ndarray:
    """Align classifier probability outputs to canonical CLASSES order ["REST", "LEFT_FIST", "RIGHT_FIST"]."""
    classes_list = list(classes_from_clf)
    aligned = np.zeros((len(probs), 3))
    for c_i, c_name in enumerate(CLASSES):
        if c_name in classes_list:
            orig_i = classes_list.index(c_name)
            aligned[:, c_i] = probs[:, orig_i]
    return aligned


def load_dataset_step60(data_dir: str = "data") -> dict:
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
            "info_calib": info_r03 + info_r07,
            "X_r11": X_r11,
            "y_r11": y_r11,
            "info_r11": info_r11
        }

    return sub_data


def estimate_subject_peak_frequency(X_calib: np.ndarray, sfreq: float = 160.0) -> dict:
    """Estimates individual Mu (8-13 Hz) and Beta (13-30 Hz) peak frequencies from C3 & C4 channels."""
    c3_sig = X_calib[:, C3_IDX, :]
    c4_sig = X_calib[:, C4_IDX, :]
    mu_peaks, beta_peaks = [], []

    for i in range(len(X_calib)):
        freqs_c3, psd_c3 = signal.welch(c3_sig[i], fs=sfreq, nperseg=int(sfreq * 2))
        freqs_c4, psd_c4 = signal.welch(c4_sig[i], fs=sfreq, nperseg=int(sfreq * 2))
        psd_avg = (psd_c3 + psd_c4) / 2.0

        mu_mask = (freqs_c3 >= 8.0) & (freqs_c3 <= 13.0)
        if np.any(mu_mask):
            mu_peaks.append(freqs_c3[mu_mask][np.argmax(psd_avg[mu_mask])])

        beta_mask = (freqs_c3 >= 13.0) & (freqs_c3 <= 30.0)
        if np.any(beta_mask):
            beta_peaks.append(freqs_c3[beta_mask][np.argmax(psd_avg[beta_mask])])

    return {
        "mu_peak_hz": round(float(np.mean(mu_peaks)), 2) if mu_peaks else 10.0,
        "beta_peak_hz": round(float(np.mean(beta_peaks)), 2) if beta_peaks else 20.0
    }


def build_subject_adaptive_filter_bank(mu_peak: float, beta_peak: float) -> list:
    """Constructs adaptive 6-band filter bank centered around subject peak frequencies."""
    b1 = (max(4.0, mu_peak - 2.5), mu_peak + 1.5)
    b2 = (mu_peak - 1.0, mu_peak + 3.0)
    b3 = (mu_peak + 1.5, min(24.0, mu_peak + 5.5))
    b4 = (max(12.0, beta_peak - 4.0), beta_peak + 2.0)
    b5 = (beta_peak - 2.0, beta_peak + 4.0)
    b6 = (beta_peak + 2.0, min(34.0, beta_peak + 8.0))
    return [b1, b2, b3, b4, b5, b6]


def extract_custom_filter_bank_windows(X: np.ndarray, bank_bands: list, sfreq: float = 160.0) -> dict:
    """Prefilters input windows through custom frequency bands."""
    filtered_dict = {}
    nyq = sfreq / 2.0
    for idx, (f_low, f_high) in enumerate(bank_bands):
        b_low = max(0.01, f_low / nyq)
        b_high = min(0.99, f_high / nyq)
        b, a = signal.butter(4, [b_low, b_high], btype='bandpass')
        filtered = signal.filtfilt(b, a, X, axis=-1)
        filtered_dict[f"Band_{idx+1}"] = filtered
    return filtered_dict


def extract_fbcsp_features_train(filtered_dict: dict, y: np.ndarray, n_components: int = 4) -> tuple:
    """Fits CSP estimators on training filtered bands and transforms to log-variance features."""
    csp_dict = {}
    feats_list = []
    for b_name, X_b in filtered_dict.items():
        csp = CSP(n_components=n_components, reg='ledoit_wolf', log=True, norm_trace=False)
        f_b = csp.fit_transform(X_b, y)
        csp_dict[b_name] = csp
        feats_list.append(f_b)
    X_fbcsp = np.hstack(feats_list)
    return X_fbcsp, csp_dict


def extract_fbcsp_features_transform(filtered_dict: dict, csp_dict: dict) -> np.ndarray:
    """Transforms filtered bands using fitted CSP estimators."""
    feats_list = []
    for b_name, X_b in filtered_dict.items():
        f_b = csp_dict[b_name].transform(X_b)
        feats_list.append(f_b)
    return np.hstack(feats_list)


def extract_asymmetry_and_erd_features(X: np.ndarray) -> np.ndarray:
    """Extracts contralateral motor cortex asymmetry indices and channel variance features."""
    N, C, T = X.shape
    features = []
    for i in range(N):
        win = X[i]
        vars_ch = np.var(win, axis=-1)
        c3_v = vars_ch[C3_IDX]
        c4_v = vars_ch[C4_IDX]
        fc3_v = vars_ch[FC3_IDX]
        fc4_v = vars_ch[FC4_IDX]

        log_asym_motor = np.log(c4_v + 1e-8) - np.log(c3_v + 1e-8)
        log_asym_premotor = np.log(fc4_v + 1e-8) - np.log(fc3_v + 1e-8)
        ratio_motor = (c3_v - c4_v) / (c3_v + c4_v + 1e-6)

        row_feats = list(vars_ch) + [log_asym_motor, log_asym_premotor, ratio_motor]
        features.append(row_feats)
    return np.array(features, dtype=np.float32)


def run_step60_evaluation(data_dir: str = "data") -> dict:
    """
    Executes Step 60 SOTA Champion Evaluation Protocol:
    1. Subject-Specific Peak Frequency Adaptation.
    2. Riemannian Covariance + FBCSP + Asymmetry Feature Fusion.
    3. Regularized LDA Classifier with Calibrated REST Threshold Gating (th=0.60).
    4. Evaluates on genuine unseen R11 holdout dataset.
    """
    print("=== Launching Step 60 SOTA Champion Pipeline Evaluation ===")
    sub_data = load_dataset_step60(data_dir=data_dir)
    subs = sorted(list(sub_data.keys()))

    sub_adaptive_banks = {}
    for s_id in subs:
        X_cal_s = sub_data[s_id]["X_calib"]
        freq_info = estimate_subject_peak_frequency(X_cal_s)
        adapt_bank = build_subject_adaptive_filter_bank(freq_info["mu_peak_hz"], freq_info["beta_peak_hz"])
        sub_adaptive_banks[s_id] = adapt_bank

    # Fit per-subject models on R03+R07 calibration data
    start_train_t = time.time()
    for s_id in subs:
        s_out = os.path.join(SEL_MODEL_DIR, s_id)
        os.makedirs(s_out, exist_ok=True)

        X_cal_s = sub_data[s_id]["X_calib"]
        y_cal_s = sub_data[s_id]["y_calib"]
        bank_s = sub_adaptive_banks[s_id]

        # Filter Bank CSP Features
        filt_cal_s = extract_custom_filter_bank_windows(X_cal_s, bank_s)
        X_fbcsp_s, csp_d_s = extract_fbcsp_features_train(filt_cal_s, y_cal_s, n_components=4)

        # Riemannian Tangent Space Features
        cov_est = Covariances(estimator='lwf')
        ts_trans = TangentSpace(metric='riemann')
        cov_cal_s = cov_est.fit_transform(X_cal_s)
        X_ts_s = ts_trans.fit_transform(cov_cal_s)

        # Asymmetry Features
        X_asym_s = extract_asymmetry_and_erd_features(X_cal_s)

        # Fuse Feature Representations
        X_fused_s = np.hstack([X_fbcsp_s, X_ts_s, X_asym_s])

        # Feature Selection & Scaling
        k_features = min(20, X_fused_s.shape[1])
        sel_s = SelectKBest(f_classif, k=k_features).fit(X_fused_s, y_cal_s)
        sc_s = StandardScaler().fit(sel_s.transform(X_fused_s))

        # Classifier: Regularized LDA
        clf_s = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf_s.fit(sc_s.transform(sel_s.transform(X_fused_s)), y_cal_s)

        # Save artifacts per subject
        joblib.dump(bank_s, os.path.join(s_out, "adaptive_bank.pkl"))
        joblib.dump(csp_d_s, os.path.join(s_out, "csp.pkl"))
        joblib.dump(cov_est, os.path.join(s_out, "cov_estimator.pkl"))
        joblib.dump(ts_trans, os.path.join(s_out, "tangent_space.pkl"))
        joblib.dump(sel_s, os.path.join(s_out, "feature_selector.pkl"))
        joblib.dump(sc_s, os.path.join(s_out, "scaler.pkl"))
        joblib.dump(clf_s, os.path.join(s_out, "model.pkl"))

    train_time_sec = round(time.time() - start_train_t, 2)

    # Single-Pass Evaluation on Untouched Genuine R11 Holdout Data
    start_inf_t = time.time()
    all_r11_preds = []
    all_r11_gt = []

    for s_id in subs:
        X_r11_s = sub_data[s_id]["X_r11"]
        y_r11_s = sub_data[s_id]["y_r11"]

        s_out = os.path.join(SEL_MODEL_DIR, s_id)
        bank_s = joblib.load(os.path.join(s_out, "adaptive_bank.pkl"))
        csp_d_s = joblib.load(os.path.join(s_out, "csp.pkl"))
        cov_est = joblib.load(os.path.join(s_out, "cov_estimator.pkl"))
        ts_trans = joblib.load(os.path.join(s_out, "tangent_space.pkl"))
        sel_s = joblib.load(os.path.join(s_out, "feature_selector.pkl"))
        sc_s = joblib.load(os.path.join(s_out, "scaler.pkl"))
        clf_s = joblib.load(os.path.join(s_out, "model.pkl"))

        # Transform R11 features
        filt_r11_s = extract_custom_filter_bank_windows(X_r11_s, bank_s)
        X_fbcsp_r11 = extract_fbcsp_features_transform(filt_r11_s, csp_d_s)
        cov_r11_s = cov_est.transform(X_r11_s)
        X_ts_r11 = ts_trans.transform(cov_r11_s)
        X_asym_r11 = extract_asymmetry_and_erd_features(X_r11_s)

        X_fused_r11 = np.hstack([X_fbcsp_r11, X_ts_r11, X_asym_r11])
        probs_raw = clf_s.predict_proba(sc_s.transform(sel_s.transform(X_fused_r11)))
        probs_aligned = align_probs(probs_raw, clf_s.classes_)

        # Calibrated REST Threshold Gating (th=0.60)
        s_preds = []
        for k in range(len(y_r11_s)):
            pr = probs_aligned[k]
            if pr[0] >= 0.60:
                s_preds.append("REST")
            else:
                if pr[1] >= pr[2]:
                    s_preds.append("LEFT_FIST")
                else:
                    s_preds.append("RIGHT_FIST")

        all_r11_preds.extend(s_preds)
        all_r11_gt.extend(y_r11_s)

    inf_time_ms = round(((time.time() - start_inf_t) / len(all_r11_gt)) * 1000.0, 2)

    # Compute Metrics
    acc_r11 = float(accuracy_score(all_r11_gt, all_r11_preds) * 100.0)
    prec_r11 = float(precision_score(all_r11_gt, all_r11_preds, average='macro') * 100.0)
    rec_r11 = float(recall_score(all_r11_gt, all_r11_preds, average='macro') * 100.0)
    f1_r11 = float(f1_score(all_r11_gt, all_r11_preds, average='macro') * 100.0)

    report_r11 = classification_report(all_r11_gt, all_r11_preds, labels=CLASSES, output_dict=True)
    cm_r11 = confusion_matrix(all_r11_gt, all_r11_preds, labels=CLASSES)

    rest_rec = float(report_r11["REST"]["recall"] * 100.0)
    left_rec = float(report_r11["LEFT_FIST"]["recall"] * 100.0)
    right_rec = float(report_r11["RIGHT_FIST"]["recall"] * 100.0)

    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "Step 60 SOTA Champion Riemannian-FBCSP Engine",
        "step58_baseline_r11_accuracy": 60.33,
        "step59_baseline_r11_accuracy": 67.67,
        "step60_authentic_r11_accuracy": round(acc_r11, 2),
        "step60_authentic_r11_macro_f1": round(f1_r11, 2),
        "rest_recall": round(rest_rec, 2),
        "left_fist_recall": round(left_rec, 2),
        "right_fist_recall": round(right_rec, 2),
        "inference_latency_ms": inf_time_ms,
        "train_time_sec": train_time_sec,
        "confusion_matrix": cm_r11.tolist(),
        "classes": CLASSES
    }

    with open(os.path.join(OUT_DIR, "summary_report.json"), "w") as f:
        json.dump(summary_report, f, indent=2)

    print("\n================ STEP 60 CHAMPION EVALUATION RESULTS ================")
    print(f" Authentic R11 Holdout Accuracy  : {acc_r11:.2f}% (vs Step 59: 67.67%)")
    print(f" Authentic R11 Macro F1 Score    : {f1_r11:.2f}%")
    print(f" REST Recall                     : {rest_rec:.2f}%")
    print(f" LEFT_FIST Recall                : {left_rec:.2f}%")
    print(f" RIGHT_FIST Recall               : {right_rec:.2f}%")
    print(f" Inference Latency per Window    : {inf_time_ms} ms")
    print("======================================================================\n")

    return summary_report


if __name__ == "__main__":
    run_step60_evaluation()
