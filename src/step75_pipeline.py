import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import joblib
import numpy as np
import pandas as pd
from collections import Counter
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
from src.step61_pipeline import extract_asymmetry_and_erd_features, load_dataset_step61

OUT_DIR = "models/step75"
SEL_MODEL_DIR = os.path.join(OUT_DIR, "selected_model")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SEL_MODEL_DIR, exist_ok=True)


def apply_causal_temporal_probability_smoothing(prob_history: list, alpha: float = 0.35) -> np.ndarray:
    """
    Applies causal exponential moving average (EMA) smoothing across consecutive probability vectors.
    S_t = alpha * S_{t-1} + (1 - alpha) * P_t.
    No future data is ever accessed.
    """
    smoothed = prob_history[0].copy()
    for p in prob_history[1:]:
        smoothed = alpha * smoothed + (1.0 - alpha) * p
    return smoothed


def apply_causal_majority_voting(pred_history: list, window_size: int = 3) -> str:
    """Applies causal majority voting over the past `window_size` predictions."""
    recent = pred_history[-window_size:]
    counts = Counter(recent)
    max_cnt = max(counts.values())
    candidates = [pred for pred, cnt in counts.items() if cnt == max_cnt]
    if len(candidates) == 1:
        return candidates[0]
    return recent[-1]


def train_and_evaluate_step75(data_dir: str = "data"):
    print("=== Launching Step 75 Causal Temporal Probability Smoothing Engine ===")
    sub_data = load_dataset_step61(data_dir=data_dir)
    subs = sorted(list(sub_data.keys()))

    raw_r11_preds = []
    smoothed_r11_preds = []
    voted_r11_preds = []
    all_r11_gt = []
    latencies = []

    for s_id in subs:
        X_cal_s = sub_data[s_id]["X_calib"]
        y_cal_s = sub_data[s_id]["y_calib"]
        X_r11_s = sub_data[s_id]["X_r11"]
        y_r11_s = sub_data[s_id]["y_r11"]

        s_out = os.path.join(SEL_MODEL_DIR, s_id)
        os.makedirs(s_out, exist_ok=True)

        # 1. Subject Peak ERD Frequency & Adaptive Filter Bank
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

        # 5. Feature Selection & Scaling
        k_features = min(24, X_fused_s.shape[1])
        sel_s = SelectKBest(f_classif, k=k_features).fit(X_fused_s, y_cal_s)
        X_sel_s = sel_s.transform(X_fused_s)

        sc_s = StandardScaler().fit(X_sel_s)
        X_scaled_s = sc_s.transform(X_sel_s)

        # 6. Soft Voting Ensemble (RegLDA + RBF-SVM)
        lda_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        svm_clf = SVC(kernel='rbf', C=1.8, gamma='scale', probability=True, random_state=42)

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

        # 7. Sequential Stream Processing with Causal Temporal Smoothing
        t0 = time.time()
        filt_r11_s = extract_custom_filter_bank_windows(X_r11_s, bank_s)
        X_fbcsp_r11 = extract_fbcsp_features_transform(filt_r11_s, csp_d_s)
        
        cov_r11 = cov_est.transform(X_r11_s)
        X_ts_r11 = ts_trans.transform(cov_r11)

        X_asym_r11 = extract_asymmetry_and_erd_features(X_r11_s)
        X_fused_r11 = np.hstack([X_fbcsp_r11, X_ts_r11, X_asym_r11])

        X_sel_r11 = sel_s.transform(X_fused_r11)
        X_scaled_r11 = sc_s.transform(X_sel_r11)

        probs_r11 = ensemble_clf.predict_proba(X_scaled_r11)
        classes_clf = list(ensemble_clf.classes_)
        rest_i = classes_clf.index("REST") if "REST" in classes_clf else 0

        # Simulate Real-Time Causal Stream Prediction
        prob_history = []
        raw_preds_s = []
        smoothed_preds_s = []
        voted_preds_s = []

        for t_idx in range(len(X_scaled_r11)):
            raw_p = probs_r11[t_idx]
            prob_history.append(raw_p)

            # Raw prediction
            if raw_p[rest_i] >= 0.60:
                raw_pred = "REST"
            else:
                raw_pred = classes_clf[np.argmax(raw_p)]
            raw_preds_s.append(raw_pred)

            # 1. Causal EMA Exponential Probability Smoothing
            smoothed_p = apply_causal_temporal_probability_smoothing(prob_history[-3:], alpha=0.35)
            if smoothed_p[rest_i] >= 0.58:
                smooth_pred = "REST"
            else:
                smooth_pred = classes_clf[np.argmax(smoothed_p)]
            smoothed_preds_s.append(smooth_pred)

            # 2. Causal Majority Voting over smoothed predictions
            vote_pred = apply_causal_majority_voting(smoothed_preds_s, window_size=3)
            voted_preds_s.append(vote_pred)

        lat_ms = (time.time() - t0) * 1000.0 / len(X_r11_s)
        latencies.append(lat_ms)

        raw_r11_preds.extend(raw_preds_s)
        smoothed_r11_preds.extend(smoothed_preds_s)
        voted_r11_preds.extend(voted_preds_s)
        all_r11_gt.extend(y_r11_s)

    # Calculate Performance Metrics
    all_r11_gt = np.array(all_r11_gt)
    
    raw_acc = float(accuracy_score(all_r11_gt, raw_r11_preds) * 100.0)
    raw_f1 = float(f1_score(all_r11_gt, raw_r11_preds, average='macro') * 100.0)

    smooth_acc = float(accuracy_score(all_r11_gt, smoothed_r11_preds) * 100.0)
    smooth_f1 = float(f1_score(all_r11_gt, smoothed_r11_preds, average='macro') * 100.0)

    voted_acc = float(accuracy_score(all_r11_gt, voted_r11_preds) * 100.0)
    voted_f1 = float(f1_score(all_r11_gt, voted_r11_preds, average='macro') * 100.0)
    avg_lat = float(np.mean(latencies))

    cm = confusion_matrix(all_r11_gt, voted_r11_preds, labels=CLASSES)
    rest_rec = float(cm[0, 0] / np.sum(cm[0, :]) * 100.0)
    left_rec = float(cm[1, 1] / np.sum(cm[1, :]) * 100.0)
    right_rec = float(cm[2, 2] / np.sum(cm[2, :]) * 100.0)

    print("\n=======================================================")
    print("STEP 75 CAUSAL TEMPORAL SMOOTHING R11 STREAM RESULTS:")
    print(f"   Raw Single-Frame Accuracy  : {raw_acc:.2f}% (F1: {raw_f1:.2f}%)")
    print(f"   EMA Prob Smoothed Accuracy : {smooth_acc:.2f}% (F1: {smooth_f1:.2f}%)")
    print(f"   Temporal Voted Accuracy     : {voted_acc:.2f}% (F1: {voted_f1:.2f}%)")
    print(f"   REST Recall                : {rest_rec:.2f}%")
    print(f"   LEFT FIST Recall           : {left_rec:.2f}%")
    print(f"   RIGHT FIST Rec             : {right_rec:.2f}%")
    print(f"   Avg Latency                : {avg_lat:.2f} ms")
    print("=======================================================\n")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "Step 75 Causal Temporal Probability Smoothing Engine",
        "raw_accuracy": round(raw_acc, 2),
        "smoothed_accuracy": round(smooth_acc, 2),
        "voted_accuracy": round(voted_acc, 2),
        "step75_authentic_r11_accuracy": round(voted_acc, 2),
        "step75_authentic_r11_macro_f1": round(voted_f1, 2),
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
    train_and_evaluate_step75()
