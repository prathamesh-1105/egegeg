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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
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
from src.step63_pipeline import (
    extract_advanced_spectral_and_hilbert_features,
    load_dataset_step63,
    optimize_subject_rest_threshold
)

OUT_DIR = "models/step65"
SEL_MODEL_DIR = os.path.join(OUT_DIR, "selected_model")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SEL_MODEL_DIR, exist_ok=True)


def extract_subwindow_features(X: np.ndarray, sfreq: float = 160.0) -> np.ndarray:
    """Extracts features from the early motor onset sub-window (0.5s to 2.5s within the 3.0s epoch)."""
    # 3.0s epoch at 160 Hz has 480 time points (0 to 480)
    # 0.5s to 2.5s corresponds to indices 80 to 400
    start_idx = int(0.5 * sfreq)
    end_idx = int(2.5 * sfreq)
    X_sub = X[:, :, start_idx:end_idx]
    
    return extract_advanced_spectral_and_hilbert_features(X_sub, sfreq=sfreq)


def train_and_evaluate_step65(data_dir: str = "data"):
    print("=== Launching Step 65 Multi-Window ERD & Stacking Meta-Learner Pipeline ===")
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

        # 1. Subject Peak ERD Frequency & Adaptive Filter Bank
        freq_info = estimate_subject_peak_frequency(X_cal_s)
        bank_s = build_subject_adaptive_filter_bank(freq_info["mu_peak_hz"], freq_info["beta_peak_hz"])

        # 2. Filter Bank CSP Features
        filt_cal_s = extract_custom_filter_bank_windows(X_cal_s, bank_s)
        X_fbcsp_s, csp_d_s = extract_fbcsp_features_train(filt_cal_s, y_cal_s, n_components=4)

        # 3. Riemannian Tangent Space Features
        cov_est = Covariances(estimator='oas')
        ts_trans = TangentSpace(metric='riemann')
        cov_cal_s = cov_est.fit_transform(X_cal_s)
        X_ts_s = ts_trans.fit_transform(cov_cal_s)

        # 4. Multi-Window Features (Full 3.0s + Early Onset 0.5s-2.5s)
        X_spec_full_s = extract_advanced_spectral_and_hilbert_features(X_cal_s)
        X_spec_sub_s = extract_subwindow_features(X_cal_s)

        # 5. Multimodal Feature Fusion
        X_fused_s = np.hstack([X_fbcsp_s, X_ts_s, X_spec_full_s, X_spec_sub_s])

        # 6. Feature Selection & Scaling
        k_features = min(30, X_fused_s.shape[1])
        sel_s = SelectKBest(f_classif, k=k_features).fit(X_fused_s, y_cal_s)
        X_sel_s = sel_s.transform(X_fused_s)

        sc_s = StandardScaler().fit(X_sel_s)
        X_scaled_s = sc_s.transform(X_sel_s)

        # 7. Stacking Meta-Learner Ensemble
        base_estimators = [
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
            ('svm', SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=42)),
            ('lr', LogisticRegression(C=1.5, penalty='l2', max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
        ]
        meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            stack_method='predict_proba'
        )
        stacking_clf.fit(X_scaled_s, y_cal_s)

        # 8. Dynamic Per-Subject REST Threshold Calibration
        opt_th = optimize_subject_rest_threshold(stacking_clf, X_scaled_s, y_cal_s)
        opt_thresholds[s_id] = round(opt_th, 3)

        # Save artifacts
        joblib.dump(bank_s, os.path.join(s_out, "adaptive_bank.pkl"))
        joblib.dump(csp_d_s, os.path.join(s_out, "csp.pkl"))
        joblib.dump(cov_est, os.path.join(s_out, "cov_estimator.pkl"))
        joblib.dump(ts_trans, os.path.join(s_out, "tangent_space.pkl"))
        joblib.dump(sel_s, os.path.join(s_out, "feature_selector.pkl"))
        joblib.dump(sc_s, os.path.join(s_out, "scaler.pkl"))
        joblib.dump(stacking_clf, os.path.join(s_out, "model.pkl"))

        # 9. Single-Pass Evaluation on R11 Holdout Data
        t0 = time.time()
        filt_r11_s = extract_custom_filter_bank_windows(X_r11_s, bank_s)
        X_fbcsp_r11 = extract_fbcsp_features_transform(filt_r11_s, csp_d_s)
        
        cov_r11 = cov_est.transform(X_r11_s)
        X_ts_r11 = ts_trans.transform(cov_r11)

        X_spec_full_r11 = extract_advanced_spectral_and_hilbert_features(X_r11_s)
        X_spec_sub_r11 = extract_subwindow_features(X_r11_s)
        X_fused_r11 = np.hstack([X_fbcsp_r11, X_ts_r11, X_spec_full_r11, X_spec_sub_r11])

        X_sel_r11 = sel_s.transform(X_fused_r11)
        X_scaled_r11 = sc_s.transform(X_sel_r11)

        # Probability Prediction + Dynamic REST Threshold Gating
        probs_r11 = stacking_clf.predict_proba(X_scaled_r11)
        classes_clf = list(stacking_clf.classes_)

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
    print("STEP 65 MULTI-WINDOW & STACKING META-LEARNER R11 HOLDOUT RESULTS:")
    print(f"   Accuracy        : {acc:.2f}%")
    print(f"   Macro F1 Score  : {macro_f1:.2f}%")
    print(f"   REST Recall     : {rest_rec:.2f}%")
    print(f"   LEFT FIST Recall: {left_rec:.2f}%")
    print(f"   RIGHT FIST Rec  : {right_rec:.2f}%")
    print(f"   Avg Latency     : {avg_lat:.2f} ms")
    print("=======================================================\n")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "Step 65 Multi-Window & Stacking Meta-Learner Engine",
        "step61_baseline_r11_accuracy": 72.0,
        "step65_authentic_r11_accuracy": round(acc, 2),
        "step65_authentic_r11_macro_f1": round(macro_f1, 2),
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
    train_and_evaluate_step65()
