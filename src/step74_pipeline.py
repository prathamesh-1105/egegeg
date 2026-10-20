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
from src.step61_pipeline import extract_asymmetry_and_erd_features

OUT_DIR = "models/step74"
SEL_MODEL_DIR = os.path.join(OUT_DIR, "selected_model")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SEL_MODEL_DIR, exist_ok=True)


def fetch_and_load_expanded_subject_data(num_subjects: int = 10, data_dir: str = "data") -> dict:
    """
    Loads PhysioNet EEG data for subjects S001..S010.
    Includes both Motor Execution (R03, R07, R11) and Motor Imagery (R04, R08, R12) runs.
    Calibration: R03, R04, R07, R08 (120 trials per subject = 1,200 total calibration trials).
    Holdout Test: R11, R12 (60 trials per subject = 600 total holdout test trials).
    """
    sub_data = {}
    print(f"=== Loading Doubled-Calibration (Execution + Imagery) Dataset for {num_subjects} Subjects ===")

    for sub_idx in range(1, num_subjects + 1):
        sub_id = f"S{sub_idx:03d}"
        run_ids = [3, 4, 7, 8, 11, 12]
        try:
            edf_paths = mne.datasets.eegbci.load_data(sub_idx, run_ids, update_path=True, verbose=False)
        except Exception as e:
            print(f"Warning: Could not fetch data for {sub_id}: {e}")
            continue

        path_dict = {}
        for path in edf_paths:
            fname = os.path.basename(path)
            if "R03" in fname: path_dict["R03"] = path
            elif "R04" in fname: path_dict["R04"] = path
            elif "R07" in fname: path_dict["R07"] = path
            elif "R08" in fname: path_dict["R08"] = path
            elif "R11" in fname: path_dict["R11"] = path
            elif "R12" in fname: path_dict["R12"] = path

        calib_wins, calib_lbls = [], []
        for r_key in ["R03", "R04", "R07", "R08"]:
            if r_key in path_dict:
                X_r, y_r, _ = extract_event_aligned_windows(path_dict[r_key], channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
                if len(X_r) > 0:
                    calib_wins.append(X_r)
                    calib_lbls.append(y_r)

        eval_wins, eval_lbls = [], []
        for r_key in ["R11", "R12"]:
            if r_key in path_dict:
                X_r, y_r, _ = extract_event_aligned_windows(path_dict[r_key], channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
                if len(X_r) > 0:
                    eval_wins.append(X_r)
                    eval_lbls.append(y_r)

        if calib_wins and eval_wins:
            sub_data[sub_id] = {
                "X_calib": np.concatenate(calib_wins, axis=0),
                "y_calib": np.concatenate(calib_lbls, axis=0),
                "X_holdout": np.concatenate(eval_wins, axis=0),
                "y_holdout": np.concatenate(eval_lbls, axis=0)
            }

    return sub_data


def train_and_evaluate_step74(num_subjects: int = 10, data_dir: str = "data"):
    print(f"=== Launching Step 74 Doubled-Calibration Riemannian-FBCSP Pipeline ===")
    sub_data = fetch_and_load_expanded_subject_data(num_subjects=num_subjects, data_dir=data_dir)
    subs = sorted(list(sub_data.keys()))

    if not subs:
        print("Error: No subjects were loaded successfully.")
        return

    all_holdout_preds = []
    all_holdout_gt = []
    latencies = []

    for s_id in subs:
        X_cal_s = sub_data[s_id]["X_calib"]
        y_cal_s = sub_data[s_id]["y_calib"]
        X_eval_s = sub_data[s_id]["X_holdout"]
        y_eval_s = sub_data[s_id]["y_holdout"]

        s_out = os.path.join(SEL_MODEL_DIR, s_id)
        os.makedirs(s_out, exist_ok=True)

        # 1. Subject Adaptive Filter Bank & FBCSP
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
        k_features = min(28, X_fused_s.shape[1])
        sel_s = SelectKBest(f_classif, k=k_features).fit(X_fused_s, y_cal_s)
        X_sel_s = sel_s.transform(X_fused_s)

        sc_s = StandardScaler().fit(X_sel_s)
        X_scaled_s = sc_s.transform(X_sel_s)

        # 6. Soft Voting Ensemble (RegLDA + RBF-SVM)
        lda_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        svm_clf = SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=42)

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

        # 7. Single-Pass Evaluation on R11+R12 Holdout Data
        t0 = time.time()
        filt_eval_s = extract_custom_filter_bank_windows(X_eval_s, bank_s)
        X_fbcsp_eval = extract_fbcsp_features_transform(filt_eval_s, csp_d_s)
        
        cov_eval = cov_est.transform(X_eval_s)
        X_ts_eval = ts_trans.transform(cov_eval)

        X_asym_eval = extract_asymmetry_and_erd_features(X_eval_s)
        X_fused_eval = np.hstack([X_fbcsp_eval, X_ts_eval, X_asym_eval])

        X_sel_eval = sel_s.transform(X_fused_eval)
        X_scaled_eval = sc_s.transform(X_sel_eval)

        # Probability Prediction + REST Threshold Gating (th=0.60)
        probs_eval = ensemble_clf.predict_proba(X_scaled_eval)
        classes_clf = list(ensemble_clf.classes_)

        preds_eval = []
        rest_i = classes_clf.index("REST") if "REST" in classes_clf else 0
        for row_p in probs_eval:
            rest_p = row_p[rest_i]
            if rest_p >= 0.60:
                preds_eval.append("REST")
            else:
                preds_eval.append(classes_clf[np.argmax(row_p)])

        lat_ms = (time.time() - t0) * 1000.0 / len(X_eval_s)
        latencies.append(lat_ms)

        all_holdout_preds.extend(preds_eval)
        all_holdout_gt.extend(y_eval_s)

    # Calculate Aggregate Performance across all subjects and trials
    all_holdout_gt = np.array(all_holdout_gt)
    all_holdout_preds = np.array(all_holdout_preds)

    acc = float(accuracy_score(all_holdout_gt, all_holdout_preds) * 100.0)
    macro_f1 = float(f1_score(all_holdout_gt, all_holdout_preds, average='macro') * 100.0)
    avg_lat = float(np.mean(latencies))

    cm = confusion_matrix(all_holdout_gt, all_holdout_preds, labels=CLASSES)
    rest_rec = float(cm[0, 0] / np.sum(cm[0, :]) * 100.0)
    left_rec = float(cm[1, 1] / np.sum(cm[1, :]) * 100.0)
    right_rec = float(cm[2, 2] / np.sum(cm[2, :]) * 100.0)

    print("\n=======================================================")
    print(f"STEP 74 DOUBLED-CALIBRATION ({len(subs)} SUBS, 600 HOLDOUT TRIALS) RESULTS:")
    print(f"   Total Test Trials : {len(all_holdout_gt)}")
    print(f"   Accuracy          : {acc:.2f}%")
    print(f"   Macro F1 Score    : {macro_f1:.2f}%")
    print(f"   REST Recall       : {rest_rec:.2f}%")
    print(f"   LEFT FIST Recall  : {left_rec:.2f}%")
    print(f"   RIGHT FIST Rec    : {right_rec:.2f}%")
    print(f"   Avg Latency       : {avg_lat:.2f} ms")
    print("=======================================================\n")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": f"Step 74 Doubled-Calibration Riemannian Engine",
        "num_subjects": len(subs),
        "total_test_trials": len(all_holdout_gt),
        "step61_baseline_r11_accuracy": 72.0,
        "step74_authentic_holdout_accuracy": round(acc, 2),
        "step74_authentic_holdout_macro_f1": round(macro_f1, 2),
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
    train_and_evaluate_step74(num_subjects=10)
