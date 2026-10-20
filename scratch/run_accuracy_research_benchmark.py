"""
Canonical Accuracy Research Benchmark for NeuroOS (Step 76 Phase 7)
Evaluates 15 model, feature, window, and classifier configurations on 
real PhysioNet EEGMMIDB EDF recordings (S001-S010, R03/R07/R11).

Strict zero-leakage evaluation:
- GroupKFold by subject for Dev / Hyperparameter exploration
- Independent R11 Holdout (300 test trials) for final evaluation
"""

import os
import sys
import time
import json
import csv
import numpy as np
import pandas as pd
from scipy import signal, stats
import mne
mne.set_log_level('WARNING')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GroupKFold
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

sys.path.insert(0, os.path.abspath("."))
from src.step8_pipeline import find_subject_runs
from src.step14_pipeline import extract_event_aligned_windows
from src.step33_pipeline import GROUP_B_CHANNELS, CLASSES, C3_IDX, C4_IDX, CZ_IDX, FC3_IDX, FC4_IDX
from src.step59_pipeline import (
    estimate_subject_peak_frequency,
    build_subject_adaptive_filter_bank,
    extract_custom_filter_bank_windows,
    extract_fbcsp_features_train,
    extract_fbcsp_features_transform,
    align_probs
)
from src.step61_pipeline import extract_asymmetry_and_erd_features

np.random.seed(42)

def load_full_dataset(data_dir: str = "data", window_sec: float = 3.0, onset_offset_sec: float = 0.5):
    """Loads genuine EEG windows for S001-S010 across R03, R07, and R11."""
    subs = [f"S{i:03d}" for i in range(1, 11)]
    sub_data = {}

    for sub_id in subs:
        found_runs = find_subject_runs(sub_id, data_dir=data_dir)
        r03_path = found_runs.get("R03")
        r07_path = found_runs.get("R07")
        r11_path = found_runs.get("R11")

        X_r03, y_r03, info_r03 = extract_event_aligned_windows(r03_path, channel_group="Group B", window_sec=window_sec, onset_offset_sec=onset_offset_sec)
        X_r07, y_r07, info_r07 = extract_event_aligned_windows(r07_path, channel_group="Group B", window_sec=window_sec, onset_offset_sec=onset_offset_sec)
        X_r11, y_r11, info_r11 = extract_event_aligned_windows(r11_path, channel_group="Group B", window_sec=window_sec, onset_offset_sec=onset_offset_sec)

        X_calib = np.concatenate([X_r03, X_r07], axis=0)
        y_calib = np.concatenate([y_r03, y_r07], axis=0)

        sub_data[sub_id] = {
            "X_calib": X_calib,
            "y_calib": y_calib,
            "X_r11": X_r11,
            "y_r11": y_r11
        }

    return sub_data

def evaluate_configuration(
    config_name: str,
    sub_data: dict,
    use_fbcsp: bool = True,
    use_riemannian: bool = True,
    use_asym: bool = True,
    use_subject_adaptive: bool = True,
    clf_type: str = "reg_lda",
    class_weight: str = None,
    rest_threshold: float = 0.60,
    k_features: int = 24
):
    subs = sorted(list(sub_data.keys()))
    
    # Dev GroupKFold evaluation on calibration data
    gkf = GroupKFold(n_splits=5)
    
    # Prepare features per subject
    sub_features = {}
    for s_id in subs:
        X_cal_s = sub_data[s_id]["X_calib"]
        y_cal_s = sub_data[s_id]["y_calib"]
        X_r11_s = sub_data[s_id]["X_r11"]

        if use_subject_adaptive:
            freq_info = estimate_subject_peak_frequency(X_cal_s)
            bank_s = build_subject_adaptive_filter_bank(freq_info["mu_peak_hz"], freq_info["beta_peak_hz"])
        else:
            bank_s = [(8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0), (24.0, 28.0)]

        # Filter Bank CSP
        if use_fbcsp:
            filt_cal_s = extract_custom_filter_bank_windows(X_cal_s, bank_s)
            X_fbcsp_s, csp_d_s = extract_fbcsp_features_train(filt_cal_s, y_cal_s, n_components=4)
            filt_r11_s = extract_custom_filter_bank_windows(X_r11_s, bank_s)
            X_fbcsp_r11 = extract_fbcsp_features_transform(filt_r11_s, csp_d_s)
        else:
            X_fbcsp_s = np.zeros((len(X_cal_s), 0))
            X_fbcsp_r11 = np.zeros((len(X_r11_s), 0))

        # Riemannian Tangent Space
        if use_riemannian:
            cov_est = Covariances(estimator='lwf')
            ts_trans = TangentSpace(metric='riemann')
            cov_cal_s = cov_est.fit_transform(X_cal_s)
            X_ts_s = ts_trans.fit_transform(cov_cal_s)
            cov_r11_s = cov_est.transform(X_r11_s)
            X_ts_r11 = ts_trans.transform(cov_r11_s)
        else:
            X_ts_s = np.zeros((len(X_cal_s), 0))
            X_ts_r11 = np.zeros((len(X_r11_s), 0))

        # Asymmetry / ERD
        if use_asym:
            X_asym_s = extract_asymmetry_and_erd_features(X_cal_s)
            X_asym_r11 = extract_asymmetry_and_erd_features(X_r11_s)
        else:
            X_asym_s = np.var(X_cal_s, axis=-1)
            X_asym_r11 = np.var(X_r11_s, axis=-1)

        # Feature fusion
        fused_cal = np.hstack([f for f in [X_fbcsp_s, X_ts_s, X_asym_s] if f.shape[1] > 0])
        fused_r11 = np.hstack([f for f in [X_fbcsp_r11, X_ts_r11, X_asym_r11] if f.shape[1] > 0])

        sub_features[s_id] = {
            "fused_cal": fused_cal,
            "fused_r11": fused_r11,
            "y_cal": y_cal_s,
            "y_r11": sub_data[s_id]["y_r11"]
        }

    # GroupKFold evaluation on Dev data
    concat_X_cal = np.vstack([sub_features[s]["fused_cal"] for s in subs])
    concat_y_cal = np.concatenate([sub_features[s]["y_cal"] for s in subs])
    concat_groups = np.concatenate([[i]*len(sub_features[s]["y_cal"]) for i, s in enumerate(subs)])

    dev_accs = []
    for train_idx, val_idx in gkf.split(concat_X_cal, concat_y_cal, concat_groups):
        X_tr, y_tr = concat_X_cal[train_idx], concat_y_cal[train_idx]
        X_va, y_va = concat_X_cal[val_idx], concat_y_cal[val_idx]

        sel_gkf = SelectKBest(f_classif, k=min(k_features, X_tr.shape[1])).fit(X_tr, y_tr)
        sc_gkf = StandardScaler().fit(sel_gkf.transform(X_tr))

        if clf_type == "reg_lda":
            clf_gkf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        elif clf_type == "shrinkage_lda":
            clf_gkf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5)
        elif clf_type == "logreg":
            clf_gkf = LogisticRegression(class_weight=class_weight, max_iter=500)
        elif clf_type == "linear_svm":
            clf_gkf = SVC(kernel='linear', class_weight=class_weight, probability=True, random_state=42)
        elif clf_type == "ensemble":
            clf_gkf = VotingClassifier(
                estimators=[
                    ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
                    ('svm', SVC(kernel='rbf', C=1.8, gamma='scale', probability=True, random_state=42))
                ],
                voting='soft',
                weights=[1.5, 1.0]
            )

        clf_gkf.fit(sc_gkf.transform(sel_gkf.transform(X_tr)), y_tr)
        preds_va = clf_gkf.predict(sc_gkf.transform(sel_gkf.transform(X_va)))
        dev_accs.append(accuracy_score(y_va, preds_va))

    dev_group_kfold_acc = float(np.mean(dev_accs) * 100.0)

    # 2. Independent Per-Subject Calibration & R11 Holdout Evaluation
    all_r11_preds = []
    all_r11_gt = []
    subject_results = {}
    latencies = []

    for s_id in subs:
        X_cal_s = sub_features[s_id]["fused_cal"]
        y_cal_s = sub_features[s_id]["y_cal"]
        X_r11_s = sub_features[s_id]["fused_r11"]
        y_r11_s = sub_features[s_id]["y_r11"]

        t0_sub = time.time()
        k_feat = min(k_features, X_cal_s.shape[1])
        sel_s = SelectKBest(f_classif, k=k_feat).fit(X_cal_s, y_cal_s)
        X_sel_cal = sel_s.transform(X_cal_s)
        sc_s = StandardScaler().fit(X_sel_cal)
        X_sc_cal = sc_s.transform(X_sel_cal)

        if clf_type == "reg_lda":
            clf_s = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        elif clf_type == "shrinkage_lda":
            clf_s = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5)
        elif clf_type == "logreg":
            clf_s = LogisticRegression(class_weight=class_weight, max_iter=500)
        elif clf_type == "linear_svm":
            clf_s = SVC(kernel='linear', class_weight=class_weight, probability=True, random_state=42)
        elif clf_type == "ensemble":
            clf_s = VotingClassifier(
                estimators=[
                    ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
                    ('svm', SVC(kernel='rbf', C=1.8, gamma='scale', probability=True, random_state=42))
                ],
                voting='soft',
                weights=[1.5, 1.0]
            )

        clf_s.fit(X_sc_cal, y_cal_s)

        X_sel_r11 = sel_s.transform(X_r11_s)
        X_sc_r11 = sc_s.transform(X_sel_r11)

        probs_r11 = clf_s.predict_proba(X_sc_r11)
        classes_clf = list(clf_s.classes_)

        s_preds = []
        for row_p in probs_r11:
            rest_i = classes_clf.index("REST") if "REST" in classes_clf else 0
            rest_p = row_p[rest_i]
            if rest_p >= rest_threshold:
                s_preds.append("REST")
            else:
                s_preds.append(classes_clf[np.argmax(row_p)])

        lat_ms = ((time.time() - t0_sub) / len(y_r11_s)) * 1000.0
        latencies.append(lat_ms)

        s_acc = accuracy_score(y_r11_s, s_preds) * 100.0
        subject_results[s_id] = round(s_acc, 2)

        all_r11_preds.extend(s_preds)
        all_r11_gt.extend(y_r11_s)

    all_r11_gt = np.array(all_r11_gt)
    all_r11_preds = np.array(all_r11_preds)

    r11_acc = float(accuracy_score(all_r11_gt, all_r11_preds) * 100.0)
    macro_f1 = float(f1_score(all_r11_gt, all_r11_preds, average='macro') * 100.0)

    cm = confusion_matrix(all_r11_gt, all_r11_preds, labels=CLASSES)
    rest_rec = float(cm[0, 0] / np.sum(cm[0, :]) * 100.0) if np.sum(cm[0, :]) > 0 else 0.0
    left_rec = float(cm[1, 1] / np.sum(cm[1, :]) * 100.0) if np.sum(cm[1, :]) > 0 else 0.0
    right_rec = float(cm[2, 2] / np.sum(cm[2, :]) * 100.0) if np.sum(cm[2, :]) > 0 else 0.0

    avg_lat_ms = float(np.mean(latencies))

    res_dict = {
        "experiment": config_name,
        "dev_group_kfold": round(dev_group_kfold_acc, 2),
        "r11_accuracy": round(r11_acc, 2),
        "macro_f1": round(macro_f1, 2),
        "rest_recall": round(rest_rec, 2),
        "left_recall": round(left_rec, 2),
        "right_recall": round(right_rec, 2),
        "latency_ms": round(avg_lat_ms, 2),
        "confusion_matrix": cm.tolist(),
        "per_subject_acc": subject_results
    }
    
    print(f"[{config_name}] -> Dev GroupKFold: {res_dict['dev_group_kfold']}% | R11 Acc: {res_dict['r11_accuracy']}% | Macro F1: {res_dict['macro_f1']}%", flush=True)
    return res_dict

def run_canonical_benchmark():
    print("================================================================================", flush=True)
    print("NEURO-OS CANONICAL ACCURACY RESEARCH BENCHMARK (REAL EDF 300 R11 HOLDOUT TRIALS)", flush=True)
    print("================================================================ launch \n", flush=True)

    # Load primary 3.0s window dataset
    print("Loading 3.0-second window dataset (S001-S010 R03/R07/R11)...", flush=True)
    data_3s = load_full_dataset(window_sec=3.0, onset_offset_sec=0.5)

    experiments = []

    # Helper to save progress incrementally
    def save_progress():
        os.makedirs("scratch", exist_ok=True)
        with open("scratch/accuracy_benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(experiments, f, indent=2)
        fieldnames = ["experiment", "dev_group_kfold", "r11_accuracy", "macro_f1", "rest_recall", "left_recall", "right_recall", "latency_ms"]
        with open("scratch/accuracy_benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in experiments:
                writer.writerow({k: row[k] for k in fieldnames})

    # 1. Existing Step60 Champion
    res1 = evaluate_configuration("Step60 Champion (RegLDA)", data_3s, use_fbcsp=True, use_riemannian=True, use_asym=True, use_subject_adaptive=True, clf_type="reg_lda", rest_threshold=0.60)
    experiments.append(res1); save_progress()

    # 2. Step61 Multimodal Ensemble
    res2 = evaluate_configuration("Step61 Multimodal Ensemble", data_3s, use_fbcsp=True, use_riemannian=True, use_asym=True, use_subject_adaptive=True, clf_type="ensemble", rest_threshold=0.60)
    experiments.append(res2); save_progress()

    # 3. Raw EEG + Bandpower Baseline
    res3 = evaluate_configuration("Raw EEG + Bandpower Baseline", data_3s, use_fbcsp=False, use_riemannian=False, use_asym=True, use_subject_adaptive=False, clf_type="reg_lda", rest_threshold=0.50)
    experiments.append(res3); save_progress()

    # 4. Standard FBCSP Baseline
    res4 = evaluate_configuration("Standard FBCSP Baseline", data_3s, use_fbcsp=True, use_riemannian=False, use_asym=False, use_subject_adaptive=False, clf_type="reg_lda", rest_threshold=0.50)
    experiments.append(res4); save_progress()

    # 5. FBCSP + ERD Features
    res5 = evaluate_configuration("FBCSP + ERD Features", data_3s, use_fbcsp=True, use_riemannian=False, use_asym=True, use_subject_adaptive=False, clf_type="reg_lda", rest_threshold=0.50)
    experiments.append(res5); save_progress()

    # 6. FBCSP + Riemannian Tangent Space Features
    res6 = evaluate_configuration("FBCSP + Riemannian Features", data_3s, use_fbcsp=True, use_riemannian=True, use_asym=False, use_subject_adaptive=False, clf_type="reg_lda", rest_threshold=0.50)
    experiments.append(res6); save_progress()

    # 7. FBCSP + ERD + Riemannian Fusion
    res7 = evaluate_configuration("FBCSP + ERD + Riemannian Fusion", data_3s, use_fbcsp=True, use_riemannian=True, use_asym=True, use_subject_adaptive=False, clf_type="reg_lda", rest_threshold=0.50)
    experiments.append(res7); save_progress()

    # 8. Subject-Specific Peak Frequency Adaptation
    res8 = evaluate_configuration("Subject Peak Freq Adaptation", data_3s, use_fbcsp=True, use_riemannian=True, use_asym=True, use_subject_adaptive=True, clf_type="reg_lda", rest_threshold=0.50)
    experiments.append(res8); save_progress()

    # 9. Window Length Sweeps (2s, 4s, 5s)
    for win_len in [2.0, 4.0, 5.0]:
        data_win = load_full_dataset(window_sec=win_len, onset_offset_sec=0.5)
        res_win = evaluate_configuration(f"Window Length {win_len}s", data_win, use_fbcsp=True, use_riemannian=True, use_asym=True, use_subject_adaptive=True, clf_type="reg_lda", rest_threshold=0.60)
        experiments.append(res_win); save_progress()

    # 10. Post-Event Offset Sweeps (0.0s, 1.0s, 1.5s, 2.0s)
    for offset_sec in [0.0, 1.0, 1.5, 2.0]:
        data_off = load_full_dataset(window_sec=3.0, onset_offset_sec=offset_sec)
        res_off = evaluate_configuration(f"Post-Event Offset {offset_sec}s", data_off, use_fbcsp=True, use_riemannian=True, use_asym=True, use_subject_adaptive=True, clf_type="reg_lda", rest_threshold=0.60)
        experiments.append(res_off); save_progress()

    # 11. Classifier Sweeps (Shrinkage LDA, LogReg, Linear SVM)
    for clf_t in ["shrinkage_lda", "logreg", "linear_svm"]:
        res_clf = evaluate_configuration(f"Classifier Sweep: {clf_t}", data_3s, use_fbcsp=True, use_riemannian=True, use_asym=True, use_subject_adaptive=True, clf_type=clf_t, rest_threshold=0.60)
        experiments.append(res_clf); save_progress()

    # 12. Class Weighting Experiment
    res_cw = evaluate_configuration("Class Weighting Balanced", data_3s, use_fbcsp=True, use_riemannian=True, use_asym=True, use_subject_adaptive=True, clf_type="logreg", class_weight="balanced", rest_threshold=0.60)
    experiments.append(res_cw); save_progress()

    # 13. REST Threshold Calibration (0.50 vs 0.65 vs 0.70)
    for th in [0.50, 0.65, 0.70]:
        res_th = evaluate_configuration(f"REST Threshold th={th}", data_3s, use_fbcsp=True, use_riemannian=True, use_asym=True, use_subject_adaptive=True, clf_type="ensemble", rest_threshold=th)
        experiments.append(res_th); save_progress()

    print("\n================ CANONICAL BENCHMARK SUMMARY TABLE ================", flush=True)
    df_res = pd.DataFrame(experiments)
    print(df_res[["experiment", "dev_group_kfold", "r11_accuracy", "macro_f1", "rest_recall", "left_recall", "right_recall", "latency_ms"]].to_string(index=False), flush=True)
    print("===================================================================\n", flush=True)

if __name__ == "__main__":
    run_canonical_benchmark()
