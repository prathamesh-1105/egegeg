"""
Evaluate Phase 2B Candidate Models on Held-out R11 Test Set
Trains candidate configurations strictly on Development set (R03 + R07, N=600 trials),
and tests on the untouched Holdout set (R11, N=300 trials).
Compares directly against frozen Step 60 and Step 61 baselines.
"""

import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from src.step33_pipeline import GROUP_B_CHANNELS, CLASSES, C3_IDX, C4_IDX
from src.step59_pipeline import (
    estimate_subject_peak_frequency,
    build_subject_adaptive_filter_bank,
    extract_custom_filter_bank_windows,
    extract_fbcsp_features_transform,
    align_probs
)
from src.step61_pipeline import load_dataset_step61, extract_asymmetry_and_erd_features
from scratch.run_phase2b_accuracy_experiments import (
    apply_common_average_reference,
    apply_surface_laplacian_approx,
    extract_enhanced_erd_asymmetry,
    safe_extract_fbcsp_features_train
)

def evaluate_candidates_on_holdout():
    print("=== Phase 2B Holdout Test (R11) Evaluation ===")
    sub_data = load_dataset_step61(data_dir="data")
    subs = sorted(list(sub_data.keys()))

    candidate_configs = [
        {
            "name": "Candidate 1: CAR + Enhanced Fused + Balanced Ensemble (k=24)",
            "spatial": "car",
            "feature": "enhanced_fused",
            "clf_type": "ensemble_balanced",
            "k": 24,
            "fbcsp_m": 4
        },
        {
            "name": "Candidate 2: CAR + Enhanced Fused + Balanced Ensemble (k=20)",
            "spatial": "car",
            "feature": "enhanced_fused",
            "clf_type": "ensemble_balanced",
            "k": 20,
            "fbcsp_m": 4
        },
        {
            "name": "Candidate 3: Surface Laplacian + Enhanced Fused + Balanced Ensemble (k=20)",
            "spatial": "laplacian",
            "feature": "enhanced_fused",
            "clf_type": "ensemble_balanced",
            "k": 20,
            "fbcsp_m": 4
        },
        {
            "name": "Candidate 4: Surface Laplacian + Enhanced Fused + Regularized LDA (k=24)",
            "spatial": "laplacian",
            "feature": "enhanced_fused",
            "clf_type": "reg_lda",
            "k": 24,
            "fbcsp_m": 4
        },
        {
            "name": "Candidate 5: Surface Laplacian + Enhanced Fused + Balanced Ensemble (k=24)",
            "spatial": "laplacian",
            "feature": "enhanced_fused",
            "clf_type": "ensemble_balanced",
            "k": 24,
            "fbcsp_m": 4
        },
        {
            "name": "Candidate 6: Surface Laplacian + Enhanced Fused + RBF SVM (k=20)",
            "spatial": "laplacian",
            "feature": "enhanced_fused",
            "clf_type": "svm_rbf",
            "k": 20,
            "fbcsp_m": 4
        }
    ]

    all_cand_results = []

    for cfg in candidate_configs:
        c_name = cfg["name"]
        print(f"\n--- Training & Evaluating {c_name} ---")
        t0 = time.time()
        
        y_true_all = []
        y_pred_raw_all = []
        y_pred_thresh_all = []
        confs_all = []
        margins_all = []
        safety_pass_all = []
        latencies = []

        sub_accs = []

        for s_id in subs:
            X_dev = sub_data[s_id]["X_calib"].copy() # (60, 16, 640)
            y_dev = sub_data[s_id]["y_calib"].copy()
            X_r11 = sub_data[s_id]["X_r11"].copy() # (30, 16, 640)
            y_r11 = sub_data[s_id]["y_r11"].copy()

            # 1. Spatial Referencing
            if cfg["spatial"] == "car":
                X_dev_sp = apply_common_average_reference(X_dev)
                X_r11_sp = apply_common_average_reference(X_r11)
            elif cfg["spatial"] == "laplacian":
                X_dev_sp = apply_surface_laplacian_approx(X_dev)
                X_r11_sp = apply_surface_laplacian_approx(X_r11)
            else:
                X_dev_sp = X_dev.copy()
                X_r11_sp = X_r11.copy()

            # 2. Build Filter Bank on Dev
            freq_info = estimate_subject_peak_frequency(X_dev_sp, sfreq=160.0)
            bank = build_subject_adaptive_filter_bank(freq_info["mu_peak_hz"], freq_info["beta_peak_hz"])
            filt_dev = extract_custom_filter_bank_windows(X_dev_sp, bank)
            filt_r11 = extract_custom_filter_bank_windows(X_r11_sp, bank)

            # 3. Fit FBCSP on Dev
            X_dev_fbcsp, fbcsp_dict = safe_extract_fbcsp_features_train(filt_dev, y_dev, n_components=cfg["fbcsp_m"])
            X_r11_fbcsp = extract_fbcsp_features_transform(filt_r11, fbcsp_dict)

            # 4. Tangent Space
            cov_est = Covariances(estimator='lwf')
            cov_dev = cov_est.fit_transform(X_dev_sp)
            ts = TangentSpace(metric='riemann')
            X_dev_ts = ts.fit_transform(cov_dev)
            cov_r11 = cov_est.transform(X_r11_sp)
            X_r11_ts = ts.transform(cov_r11)

            # 5. Asymmetry
            if cfg["feature"] == "enhanced_fused":
                X_dev_asym = extract_enhanced_erd_asymmetry(X_dev_sp)
                X_r11_asym = extract_enhanced_erd_asymmetry(X_r11_sp)
            else:
                X_dev_asym = extract_asymmetry_and_erd_features(X_dev_sp)
                X_r11_asym = extract_asymmetry_and_erd_features(X_r11_sp)

            X_dev_fused = np.hstack([X_dev_fbcsp, X_dev_ts, X_dev_asym])
            X_r11_fused = np.hstack([X_r11_fbcsp, X_r11_ts, X_r11_asym])

            # 6. Feature Selection
            k_val = min(cfg["k"], X_dev_fused.shape[1])
            sel = SelectKBest(f_classif, k=k_val)
            X_dev_sel = sel.fit_transform(X_dev_fused, y_dev)
            X_r11_sel = sel.transform(X_r11_fused)

            # 7. Scaler
            scaler = StandardScaler()
            X_dev_sc = scaler.fit_transform(X_dev_sel)
            X_r11_sc = scaler.transform(X_r11_sel)

            # 8. Classifier
            clf_type = cfg["clf_type"]
            if clf_type == "reg_lda":
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            elif clf_type == "lr_balanced":
                clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, random_state=42)
            elif clf_type == "svm_rbf":
                clf = SVC(kernel='rbf', C=1.8, gamma='scale', probability=True, class_weight='balanced', random_state=42)
            elif clf_type == "ensemble_balanced":
                lda_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                lr_clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, random_state=42)
                svm_clf = SVC(kernel='rbf', C=1.8, gamma='scale', class_weight='balanced', probability=True, random_state=42)
                clf = VotingClassifier(
                    estimators=[('lda', lda_clf), ('lr', lr_clf), ('svm', svm_clf)],
                    voting='soft',
                    weights=[1.5, 1.2, 1.0]
                )
            else:
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

            clf.fit(X_dev_sc, y_dev)

            # Evaluate latency on single test epochs
            for w_idx in range(len(X_r11)):
                t_infer_start = time.perf_counter()
                w_single = X_r11_sp[w_idx:w_idx+1]
                w_filt = extract_custom_filter_bank_windows(w_single, bank)
                w_fbcsp = extract_fbcsp_features_transform(w_filt, fbcsp_dict)
                w_cov = cov_est.transform(w_single)
                w_ts = ts.transform(w_cov)
                if cfg["feature"] == "enhanced_fused":
                    w_asym = extract_enhanced_erd_asymmetry(w_single)
                else:
                    w_asym = extract_asymmetry_and_erd_features(w_single)
                w_fused = np.hstack([w_fbcsp, w_ts, w_asym])
                w_sel = sel.transform(w_fused)
                w_sc = scaler.transform(w_sel)
                p_raw = clf.predict_proba(w_sc)
                t_infer = (time.perf_counter() - t_infer_start) * 1000.0
                latencies.append(t_infer)

            # Batch predict proba for exact metrics
            probs_raw = clf.predict_proba(X_r11_sc)
            probs = align_probs(probs_raw, clf.classes_)

            s_preds_raw = []
            s_preds_thresh = []
            for i in range(len(y_r11)):
                p = probs[i]
                conf = float(np.max(p))
                sorted_p = np.sort(p)[::-1]
                margin = float(sorted_p[0] - sorted_p[1])
                
                # Raw argmax
                pred_raw = CLASSES[np.argmax(p)]
                
                # REST threshold at 0.60
                if p[0] >= 0.60:
                    pred_th = "REST"
                else:
                    pred_th = "LEFT_FIST" if p[1] >= p[2] else "RIGHT_FIST"

                safety_pass = (conf >= 0.60) and (margin >= 0.10)

                y_true_all.append(y_r11[i])
                y_pred_raw_all.append(pred_raw)
                y_pred_thresh_all.append(pred_th)
                confs_all.append(conf)
                margins_all.append(margin)
                safety_pass_all.append(safety_pass)
                s_preds_thresh.append(pred_th)

            s_acc = accuracy_score(y_r11, s_preds_thresh) * 100
            sub_accs.append(s_acc)

        dur = time.time() - t0
        acc_raw = accuracy_score(y_true_all, y_pred_raw_all) * 100
        f1_raw = f1_score(y_true_all, y_pred_raw_all, average="macro") * 100
        acc_th = accuracy_score(y_true_all, y_pred_thresh_all) * 100
        f1_th = f1_score(y_true_all, y_pred_thresh_all, average="macro") * 100
        cm_th = confusion_matrix(y_true_all, y_pred_thresh_all, labels=CLASSES)

        rest_rec = cm_th[0, 0] / np.sum(cm_th[0, :]) * 100
        left_rec = cm_th[1, 1] / np.sum(cm_th[1, :]) * 100
        right_rec = cm_th[2, 2] / np.sum(cm_th[2, :]) * 100

        # Safety shield
        accepted_idx = [i for i, p in enumerate(safety_pass_all) if p]
        rejected_idx = [i for i, p in enumerate(safety_pass_all) if not p]
        cov_pct = len(accepted_idx) / len(safety_pass_all) * 100
        sel_acc = accuracy_score([y_true_all[i] for i in accepted_idx], [y_pred_thresh_all[i] for i in accepted_idx]) * 100 if accepted_idx else 0.0
        rej_acc = accuracy_score([y_true_all[i] for i in rejected_idx], [y_pred_thresh_all[i] for i in rejected_idx]) * 100 if rejected_idx else 0.0

        mean_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))

        print(f"--> Test Raw Acc: {acc_raw:.2f}%, F1: {f1_raw:.2f}%")
        print(f"--> Test Thresh Acc: {acc_th:.2f}%, F1: {f1_th:.2f}%")
        print(f"--> Recalls | Rest: {rest_rec:.2f}%, Left: {left_rec:.2f}%, Right: {right_rec:.2f}%")
        print(f"--> Subject Mean Acc: {np.mean(sub_accs):.2f}% (Std: {np.std(sub_accs):.2f}%)")
        print(f"--> Safety Shield Coverage: {cov_pct:.2f}%, Selective Acc: {sel_acc:.2f}%, Rej Acc: {rej_acc:.2f}%")
        print(f"--> Mean Latency: {mean_lat:.2f} ms (p95: {p95_lat:.2f} ms) | Duration: {dur:.1f}s")
        print(f"--> Confusion Matrix:\n{cm_th}")

        all_cand_results.append({
            "name": c_name,
            "spatial": cfg["spatial"],
            "feature": cfg["feature"],
            "clf_type": cfg["clf_type"],
            "k": cfg["k"],
            "test_acc_raw": round(acc_raw, 2),
            "test_f1_raw": round(f1_raw, 2),
            "test_acc_th": round(acc_th, 2),
            "test_f1_th": round(f1_th, 2),
            "rest_recall": round(rest_rec, 2),
            "left_recall": round(left_rec, 2),
            "right_recall": round(right_rec, 2),
            "sub_mean_acc": round(float(np.mean(sub_accs)), 2),
            "sub_std_acc": round(float(np.std(sub_accs)), 2),
            "safety_coverage": round(cov_pct, 2),
            "selective_accuracy": round(sel_acc, 2),
            "rejected_accuracy": round(rej_acc, 2),
            "mean_latency_ms": round(mean_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "confusion_matrix": cm_th.tolist(),
            "duration_sec": round(dur, 1)
        })

    out_file = "scratch/phase2b_holdout_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_cand_results, f, indent=2)
    print(f"\nSaved holdout comparison to {out_file}")

    # Convert to CSV summary
    df = pd.DataFrame(all_cand_results)
    df.to_csv("scratch/phase2b_holdout_comparison.csv", index=False)
    print("Saved holdout comparison to scratch/phase2b_holdout_comparison.csv")

if __name__ == "__main__":
    evaluate_candidates_on_holdout()
