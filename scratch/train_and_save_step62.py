"""
Train and Serialize Step 62 Model Artifacts (Surface Laplacian + Enhanced Fused + Balanced Tri-Ensemble)
Model Architecture:
- Spatial Referencing: Surface Laplacian Approximation
- Feature Extraction: Adaptive FBCSP (4 components) + Riemannian Tangent Space + Enhanced ERD Asymmetry
- Feature Selection: SelectKBest(k=24)
- Normalization: StandardScaler
- Classifier: Tri-Model Soft Voting Classifier (Regularized LDA, Balanced Logistic Regression, Balanced RBF SVM)
- Preserves Step 60 and Step 61 artifacts untouched.
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from src.step33_pipeline import GROUP_B_CHANNELS, CLASSES
from src.step59_pipeline import (
    estimate_subject_peak_frequency,
    build_subject_adaptive_filter_bank,
    extract_custom_filter_bank_windows,
    extract_fbcsp_features_transform,
    align_probs
)
from src.step61_pipeline import load_dataset_step61
from scratch.run_phase2b_accuracy_experiments import (
    apply_surface_laplacian_approx,
    extract_enhanced_erd_asymmetry,
    safe_extract_fbcsp_features_train
)

def train_and_save_step62():
    print("=== Training and Serializing Step 62 Champion Model ===")
    sub_data = load_dataset_step61(data_dir="data")
    subs = sorted(list(sub_data.keys()))

    out_base = os.path.join(REPO_ROOT, "models", "step62", "selected_model")
    os.makedirs(out_base, exist_ok=True)

    y_true_all = []
    y_pred_all = []
    latencies = []
    sub_metrics = []

    for s_id in subs:
        s_out = os.path.join(out_base, s_id)
        os.makedirs(s_out, exist_ok=True)

        X_dev = sub_data[s_id]["X_calib"].copy()
        y_dev = sub_data[s_id]["y_calib"].copy()
        X_r11 = sub_data[s_id]["X_r11"].copy()
        y_r11 = sub_data[s_id]["y_r11"].copy()

        # 1. Spatial Referencing
        X_dev_lap = apply_surface_laplacian_approx(X_dev)
        X_r11_lap = apply_surface_laplacian_approx(X_r11)

        # 2. Filter Bank
        freq_info = estimate_subject_peak_frequency(X_dev_lap, sfreq=160.0)
        bank = build_subject_adaptive_filter_bank(freq_info["mu_peak_hz"], freq_info["beta_peak_hz"])
        filt_dev = extract_custom_filter_bank_windows(X_dev_lap, bank)
        filt_r11 = extract_custom_filter_bank_windows(X_r11_lap, bank)

        # 3. FBCSP
        X_dev_fbcsp, fbcsp_dict = safe_extract_fbcsp_features_train(filt_dev, y_dev, n_components=4)
        X_r11_fbcsp = extract_fbcsp_features_transform(filt_r11, fbcsp_dict)

        # 4. Tangent Space
        cov_est = Covariances(estimator='lwf')
        cov_dev = cov_est.fit_transform(X_dev_lap)
        ts = TangentSpace(metric='riemann')
        X_dev_ts = ts.fit_transform(cov_dev)
        cov_r11 = cov_est.transform(X_r11_lap)
        X_r11_ts = ts.transform(cov_r11)

        # 5. Enhanced ERD Asymmetry
        X_dev_asym = extract_enhanced_erd_asymmetry(X_dev_lap)
        X_r11_asym = extract_enhanced_erd_asymmetry(X_r11_lap)

        X_dev_fused = np.hstack([X_dev_fbcsp, X_dev_ts, X_dev_asym])
        X_r11_fused = np.hstack([X_r11_fbcsp, X_r11_ts, X_r11_asym])

        # 6. Feature Selection
        sel = SelectKBest(f_classif, k=24)
        X_dev_sel = sel.fit_transform(X_dev_fused, y_dev)
        X_r11_sel = sel.transform(X_r11_fused)

        # 7. Scaler
        scaler = StandardScaler()
        X_dev_sc = scaler.fit_transform(X_dev_sel)
        X_r11_sc = scaler.transform(X_r11_sel)

        # 8. Tri-Model Balanced Voting Classifier
        lda_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lr_clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, random_state=42)
        svm_clf = SVC(kernel='rbf', C=1.8, gamma='scale', probability=True, class_weight='balanced', random_state=42)
        clf = VotingClassifier(
            estimators=[('lda', lda_clf), ('lr', lr_clf), ('svm', svm_clf)],
            voting='soft',
            weights=[1.5, 1.2, 1.0]
        )
        clf.fit(X_dev_sc, y_dev)

        # Save Subject Artifacts
        joblib.dump(bank, os.path.join(s_out, "adaptive_bank.pkl"))
        joblib.dump(fbcsp_dict, os.path.join(s_out, "csp.pkl"))
        joblib.dump(cov_est, os.path.join(s_out, "cov_estimator.pkl"))
        joblib.dump(ts, os.path.join(s_out, "tangent_space.pkl"))
        joblib.dump(sel, os.path.join(s_out, "feature_selector.pkl"))
        joblib.dump(scaler, os.path.join(s_out, "scaler.pkl"))
        joblib.dump(clf, os.path.join(s_out, "model.pkl"))

        # Latency check
        for w_idx in range(len(X_r11)):
            t0 = time.perf_counter()
            w_single = X_r11_lap[w_idx:w_idx+1]
            w_filt = extract_custom_filter_bank_windows(w_single, bank)
            w_fbcsp = extract_fbcsp_features_transform(w_filt, fbcsp_dict)
            w_cov = cov_est.transform(w_single)
            w_ts = ts.transform(w_cov)
            w_asym = extract_enhanced_erd_asymmetry(w_single)
            w_fused = np.hstack([w_fbcsp, w_ts, w_asym])
            w_sel = sel.transform(w_fused)
            w_sc = scaler.transform(w_sel)
            _ = clf.predict_proba(w_sc)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

        probs_raw = clf.predict_proba(X_r11_sc)
        probs = align_probs(probs_raw, clf.classes_)

        s_preds = []
        for p in probs:
            if p[0] >= 0.60:
                s_preds.append("REST")
            else:
                s_preds.append("LEFT_FIST" if p[1] >= p[2] else "RIGHT_FIST")

        s_acc = accuracy_score(y_r11, s_preds) * 100
        s_f1 = f1_score(y_r11, s_preds, average='macro') * 100
        sub_metrics.append({"subject": s_id, "accuracy": round(s_acc, 2), "macro_f1": round(s_f1, 2)})

        y_true_all.extend(y_r11)
        y_pred_all.extend(s_preds)

    acc = accuracy_score(y_true_all, y_pred_all) * 100
    f1_m = f1_score(y_true_all, y_pred_all, average="macro") * 100
    cm = confusion_matrix(y_true_all, y_pred_all, labels=CLASSES)
    rest_rec = cm[0, 0] / np.sum(cm[0, :]) * 100
    left_rec = cm[1, 1] / np.sum(cm[1, :]) * 100
    right_rec = cm[2, 2] / np.sum(cm[2, :]) * 100

    report = {
        "model_name": "Step 62 Surface Laplacian Multimodal Balanced Ensemble",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": {
            "spatial_referencing": "surface_laplacian_approx",
            "feature_family": "enhanced_fused (Adaptive FBCSP + Riemannian Tangent Space + Enhanced ERD Asymmetry)",
            "k_features": 24,
            "classifier": "VotingClassifier (Reg-LDA, Balanced-LR, Balanced-RBF-SVM)",
            "weights": [1.5, 1.2, 1.0]
        },
        "holdout_results_r11": {
            "accuracy": round(acc, 2),
            "macro_f1": round(f1_m, 2),
            "rest_recall": round(rest_rec, 2),
            "left_fist_recall": round(left_rec, 2),
            "right_fist_recall": round(right_rec, 2),
            "subject_mean_acc": round(float(np.mean([m['accuracy'] for m in sub_metrics])), 2),
            "subject_std_acc": round(float(np.std([m['accuracy'] for m in sub_metrics])), 2),
            "mean_inference_latency_ms": round(float(np.mean(latencies)), 2),
            "p95_inference_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "confusion_matrix": cm.tolist()
        },
        "per_subject": sub_metrics,
        "classes": CLASSES
    }

    report_path = os.path.join(REPO_ROOT, "models", "step62", "metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nStep 62 Successfully Saved! Report saved to {report_path}")
    print(f"Overall Accuracy: {acc:.2f}%, Macro F1: {f1_m:.2f}%")
    print(f"Rest Recall: {rest_rec:.2f}%, Left: {left_rec:.2f}%, Right: {right_rec:.2f}%")

if __name__ == "__main__":
    train_and_save_step62()
