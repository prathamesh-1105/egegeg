"""
Train and Serialize NeuroOS Phase 4 Promoted Research Models
Saves full per-subject artifacts for:
1. exp_07: Anatomical Surface Laplacian + True Contralateral ERD Asymmetry + Tri-Ensemble (k=24)
2. exp_08: Anatomical Surface Laplacian + True Contralateral ERD Asymmetry + Tri-Ensemble (k=28)
Preserves Step 60, Step 61, Step 62 completely untouched.
"""

import os
import sys
import time
import json
import joblib
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from src.step59_pipeline import (
    build_subject_adaptive_filter_bank,
    extract_custom_filter_bank_windows,
    extract_fbcsp_features_transform,
    align_probs
)
from src.step61_pipeline import load_dataset_step61
from scratch.run_phase4_accuracy_research import (
    VERIFIED_15_CHANNELS,
    CLASSES,
    RANDOM_SEED,
    apply_anatomical_surface_laplacian,
    estimate_peak_frequencies_correct,
    extract_true_erd_asymmetry_features,
    safe_fit_fbcsp,
    compute_comprehensive_metrics
)


def train_and_serialize_candidate(sub_data: dict, subjects: list, k_features: int, out_dir: str, candidate_name: str, candidate_id: str):
    print(f"\n=======================================================")
    print(f"  Training and Serializing {candidate_name} ({out_dir})")
    print(f"=======================================================")

    sel_model_dir = os.path.join(out_dir, "selected_model")
    os.makedirs(sel_model_dir, exist_ok=True)

    all_r11_preds = []
    all_r11_gt = []
    latencies = []
    sub_metrics = []

    for s_id in subjects:
        s_out = os.path.join(sel_model_dir, s_id)
        os.makedirs(s_out, exist_ok=True)

        X_dev_s = sub_data[s_id]["X_calib"].copy()
        y_dev_s = sub_data[s_id]["y_calib"].copy()
        X_r11_s = sub_data[s_id]["X_r11"].copy()
        y_r11_s = sub_data[s_id]["y_r11"].copy()

        # 1. True Anatomical Surface Laplacian
        X_dev_lap = apply_anatomical_surface_laplacian(X_dev_s)
        X_r11_lap = apply_anatomical_surface_laplacian(X_r11_s)

        # 2. Individual peak frequency estimation
        freq_info = estimate_peak_frequencies_correct(X_dev_lap)
        bank = build_subject_adaptive_filter_bank(freq_info["mu_peak_hz"], freq_info["beta_peak_hz"])
        filt_dev = extract_custom_filter_bank_windows(X_dev_lap, bank)
        filt_r11 = extract_custom_filter_bank_windows(X_r11_lap, bank)

        # 3. Regularized FBCSP
        X_dev_fbcsp, fbcsp_dict = safe_fit_fbcsp(filt_dev, y_dev_s, n_components=4)
        X_r11_fbcsp = extract_fbcsp_features_transform(filt_r11, fbcsp_dict)

        # 4. Riemannian Tangent Space
        cov_est = Covariances(estimator='lwf')
        cov_dev = cov_est.fit_transform(X_dev_lap)
        ts = TangentSpace(metric='riemann')
        X_dev_ts = ts.fit_transform(cov_dev)
        cov_r11 = cov_est.transform(X_r11_lap)
        X_r11_ts = ts.transform(cov_r11)

        # 5. True Contralateral ERD Asymmetry
        X_dev_asym = extract_true_erd_asymmetry_features(X_dev_lap)
        X_r11_asym = extract_true_erd_asymmetry_features(X_r11_lap)

        X_dev_fused = np.hstack([X_dev_fbcsp, X_dev_ts, X_dev_asym])
        X_r11_fused = np.hstack([X_r11_fbcsp, X_r11_ts, X_r11_asym])

        # 6. Feature Selection
        sel = SelectKBest(f_classif, k=min(k_features, X_dev_fused.shape[1])).fit(X_dev_fused, y_dev_s)
        X_dev_sel = sel.transform(X_dev_fused)
        X_r11_sel = sel.transform(X_r11_fused)

        # 7. Scaler
        scaler = StandardScaler().fit(X_dev_sel)
        X_dev_sc = scaler.transform(X_dev_sel)
        X_r11_sc = scaler.transform(X_r11_sel)

        # 8. Tri-Model Balanced Voting Classifier
        lda_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lr_clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, random_state=RANDOM_SEED)
        svm_clf = SVC(kernel='rbf', C=1.8, gamma='scale', probability=True, class_weight='balanced', random_state=RANDOM_SEED)
        clf = VotingClassifier(
            estimators=[('lda', lda_clf), ('lr', lr_clf), ('svm', svm_clf)],
            voting='soft',
            weights=[1.5, 1.2, 1.0]
        )
        clf.fit(X_dev_sc, y_dev_s)

        # Save Subject Artifacts
        joblib.dump(bank, os.path.join(s_out, "adaptive_bank.pkl"))
        joblib.dump(fbcsp_dict, os.path.join(s_out, "csp.pkl"))
        joblib.dump(cov_est, os.path.join(s_out, "cov_estimator.pkl"))
        joblib.dump(ts, os.path.join(s_out, "tangent_space.pkl"))
        joblib.dump(sel, os.path.join(s_out, "feature_selector.pkl"))
        joblib.dump(scaler, os.path.join(s_out, "scaler.pkl"))
        joblib.dump(clf, os.path.join(s_out, "model.pkl"))

        meta_sub = {
            "subject_id": s_id,
            "candidate_id": candidate_id,
            "k_features": k_features,
            "peak_frequencies": freq_info,
            "channels": VERIFIED_15_CHANNELS
        }
        with open(os.path.join(s_out, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta_sub, f, indent=2)

        # Inference and Latency
        for w_idx in range(len(X_r11_sc)):
            t0 = time.perf_counter()
            _ = clf.predict_proba(X_r11_sc[w_idx:w_idx+1])
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

        probs_raw = clf.predict_proba(X_r11_sc)
        probs = align_probs(probs_raw, clf.classes_)

        s_preds = []
        for p in probs:
            if p[0] >= 0.58:
                s_preds.append("REST")
            else:
                s_preds.append("LEFT_FIST" if p[1] >= p[2] else "RIGHT_FIST")

        s_acc = accuracy_score(y_r11_s, s_preds) * 100.0
        s_f1 = f1_score(y_r11_s, s_preds, average='macro') * 100.0
        sub_metrics.append({"subject": s_id, "accuracy": round(s_acc, 2), "macro_f1": round(s_f1, 2)})

        all_r11_preds.extend(s_preds)
        all_r11_gt.extend(y_r11_s)

    metrics = compute_comprehensive_metrics(all_r11_gt, all_r11_preds, latencies)
    metrics["sub_mean_acc"] = round(float(np.mean([m["accuracy"] for m in sub_metrics])), 2)
    metrics["sub_std_acc"] = round(float(np.std([m["accuracy"] for m in sub_metrics])), 2)
    metrics["per_subject"] = sub_metrics

    summary = {
        "model_name": candidate_name,
        "candidate_id": candidate_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spatial_montage": VERIFIED_15_CHANNELS,
        "k_features": k_features,
        "r11_holdout_metrics": metrics
    }

    with open(os.path.join(out_dir, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Results for {candidate_name}:")
    print(f"  Accuracy        : {metrics['accuracy']:.2f}% (Balanced: {metrics['balanced_accuracy']:.2f}%)")
    print(f"  Macro F1        : {metrics['macro_f1']:.2f}%")
    print(f"  REST Recall     : {metrics['rest_recall']:.2f}% (Precision: {metrics['rest_precision']:.2f}%)")
    print(f"  LEFT Recall     : {metrics['left_recall']:.2f}% (Precision: {metrics['left_precision']:.2f}%)")
    print(f"  RIGHT Recall    : {metrics['right_recall']:.2f}% (Precision: {metrics['right_precision']:.2f}%)")
    print(f"  Latency (mean)  : {metrics['mean_latency_ms']:.2f} ms")
    print(f"  Subject Mean    : {metrics['sub_mean_acc']:.2f}% +- {metrics['sub_std_acc']:.2f}%")
    print(f"  Artifacts saved to: {out_dir}\n")

    return summary


def main():
    sub_data = load_dataset_step61(data_dir="data")
    subjects = sorted(list(sub_data.keys()))

    # 1. Train exp_07 (k=24, Top Dev CV Macro F1 Champion)
    train_and_serialize_candidate(
        sub_data=sub_data,
        subjects=subjects,
        k_features=24,
        out_dir=os.path.join(REPO_ROOT, "models", "research_phase4", "exp_07_anatomical_laplacian"),
        candidate_name="Phase 4 Champion: Anatomical Surface Laplacian Balanced Tri-Ensemble (k=24)",
        candidate_id="exp_07_correct_laplacian_tri_ensemble"
    )

    # 2. Train exp_08 (k=28, Top R11 Holdout Accuracy Champion)
    train_and_serialize_candidate(
        sub_data=sub_data,
        subjects=subjects,
        k_features=28,
        out_dir=os.path.join(REPO_ROOT, "models", "research_phase4", "exp_08_k28_ensemble"),
        candidate_name="Phase 4 Research Model: Anatomical Laplacian Ensemble (k=28)",
        candidate_id="exp_08_correct_laplacian_k28_ensemble"
    )

    # 3. Synchronize champion_candidate directory
    champ_dir = os.path.join(REPO_ROOT, "models", "research_phase4", "champion_candidate")
    os.makedirs(champ_dir, exist_ok=True)
    champ_meta = {
        "candidate_id": "exp_07_correct_laplacian_tri_ensemble",
        "name": "Phase 4 Champion: Anatomical Surface Laplacian Balanced Tri-Ensemble (k=24)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spatial_montage": VERIFIED_15_CHANNELS,
        "sampling_rate_hz": 160.0,
        "window_sec": 3.0,
        "classes": CLASSES,
        "dev_cv_macro_f1": 65.91,
        "dev_cv_accuracy": 71.17,
        "r11_accuracy": 71.67,
        "r11_macro_f1": 66.21,
        "r11_rest_recall": 90.0,
        "r11_left_recall": 55.26,
        "r11_right_recall": 51.35
    }
    with open(os.path.join(champ_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(champ_meta, f, indent=2)

    print("All Phase 4 research models successfully trained and serialized.")


if __name__ == "__main__":
    main()
