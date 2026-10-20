"""
Train and Serialize NeuroOS Phase 5 Promoted Research Models
Saves full per-subject artifacts for:
1. p5_advanced_motor_champion: Anatomical Surface Laplacian + Advanced Contralateral Motor Features + Balanced Tri-Ensemble (k=32)
   - Dev CV: 72.17% Acc, 67.52% Macro F1
   - R11 Holdout: 74.67% Acc, 70.22% Macro F1, 91.33% REST Recall, 8.67% False Actuation, 0.64 ms latency
2. p5_calibrated_hierarchical: Calibrated Two-Stage Hierarchical Intent-Lateralization Decoder
   - Dev CV: 72.67% Acc, 68.34% Macro F1, 66.23% Left Recall

Preserves models/step60/, models/step61/, models/step62/, and models/research_phase4/ completely untouched.
"""

import os
import sys
import time
import json
import joblib
import warnings
warnings.filterwarnings("ignore")
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
    safe_fit_fbcsp
)
from scratch.run_phase5_accuracy_research import (
    extract_advanced_motor_features,
    CalibratedHierarchicalBCI,
    compute_metrics
)


def train_and_serialize_phase5_advanced_motor(sub_data: dict, subjects: list, out_dir: str):
    print(f"\n=======================================================")
    print(f"  Training & Serializing Phase 5 Advanced Motor Champion")
    print(f"  Destination: {out_dir}")
    print(f"=======================================================")

    sel_model_dir = os.path.join(out_dir, "selected_model")
    os.makedirs(sel_model_dir, exist_ok=True)

    all_r11_preds, all_r11_gt, latencies = [], [], []
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

        # 2. Filter Bank
        bank = build_subject_adaptive_filter_bank(10.0, 20.0)
        filt_dev = extract_custom_filter_bank_windows(X_dev_lap, bank)
        filt_r11 = extract_custom_filter_bank_windows(X_r11_lap, bank)

        # 3. Filter Bank CSP
        X_dev_fbcsp, fbcsp_dict = safe_fit_fbcsp(filt_dev, y_dev_s, n_components=4)
        X_r11_fbcsp = extract_fbcsp_features_transform(filt_r11, fbcsp_dict)

        # 4. Riemannian Tangent Space
        cov_est = Covariances(estimator='lwf')
        cov_dev = cov_est.fit_transform(X_dev_lap)
        ts = TangentSpace(metric='riemann')
        X_dev_ts = ts.fit_transform(cov_dev)
        X_r11_ts = ts.transform(cov_est.transform(X_r11_lap))

        # 5. Advanced Contralateral Motor Features
        X_dev_asym = extract_advanced_motor_features(X_dev_lap)
        X_r11_asym = extract_advanced_motor_features(X_r11_lap)

        # Multimodal fusion
        X_dev_fused = np.hstack([X_dev_fbcsp, X_dev_ts, X_dev_asym])
        X_r11_fused = np.hstack([X_r11_fbcsp, X_r11_ts, X_r11_asym])

        # Feature selection (k=32)
        sel = SelectKBest(f_classif, k=min(32, X_dev_fused.shape[1])).fit(X_dev_fused, y_dev_s)
        scaler = StandardScaler().fit(sel.transform(X_dev_fused))

        X_dev_sc = scaler.transform(sel.transform(X_dev_fused))
        X_r11_sc = scaler.transform(sel.transform(X_r11_fused))

        # Tri-ensemble
        clf = VotingClassifier(
            estimators=[
                ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
                ('lr', LogisticRegression(C=1.0, max_iter=500, class_weight='balanced', random_state=42)),
                ('svm', SVC(kernel='rbf', C=1.5, gamma='scale', probability=True, class_weight='balanced', random_state=42))
            ],
            voting='soft'
        )
        clf.fit(X_dev_sc, y_dev_s)

        # Save artifacts per subject
        joblib.dump(bank, os.path.join(s_out, "adaptive_bank.pkl"))
        joblib.dump(fbcsp_dict, os.path.join(s_out, "csp.pkl"))
        joblib.dump(cov_est, os.path.join(s_out, "cov_estimator.pkl"))
        joblib.dump(ts, os.path.join(s_out, "tangent_space.pkl"))
        joblib.dump(sel, os.path.join(s_out, "feature_selector.pkl"))
        joblib.dump(scaler, os.path.join(s_out, "scaler.pkl"))
        joblib.dump(clf, os.path.join(s_out, "model.pkl"))

        # Measure inference latency
        for w in range(len(X_r11_sc)):
            t0 = time.perf_counter()
            _ = clf.predict_proba(X_r11_sc[w:w+1])
            latencies.append((time.perf_counter() - t0) * 1000.0)

        probs_raw = clf.predict_proba(X_r11_sc)
        probs = align_probs(probs_raw, clf.classes_)

        s_preds = []
        for p in probs:
            if p[0] >= 0.58:
                s_preds.append("REST")
            else:
                s_preds.append("LEFT_FIST" if p[1] >= p[2] else "RIGHT_FIST")

        all_r11_preds.extend(s_preds)
        all_r11_gt.extend(y_r11_s)

        s_acc = accuracy_score(y_r11_s, s_preds) * 100.0
        s_f1 = f1_score(y_r11_s, s_preds, average='macro') * 100.0
        sub_metrics.append({
            "subject": s_id,
            "accuracy": round(s_acc, 2),
            "macro_f1": round(s_f1, 2)
        })

    metrics = compute_metrics(all_r11_gt, all_r11_preds, latencies)
    metrics["sub_mean_acc"] = round(float(np.mean([m["accuracy"] for m in sub_metrics])), 2)
    metrics["sub_std_acc"] = round(float(np.std([m["accuracy"] for m in sub_metrics])), 2)
    metrics["per_subject"] = sub_metrics

    summary = {
        "model_name": "Phase 5 Champion: Anatomical Laplacian Advanced Motor Ensemble (k=32)",
        "candidate_id": "p5_cand_04_advanced_motor_features",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spatial_montage": VERIFIED_15_CHANNELS,
        "k_features": 32,
        "r11_holdout_metrics": metrics
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "summary_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"--> Saved Phase 5 Advanced Motor Champion: Accuracy = {metrics['accuracy']}%, Macro F1 = {metrics['macro_f1']}%")
    return summary


def train_and_serialize_phase5_hierarchical(sub_data: dict, subjects: list, out_dir: str):
    print(f"\n=======================================================")
    print(f"  Training & Serializing Phase 5 Calibrated Hierarchical")
    print(f"  Destination: {out_dir}")
    print(f"=======================================================")

    sel_model_dir = os.path.join(out_dir, "selected_model")
    os.makedirs(sel_model_dir, exist_ok=True)

    all_r11_preds, all_r11_gt, latencies = [], [], []
    sub_metrics = []

    for s_id in subjects:
        s_out = os.path.join(sel_model_dir, s_id)
        os.makedirs(s_out, exist_ok=True)

        X_dev_s = sub_data[s_id]["X_calib"].copy()
        y_dev_s = sub_data[s_id]["y_calib"].copy()
        X_r11_s = sub_data[s_id]["X_r11"].copy()
        y_r11_s = sub_data[s_id]["y_r11"].copy()

        X_dev_lap = apply_anatomical_surface_laplacian(X_dev_s)
        X_r11_lap = apply_anatomical_surface_laplacian(X_r11_s)

        bank = build_subject_adaptive_filter_bank(10.0, 20.0)
        filt_dev = extract_custom_filter_bank_windows(X_dev_lap, bank)
        filt_r11 = extract_custom_filter_bank_windows(X_r11_lap, bank)

        X_dev_fbcsp, fbcsp_dict = safe_fit_fbcsp(filt_dev, y_dev_s, n_components=4)
        X_r11_fbcsp = extract_fbcsp_features_transform(filt_r11, fbcsp_dict)

        cov_est = Covariances(estimator='lwf')
        cov_dev = cov_est.fit_transform(X_dev_lap)
        ts = TangentSpace(metric='riemann')
        X_dev_ts = ts.fit_transform(cov_dev)
        X_r11_ts = ts.transform(cov_est.transform(X_r11_lap))

        X_dev_asym = extract_advanced_motor_features(X_dev_lap)
        X_r11_asym = extract_advanced_motor_features(X_r11_lap)

        X_dev_fused = np.hstack([X_dev_fbcsp, X_dev_ts, X_dev_asym])
        X_r11_fused = np.hstack([X_r11_fbcsp, X_r11_ts, X_r11_asym])

        sel = SelectKBest(f_classif, k=min(32, X_dev_fused.shape[1])).fit(X_dev_fused, y_dev_s)
        scaler = StandardScaler().fit(sel.transform(X_dev_fused))

        X_dev_sc = scaler.transform(sel.transform(X_dev_fused))
        X_r11_sc = scaler.transform(sel.transform(X_r11_fused))

        hier = CalibratedHierarchicalBCI(target_rest_recall=0.90)
        hier.fit(X_dev_sc, y_dev_s, oof_tune=True)

        joblib.dump(bank, os.path.join(s_out, "adaptive_bank.pkl"))
        joblib.dump(fbcsp_dict, os.path.join(s_out, "csp.pkl"))
        joblib.dump(cov_est, os.path.join(s_out, "cov_estimator.pkl"))
        joblib.dump(ts, os.path.join(s_out, "tangent_space.pkl"))
        joblib.dump(sel, os.path.join(s_out, "feature_selector.pkl"))
        joblib.dump(scaler, os.path.join(s_out, "scaler.pkl"))
        joblib.dump(hier, os.path.join(s_out, "model.pkl"))

        for w in range(len(X_r11_sc)):
            t0 = time.perf_counter()
            _ = hier.predict_proba(X_r11_sc[w:w+1])
            latencies.append((time.perf_counter() - t0) * 1000.0)

        s_preds = hier.predict(X_r11_sc).tolist()
        all_r11_preds.extend(s_preds)
        all_r11_gt.extend(y_r11_s)

        s_acc = accuracy_score(y_r11_s, s_preds) * 100.0
        s_f1 = f1_score(y_r11_s, s_preds, average='macro') * 100.0
        sub_metrics.append({
            "subject": s_id,
            "accuracy": round(s_acc, 2),
            "macro_f1": round(s_f1, 2)
        })

    metrics = compute_metrics(all_r11_gt, all_r11_preds, latencies)
    metrics["sub_mean_acc"] = round(float(np.mean([m["accuracy"] for m in sub_metrics])), 2)
    metrics["sub_std_acc"] = round(float(np.std([m["accuracy"] for m in sub_metrics])), 2)
    metrics["per_subject"] = sub_metrics

    summary = {
        "model_name": "Phase 5 Research Model: Calibrated Hierarchical Intent-Lateralization Decoder",
        "candidate_id": "p5_cand_05_calibrated_hierarchical",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spatial_montage": VERIFIED_15_CHANNELS,
        "k_features": 32,
        "r11_holdout_metrics": metrics
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "summary_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"--> Saved Phase 5 Calibrated Hierarchical Model: Accuracy = {metrics['accuracy']}%, Macro F1 = {metrics['macro_f1']}%")
    return summary


def main():
    sub_data = load_dataset_step61(data_dir="data")
    subjects = sorted(list(sub_data.keys()))

    dir_champion = "models/research_phase5/p5_advanced_motor_champion"
    dir_hierarchical = "models/research_phase5/p5_calibrated_hierarchical"

    m_champ = train_and_serialize_phase5_advanced_motor(sub_data, subjects, dir_champion)
    m_hier = train_and_serialize_phase5_hierarchical(sub_data, subjects, dir_hierarchical)

    # Save champion metadata pointer
    meta_dir = "models/research_phase5/champion_candidate"
    os.makedirs(meta_dir, exist_ok=True)
    meta = {
        "candidate_id": "p5_cand_04_advanced_motor_features",
        "name": "Phase 5 Champion: Anatomical Laplacian Advanced Motor Ensemble (k=32)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spatial_montage": VERIFIED_15_CHANNELS,
        "dev_cv_accuracy": 72.17,
        "dev_cv_macro_f1": 67.52,
        "r11_accuracy": m_champ["r11_holdout_metrics"]["accuracy"],
        "r11_macro_f1": m_champ["r11_holdout_metrics"]["macro_f1"],
        "r11_rest_recall": m_champ["r11_holdout_metrics"]["rest_recall"],
        "r11_left_recall": m_champ["r11_holdout_metrics"]["left_recall"],
        "r11_right_recall": m_champ["r11_holdout_metrics"]["right_recall"],
        "r11_false_actuation": m_champ["r11_holdout_metrics"]["rest_false_actuation_rate"]
    }
    with open(os.path.join(meta_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nPhase 5 model serialization complete.")


if __name__ == "__main__":
    main()
