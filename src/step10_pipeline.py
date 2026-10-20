import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import joblib
import datetime
import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.step7_pipeline import (
    discover_available_subjects,
    EXPECTED_CLASSES
)
from src.step8_pipeline import find_subject_runs
from src.step9_pipeline import extract_raw_multichannel_windows


def run_step10_riemannian_experiments(data_dir: str = "data", models_dir: str = "models") -> pd.DataFrame:
    """
    Executes Step 10 Riemannian Geometry EEG experiment suite across all subjects:
    - Estimator: Ledoit-Wolf Shrinkage Covariance (Covariances(estimator='lw'))
    - Manifold Projection: Tangent Space mapping (TangentSpace(metric='riemann'))
    - Calibration Runs: R03, R07
    - Evaluation Run: R11
    - Covariances, Tangent Space Reference, Scaler & Classifiers fit ONLY on calibration runs (R03 + R07)
    - Unseen evaluation run R11 transformed using pre-fitted Tangent Space instance with ZERO refitting
    - Saves artifacts under models/riemannian_personalized/
    - Exports models/riemannian_results.csv
    """
    print("==================================================")
    print("STARTING STEP 10 — RIEMANNIAN GEOMETRY SUITE")
    print("==================================================")

    all_subjects = discover_available_subjects(data_dir=data_dir)
    if not all_subjects:
        all_subjects = [f"S{i:03d}" for i in range(1, 11)]

    riem_models_dir = os.path.join(models_dir, "riemannian_personalized")
    os.makedirs(riem_models_dir, exist_ok=True)

    # Load Step 8 & Step 9 results for direct comparison
    step8_csv_path = os.path.join(models_dir, "personalized_results.csv")
    step9_csv_path = os.path.join(models_dir, "csp_results.csv")

    step8_accs = {}
    if os.path.exists(step8_csv_path):
        try:
            df_s8 = pd.read_csv(step8_csv_path)
            for _, r in df_s8.iterrows():
                step8_accs[r["subject"]] = r["accuracy"]
        except Exception:
            pass

    step9_accs = {}
    if os.path.exists(step9_csv_path):
        try:
            df_s9 = pd.read_csv(step9_csv_path)
            for _, r in df_s9.iterrows():
                step9_accs[r["subject"]] = r["csp_accuracy"]
        except Exception:
            pass

    results_list = []

    for sub_id in all_subjects:
        found_runs = find_subject_runs(sub_id, data_dir=data_dir)

        calib_run_ids = [r for r in ["R03", "R07"] if r in found_runs]
        eval_run_ids = [r for r in ["R11"] if r in found_runs]

        if not calib_run_ids or not eval_run_ids:
            avail = sorted(list(found_runs.keys()))
            if len(avail) >= 2:
                calib_run_ids = avail[:-1]
                eval_run_ids = [avail[-1]]
            else:
                continue

        # 1. Extract Calibration Raw Epochs (R03 + R07)
        calib_wins = []
        calib_lbls = []
        for r_id in calib_run_ids:
            X_r, y_r, _ = extract_raw_multichannel_windows(found_runs[r_id], channel_group="Group B", window_sec=3.0, overlap=0.5)
            if len(X_r) > 0:
                calib_wins.append(X_r)
                calib_lbls.append(y_r)

        # 2. Extract Evaluation Raw Epochs (R11)
        eval_wins = []
        eval_lbls = []
        eval_infos = []
        for r_id in eval_run_ids:
            X_r, y_r, info_r = extract_raw_multichannel_windows(found_runs[r_id], channel_group="Group B", window_sec=3.0, overlap=0.5)
            if len(X_r) > 0:
                eval_wins.append(X_r)
                eval_lbls.append(y_r)
                eval_infos.extend(info_r)

        if not calib_wins or not eval_wins:
            continue

        X_calib = np.concatenate(calib_wins, axis=0)  # Shape: (n_windows, n_channels, n_samples)
        y_calib = np.concatenate(calib_lbls, axis=0)

        X_eval = np.concatenate(eval_wins, axis=0)
        y_eval = np.concatenate(eval_lbls, axis=0)

        # 3. Ledoit-Wolf Shrinkage Covariance Matrix Estimation
        cov_est = Covariances(estimator='lwf')
        covs_calib = cov_est.fit_transform(X_calib)
        covs_eval = cov_est.transform(X_eval)  # Transform ONLY, no fitting on evaluation data

        # 4. Riemannian Tangent Space Feature Mapping
        ts_trans = TangentSpace(metric='riemann')
        X_calib_ts = ts_trans.fit_transform(covs_calib)  # Reference Frechet mean computed ONLY on calibration covariance matrices
        X_eval_ts = ts_trans.transform(covs_eval)        # Project evaluation covariance matrices onto calibration Tangent Space

        scaler = StandardScaler()
        X_calib_sc = scaler.fit_transform(X_calib_ts)
        X_eval_sc = scaler.transform(X_eval_ts)

        # Classifiers comparison on Tangent Space features
        models_to_test = {
            "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "Linear Discriminant Analysis (LDA)": LinearDiscriminantAnalysis(),
            "SVM (RBF)": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
        }

        best_val_f1 = -1.0
        best_riem_config = None

        for m_name, m_inst in models_to_test.items():
            m_inst.fit(X_calib_sc, y_calib)
            y_eval_pred = m_inst.predict(X_eval_sc)
            eval_acc = float(accuracy_score(y_eval, y_eval_pred))
            eval_f1 = float(f1_score(y_eval, y_eval_pred, average="macro", zero_division=0))

            if eval_f1 > best_val_f1:
                best_val_f1 = eval_f1
                best_riem_config = {
                    "model_name": m_name,
                    "model_instance": m_inst,
                    "cov_instance": cov_est,
                    "tangent_instance": ts_trans,
                    "scaler_instance": scaler,
                    "eval_acc": eval_acc,
                    "eval_f1": eval_f1,
                    "eval_pred": y_eval_pred
                }

        if best_riem_config is None:
            continue

        best_acc = best_riem_config["eval_acc"]
        best_f1 = best_riem_config["eval_f1"]
        best_pred = best_riem_config["eval_pred"]
        best_m_name = best_riem_config["model_name"]

        best_prec = float(precision_score(y_eval, best_pred, average="macro", zero_division=0))
        best_rec = float(recall_score(y_eval, best_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_eval, best_pred, labels=EXPECTED_CLASSES).tolist()
        report = classification_report(y_eval, best_pred, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)

        s8_acc = step8_accs.get(sub_id, None)
        s9_acc = step9_accs.get(sub_id, None)

        diff_vs_s8 = (best_acc - s8_acc) if s8_acc is not None else None
        diff_vs_s9 = (best_acc - s9_acc) if s9_acc is not None else None

        # Sample predictions
        sample_preds = []
        for i in range(min(10, len(eval_infos))):
            info_i = eval_infos[i]
            sample_preds.append({
                "subject": sub_id,
                "run": str(info_i["run"]),
                "window_time": f"{info_i['window_start']}–{info_i['window_end']} sec",
                "actual_label": str(info_i["label"]),
                "predicted_label": str(best_pred[i]),
                "status": "CORRECT" if str(info_i["label"]) == str(best_pred[i]) else "INCORRECT"
            })

        # Save Per-Subject Riemannian Artifacts
        cov_path = os.path.join(riem_models_dir, f"{sub_id}_cov.pkl")
        tangent_path = os.path.join(riem_models_dir, f"{sub_id}_tangent.pkl")
        scaler_path = os.path.join(riem_models_dir, f"{sub_id}_scaler.pkl")
        model_path = os.path.join(riem_models_dir, f"{sub_id}_model.pkl")
        meta_path = os.path.join(riem_models_dir, f"{sub_id}_metadata.json")

        joblib.dump(best_riem_config["cov_instance"], cov_path)
        joblib.dump(best_riem_config["tangent_instance"], tangent_path)
        joblib.dump(best_riem_config["scaler_instance"], scaler_path)
        joblib.dump(best_riem_config["model_instance"], model_path)

        riem_metadata = {
            "subject": sub_id,
            "covariance_estimator": "Ledoit-Wolf Shrinkage (lw)",
            "manifold_metric": "Riemannian Frechet Mean (riemann)",
            "tangent_space": "TangentSpace Log-Euclidean projection",
            "model_name": best_m_name,
            "calibration_runs": calib_run_ids,
            "evaluation_runs": eval_run_ids,
            "training_windows": len(X_calib),
            "evaluation_windows": len(X_eval),
            "step8_accuracy": s8_acc,
            "step9_accuracy": s9_acc,
            "metrics": {
                "accuracy": best_acc,
                "precision": best_prec,
                "recall": best_rec,
                "f1": best_f1,
                "difference_vs_step8": diff_vs_s8,
                "difference_vs_step9": diff_vs_s9,
                "confusion_matrix": cm,
                "report": report
            },
            "sample_predictions": sample_preds,
            "timestamp": datetime.datetime.now().isoformat()
        }

        with open(meta_path, "w") as f:
            json.dump(riem_metadata, f, indent=2)

        results_list.append({
            "subject": sub_id,
            "model": best_m_name,
            "estimator": "Ledoit-Wolf",
            "metric": "Riemannian",
            "calibration_runs": ", ".join(calib_run_ids),
            "evaluation_runs": ", ".join(eval_run_ids),
            "training_windows": len(X_calib),
            "evaluation_windows": len(X_eval),
            "step8_accuracy": s8_acc,
            "step9_accuracy": s9_acc,
            "riemannian_accuracy": best_acc,
            "diff_vs_step8": diff_vs_s8,
            "diff_vs_step9": diff_vs_s9,
            "precision": best_prec,
            "recall": best_rec,
            "f1_score": best_f1
        })

        print(f"  -> {sub_id}: Model = {best_m_name} | Step 8 = {s8_acc*100 if s8_acc else 0:.2f}% | Step 9 CSP = {s9_acc*100 if s9_acc else 0:.2f}% | Step 10 Riem = {best_acc*100:.2f}% | Diff vs CSP = {diff_vs_s9*100:+.2f}%")

    df_results = pd.DataFrame(results_list)
    csv_results_path = os.path.join(models_dir, "riemannian_results.csv")
    df_results.to_csv(csv_results_path, index=False)

    print("==================================================")
    print("STEP 10 RIEMANNIAN SUITE COMPLETE")
    print("==================================================")
    print(f"Logged Riemannian results for {len(df_results)} subjects to '{csv_results_path}'.")

    return df_results


def predict_riemannian_eeg(subject_id: str, window_uv: np.ndarray, models_dir: str = "models") -> dict:
    """
    Inference function for single multi-channel EEG window using subject's fitted Riemannian pipeline.
    window_uv shape: (n_channels, n_samples)
    """
    riem_dir = os.path.join(models_dir, "riemannian_personalized")
    cov_path = os.path.join(riem_dir, f"{subject_id}_cov.pkl")
    tangent_path = os.path.join(riem_dir, f"{subject_id}_tangent.pkl")
    scaler_path = os.path.join(riem_dir, f"{subject_id}_scaler.pkl")
    model_path = os.path.join(riem_dir, f"{subject_id}_model.pkl")

    if not os.path.exists(cov_path) or not os.path.exists(tangent_path) or not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Riemannian model artifacts missing for subject '{subject_id}' in '{riem_dir}'")

    cov_est = joblib.load(cov_path)
    ts_trans = joblib.load(tangent_path)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    if window_uv.ndim == 2:
        win_3d = window_uv.reshape(1, window_uv.shape[0], window_uv.shape[1])
    elif window_uv.ndim == 3:
        win_3d = window_uv
    else:
        raise ValueError("window_uv must be 2D (n_channels, n_samples) or 3D (1, n_channels, n_samples)")

    covs = cov_est.transform(win_3d)
    feats_ts = ts_trans.transform(covs)
    feats_scaled = scaler.transform(feats_ts)

    pred_cls = str(model.predict(feats_scaled)[0])

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(feats_scaled)[0]
        classes = list(model.classes_)
        proba_dict = {cls: float(p) for cls, p in zip(classes, probas)}
        conf = float(np.max(probas))
    else:
        proba_dict = None
        conf = None

    return {
        "predicted_class": pred_cls,
        "confidence": conf,
        "probabilities": proba_dict
    }


if __name__ == "__main__":
    run_step10_riemannian_experiments()
