import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import joblib
import datetime
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.integrate as integrate
from collections import Counter
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from mne.decoding import CSP

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_rel, wilcoxon
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
    load_and_preprocess_cached,
    EXPECTED_CLASSES,
    LABEL_MAPPING
)
from src.step8_pipeline import find_subject_runs
from src.step9_pipeline import extract_raw_multichannel_windows
from src.step11_pipeline import apply_causal_temporal_smoothing


def compute_causal_smoothing_series(predictions: list, window_size: int = 3) -> list:
    smoothed = []
    history = []
    for p in predictions:
        history.append(p)
        smoothed.append(apply_causal_temporal_smoothing(history, window_size=window_size))
    return smoothed


def compute_spectral_features_single_window(X_win: np.ndarray, sfreq: float = 160.0) -> np.ndarray:
    """
    Computes Mu (8-13 Hz) and Beta (13-30 Hz) spectral features for a single 2D EEG window (n_channels, n_samples).
    Features per channel:
    - absolute Mu power
    - absolute Beta power
    - relative Mu power (Mu / Total 1-45 Hz power)
    - relative Beta power (Beta / Total 1-45 Hz power)
    - Mu / Beta power ratio
    - Alpha / Beta power ratio (Alpha equivalent to Mu)
    Also includes channel-aggregated mean spectral summary metrics.
    """
    n_channels, n_samples = X_win.shape
    ch_features = []

    mu_powers, beta_powers, rel_mus, rel_betas = [], [], [], []

    for c in range(n_channels):
        sig_c = X_win[c]
        nperseg = min(n_samples, int(sfreq * 2))
        freqs, psd = signal.welch(sig_c, fs=sfreq, nperseg=nperseg)

        # Total power (1 - 45 Hz)
        idx_tot = (freqs >= 1.0) & (freqs <= 45.0)
        tot_p = max(1e-8, float(integrate.simpson(psd[idx_tot], freqs[idx_tot])) if np.any(idx_tot) else 1e-8)

        # Mu power (8 - 13 Hz)
        idx_mu = (freqs >= 8.0) & (freqs <= 13.0)
        mu_p = max(0.0, float(integrate.simpson(psd[idx_mu], freqs[idx_mu])) if np.any(idx_mu) else 0.0)

        # Beta power (13 - 30 Hz)
        idx_beta = (freqs >= 13.0) & (freqs <= 30.0)
        beta_p = max(0.0, float(integrate.simpson(psd[idx_beta], freqs[idx_beta])) if np.any(idx_beta) else 0.0)

        rel_mu = mu_p / tot_p
        rel_beta = beta_p / tot_p
        mu_beta_ratio = mu_p / (beta_p + 1e-8)

        mu_powers.append(mu_p)
        beta_powers.append(beta_p)
        rel_mus.append(rel_mu)
        rel_betas.append(rel_beta)

        ch_features.extend([mu_p, beta_p, rel_mu, rel_beta, mu_beta_ratio])

    # Global channel summary features
    mean_mu = float(np.mean(mu_powers))
    mean_beta = float(np.mean(beta_powers))
    mean_rel_mu = float(np.mean(rel_mus))
    mean_rel_beta = float(np.mean(rel_betas))
    global_mu_beta_ratio = mean_mu / (mean_beta + 1e-8)

    summary_feats = [mean_mu, mean_beta, mean_rel_mu, mean_rel_beta, global_mu_beta_ratio]
    return np.array(ch_features + summary_feats)


def extract_spectral_features_batch(X_windows: np.ndarray, sfreq: float = 160.0) -> np.ndarray:
    """
    Extracts spectral features across a batch of 3D EEG windows (n_windows, n_channels, n_samples).
    """
    feats = []
    for i in range(len(X_windows)):
        feats.append(compute_spectral_features_single_window(X_windows[i], sfreq=sfreq))
    return np.array(feats)


def extract_event_aligned_windows(
    edf_path: str,
    channel_group: str = "Group B",
    window_sec: float = 3.0,
    onset_offset_sec: float = 0.5
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Extracts event-aligned multi-channel EEG windows starting at (onset_sec + onset_offset_sec).
    """
    try:
        raw_sub, df_events = load_and_preprocess_cached(edf_path, channel_group)
    except Exception:
        return np.array([]), np.array([]), []

    if df_events.empty:
        return np.array([]), np.array([]), []

    sfreq = raw_sub.info['sfreq']
    win_samples = int(window_sec * sfreq)

    windows = []
    labels = []
    info_list = []

    filename = os.path.basename(edf_path)
    subject_id = filename[:4] if filename.startswith("S") else "S001"
    run_id = filename[4:7] if len(filename) >= 7 else "R03"

    for _, ev in df_events.iterrows():
        code = ev['event_code']
        if code not in LABEL_MAPPING:
            continue

        label = LABEL_MAPPING[code]
        aligned_onset = ev['onset_sec'] + onset_offset_sec
        start_sample = int(aligned_onset * sfreq)
        end_sample = start_sample + win_samples

        max_sample = int(ev['end_sec'] * sfreq)
        if end_sample > max_sample or end_sample > len(raw_sub.times):
            # Fallback to onset without offset if offset exceeds boundary
            start_sample = int(ev['onset_sec'] * sfreq)
            end_sample = start_sample + win_samples
            if end_sample > len(raw_sub.times):
                continue

        sig_data = raw_sub.get_data(start=start_sample, stop=end_sample)

        if np.isnan(sig_data).any() or np.isinf(sig_data).any():
            continue

        windows.append(sig_data)
        labels.append(label)
        info_list.append({
            "subject": subject_id,
            "run": run_id,
            "window_start": round(start_sample / sfreq, 4),
            "window_end": round(end_sample / sfreq, 4),
            "label": label
        })

    return np.array(windows), np.array(labels), info_list


def run_step14_motor_rhythm_experiments(data_dir: str = "data", models_dir: str = "models") -> dict:
    print("==================================================")
    print("STARTING STEP 14 — EVENT-ALIGNED + MOTOR-RHYTHM EEG OPTIMIZATION")
    print("==================================================")

    out_dir = os.path.join(models_dir, "step14")
    os.makedirs(out_dir, exist_ok=True)

    all_subjects = discover_available_subjects(data_dir=data_dir)
    if not all_subjects:
        all_subjects = [f"S{i:03d}" for i in range(1, 11)]

    # Experiment containers
    window_exp_records = []
    feature_exp_records = []
    channel_exp_records = []
    grid_cv_results = []
    subject_r11_results = []

    # Storage for untouched R11 evaluation
    all_s13b_r11_actuals, all_s13b_r11_preds = [], []
    all_s14_r11_actuals, all_s14_r11_raw_preds = [], []
    all_s14_r11_smooth3_preds = []

    # Track best pipeline config on R03 + R07 validation
    candidate_configs = [
        "Config A: Standard Sliding (3s, 50% ov) + Riemannian",
        "Config B: Event-Aligned (0.5s offset) + Riemannian",
        "Config C: Standard Sliding + Riemannian + Motor Spectral (Mu/Beta)",
        "Config D: Event-Aligned (0.5s offset) + Riemannian + Motor Spectral (Mu/Beta)",
        "Config E: Standard Sliding + CSP + Riemannian",
        "Config F: 3-Channel Group A + Riemannian + Motor Spectral"
    ]

    config_cv_scores = {cfg: [] for cfg in candidate_configs}

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

        # Load Calibration Data (R03 + R07) - Standard Group B Sliding
        calib_wins_std, calib_lbls_std = [], []
        calib_wins_aligned, calib_lbls_aligned = [], []
        calib_wins_grpA, calib_lbls_grpA = [], []

        for r_id in calib_run_ids:
            X_std, y_std, _ = extract_raw_multichannel_windows(found_runs[r_id], channel_group="Group B", window_sec=3.0, overlap=0.5)
            X_align, y_align, _ = extract_event_aligned_windows(found_runs[r_id], channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
            X_grpA, y_grpA, _ = extract_raw_multichannel_windows(found_runs[r_id], channel_group="Group A", window_sec=3.0, overlap=0.5)

            if len(X_std) > 0:
                calib_wins_std.append(X_std)
                calib_lbls_std.append(y_std)
            if len(X_align) > 0:
                calib_wins_aligned.append(X_align)
                calib_lbls_aligned.append(y_align)
            if len(X_grpA) > 0:
                calib_wins_grpA.append(X_grpA)
                calib_lbls_grpA.append(y_grpA)

        # Load Evaluation Data (R11)
        eval_wins_std, eval_lbls_std = [], []
        eval_wins_aligned, eval_lbls_aligned = [], []

        for r_id in eval_run_ids:
            X_std, y_std, _ = extract_raw_multichannel_windows(found_runs[r_id], channel_group="Group B", window_sec=3.0, overlap=0.5)
            X_align, y_align, _ = extract_event_aligned_windows(found_runs[r_id], channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
            if len(X_std) > 0:
                eval_wins_std.append(X_std)
                eval_lbls_std.append(y_std)
            if len(X_align) > 0:
                eval_wins_aligned.append(X_align)
                eval_lbls_aligned.append(y_align)

        if not calib_wins_std or not eval_wins_std:
            continue

        X_calib_std = np.concatenate(calib_wins_std, axis=0)
        y_calib_std = np.concatenate(calib_lbls_std, axis=0)

        X_calib_aligned = np.concatenate(calib_wins_aligned, axis=0) if calib_wins_aligned else X_calib_std
        y_calib_aligned = np.concatenate(calib_lbls_aligned, axis=0) if calib_wins_aligned else y_calib_std

        X_calib_grpA = np.concatenate(calib_wins_grpA, axis=0) if calib_wins_grpA else X_calib_std
        y_calib_grpA = np.concatenate(calib_lbls_grpA, axis=0) if calib_wins_grpA else y_calib_std

        X_eval_std = np.concatenate(eval_wins_std, axis=0)
        y_eval_std = np.concatenate(eval_lbls_std, axis=0)

        X_eval_aligned = np.concatenate(eval_wins_aligned, axis=0) if eval_wins_aligned else X_eval_std
        y_eval_aligned = np.concatenate(eval_lbls_aligned, axis=0) if eval_wins_aligned else y_eval_std

        # -------------------------------------------------------------
        # SYSTEMATIC EXPERIMENTS A, B, C, D, E, F ON R03 + R07 (5-FOLD CV)
        # -------------------------------------------------------------
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Config A: Standard Sliding + Riemannian (Step 13B Baseline)
        cov_A = Covariances(estimator='lwf')
        ts_A = TangentSpace(metric='riemann')
        X_ts_A = ts_A.fit_transform(cov_A.fit_transform(X_calib_std))
        
        cv_preds_A = np.empty(len(y_calib_std), dtype=object)
        for train_idx, val_idx in skf.split(X_ts_A, y_calib_std):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_ts_A[train_idx])
            X_va = sc.transform(X_ts_A[val_idx])
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
            clf.fit(X_tr, y_calib_std[train_idx])
            cv_preds_A[val_idx] = clf.predict(X_va)
        f1_cfg_A = f1_score(y_calib_std, cv_preds_A, average="macro", zero_division=0)
        config_cv_scores["Config A: Standard Sliding (3s, 50% ov) + Riemannian"].append(f1_cfg_A)

        # Config B: Event-Aligned (0.5s offset) + Riemannian
        cov_B = Covariances(estimator='lwf')
        ts_B = TangentSpace(metric='riemann')
        X_ts_B = ts_B.fit_transform(cov_B.fit_transform(X_calib_aligned))
        skf_align = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_preds_B = np.empty(len(y_calib_aligned), dtype=object)
        for train_idx, val_idx in skf_align.split(X_ts_B, y_calib_aligned):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_ts_B[train_idx])
            X_va = sc.transform(X_ts_B[val_idx])
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
            clf.fit(X_tr, y_calib_aligned[train_idx])
            cv_preds_B[val_idx] = clf.predict(X_va)
        f1_cfg_B = f1_score(y_calib_aligned, cv_preds_B, average="macro", zero_division=0)
        config_cv_scores["Config B: Event-Aligned (0.5s offset) + Riemannian"].append(f1_cfg_B)

        # Config C: Standard Sliding + Riemannian + Motor Spectral (Mu/Beta)
        X_spec_std = extract_spectral_features_batch(X_calib_std)
        X_comb_C = np.hstack([X_ts_A, X_spec_std])
        cv_preds_C = np.empty(len(y_calib_std), dtype=object)
        for train_idx, val_idx in skf.split(X_comb_C, y_calib_std):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_comb_C[train_idx])
            X_va = sc.transform(X_comb_C[val_idx])
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
            clf.fit(X_tr, y_calib_std[train_idx])
            cv_preds_C[val_idx] = clf.predict(X_va)
        f1_cfg_C = f1_score(y_calib_std, cv_preds_C, average="macro", zero_division=0)
        config_cv_scores["Config C: Standard Sliding + Riemannian + Motor Spectral (Mu/Beta)"].append(f1_cfg_C)

        # Config D: Event-Aligned (0.5s offset) + Riemannian + Motor Spectral (Mu/Beta)
        X_spec_align = extract_spectral_features_batch(X_calib_aligned)
        X_comb_D = np.hstack([X_ts_B, X_spec_align])
        cv_preds_D = np.empty(len(y_calib_aligned), dtype=object)
        for train_idx, val_idx in skf_align.split(X_comb_D, y_calib_aligned):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_comb_D[train_idx])
            X_va = sc.transform(X_comb_D[val_idx])
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
            clf.fit(X_tr, y_calib_aligned[train_idx])
            cv_preds_D[val_idx] = clf.predict(X_va)
        f1_cfg_D = f1_score(y_calib_aligned, cv_preds_D, average="macro", zero_division=0)
        config_cv_scores["Config D: Event-Aligned (0.5s offset) + Riemannian + Motor Spectral (Mu/Beta)"].append(f1_cfg_D)

        # Config E: Standard Sliding + CSP + Riemannian
        csp_E = CSP(n_components=4, log=True, norm_trace=False)
        X_csp_E = csp_E.fit_transform(X_calib_std, y_calib_std)
        X_comb_E = np.hstack([X_ts_A, X_csp_E])
        cv_preds_E = np.empty(len(y_calib_std), dtype=object)
        for train_idx, val_idx in skf.split(X_comb_E, y_calib_std):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_comb_E[train_idx])
            X_va = sc.transform(X_comb_E[val_idx])
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
            clf.fit(X_tr, y_calib_std[train_idx])
            cv_preds_E[val_idx] = clf.predict(X_va)
        f1_cfg_E = f1_score(y_calib_std, cv_preds_E, average="macro", zero_division=0)
        config_cv_scores["Config E: Standard Sliding + CSP + Riemannian"].append(f1_cfg_E)

        # Config F: 3-Channel Group A + Riemannian + Motor Spectral
        cov_F = Covariances(estimator='lwf')
        ts_F = TangentSpace(metric='riemann')
        X_ts_F = ts_F.fit_transform(cov_F.fit_transform(X_calib_grpA))
        X_spec_F = extract_spectral_features_batch(X_calib_grpA)
        X_comb_F = np.hstack([X_ts_F, X_spec_F])
        skf_grpA = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_preds_F = np.empty(len(y_calib_grpA), dtype=object)
        for train_idx, val_idx in skf_grpA.split(X_comb_F, y_calib_grpA):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_comb_F[train_idx])
            X_va = sc.transform(X_comb_F[val_idx])
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
            clf.fit(X_tr, y_calib_grpA[train_idx])
            cv_preds_F[val_idx] = clf.predict(X_va)
        f1_cfg_F = f1_score(y_calib_grpA, cv_preds_F, average="macro", zero_division=0)
        config_cv_scores["Config F: 3-Channel Group A + Riemannian + Motor Spectral"].append(f1_cfg_F)

        grid_cv_results.append({
            "subject": sub_id,
            "f1_cfg_A": round(f1_cfg_A * 100, 2),
            "f1_cfg_B": round(f1_cfg_B * 100, 2),
            "f1_cfg_C": round(f1_cfg_C * 100, 2),
            "f1_cfg_D": round(f1_cfg_D * 100, 2),
            "f1_cfg_E": round(f1_cfg_E * 100, 2),
            "f1_cfg_F": round(f1_cfg_F * 100, 2),
        })

    # -------------------------------------------------------------
    # FREEZE WINNING PIPELINE CONFIGURATION BASED ON CV MACRO F1
    # -------------------------------------------------------------
    mean_cfg_scores = {cfg: np.mean(scores) for cfg, scores in config_cv_scores.items()}
    best_config_name = max(mean_cfg_scores, key=mean_cfg_scores.get)
    print(f"Selected Winning Pipeline (R03+R07 CV): {best_config_name} (Mean CV F1: {mean_cfg_scores[best_config_name]*100:.2f}%)")

    use_event_alignment = ("Event-Aligned" in best_config_name)
    use_motor_spectral = ("Motor Spectral" in best_config_name)
    use_csp = ("CSP" in best_config_name)
    use_groupA = ("3-Channel Group A" in best_config_name)

    # -------------------------------------------------------------
    # TRAIN WINNING MODEL PER SUBJECT & EVALUATE ON HOLDOUT R11
    # -------------------------------------------------------------
    for sub_id in all_subjects:
        found_runs = find_subject_runs(sub_id, data_dir=data_dir)
        calib_run_ids = [r for r in ["R03", "R07"] if r in found_runs]
        eval_run_ids = [r for r in ["R11"] if r in found_runs]

        if not calib_run_ids or not eval_run_ids:
            continue

        target_ch_group = "Group A" if use_groupA else "Group B"

        calib_wins, calib_lbls = [], []
        for r_id in calib_run_ids:
            if use_event_alignment:
                X_r, y_r, _ = extract_event_aligned_windows(found_runs[r_id], channel_group=target_ch_group, window_sec=3.0, onset_offset_sec=0.5)
            else:
                X_r, y_r, _ = extract_raw_multichannel_windows(found_runs[r_id], channel_group=target_ch_group, window_sec=3.0, overlap=0.5)

            if len(X_r) > 0:
                calib_wins.append(X_r)
                calib_lbls.append(y_r)

        eval_wins, eval_lbls = [], []
        for r_id in eval_run_ids:
            if use_event_alignment:
                X_r, y_r, _ = extract_event_aligned_windows(found_runs[r_id], channel_group=target_ch_group, window_sec=3.0, onset_offset_sec=0.5)
            else:
                X_r, y_r, _ = extract_raw_multichannel_windows(found_runs[r_id], channel_group=target_ch_group, window_sec=3.0, overlap=0.5)

            if len(X_r) > 0:
                eval_wins.append(X_r)
                eval_lbls.append(y_r)

        if not calib_wins or not eval_wins:
            continue

        X_calib = np.concatenate(calib_wins, axis=0)
        y_calib = np.concatenate(calib_lbls, axis=0)

        X_eval = np.concatenate(eval_wins, axis=0)
        y_eval = np.concatenate(eval_lbls, axis=0)

        # Baseline Step 13B Model Fit for Comparison
        cov_est_base = Covariances(estimator='lwf')
        ts_trans_base = TangentSpace(metric='riemann')
        X_calib_ts_base = ts_trans_base.fit_transform(cov_est_base.fit_transform(X_calib))
        scaler_base = StandardScaler()
        X_calib_sc_base = scaler_base.fit_transform(X_calib_ts_base)
        m_s13b = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
        m_s13b.fit(X_calib_sc_base, y_calib)

        # Selected Step 14 Pipeline Transformers & Model Fit
        cov_est = Covariances(estimator='lwf')
        ts_trans = TangentSpace(metric='riemann')
        covs_calib = cov_est.fit_transform(X_calib)
        X_calib_ts = ts_trans.fit_transform(covs_calib)

        feat_list_tr = [X_calib_ts]

        csp_transformer = None
        if use_csp:
            csp_transformer = CSP(n_components=4, log=True, norm_trace=False)
            X_calib_csp = csp_transformer.fit_transform(X_calib, y_calib)
            feat_list_tr.append(X_calib_csp)

        if use_motor_spectral:
            X_calib_spec = extract_spectral_features_batch(X_calib)
            feat_list_tr.append(X_calib_spec)

        X_calib_combined = np.hstack(feat_list_tr) if len(feat_list_tr) > 1 else feat_list_tr[0]
        scaler = StandardScaler()
        X_calib_scaled = scaler.fit_transform(X_calib_combined)

        model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
        model.fit(X_calib_scaled, y_calib)

        # Save Selected Step 14 Model Artifacts
        sub_out_dir = os.path.join(out_dir, "selected_model", sub_id)
        os.makedirs(sub_out_dir, exist_ok=True)

        joblib.dump(cov_est, os.path.join(sub_out_dir, "cov.pkl"))
        joblib.dump(ts_trans, os.path.join(sub_out_dir, "riemannian.pkl"))
        if csp_transformer is not None:
            joblib.dump(csp_transformer, os.path.join(sub_out_dir, "csp.pkl"))
        joblib.dump(scaler, os.path.join(sub_out_dir, "scaler.pkl"))
        joblib.dump(model, os.path.join(sub_out_dir, "model.pkl"))

        meta_dict = {
            "subject": sub_id,
            "pipeline": f"Step 14: {best_config_name}",
            "channel_group": target_ch_group,
            "use_event_alignment": use_event_alignment,
            "use_motor_spectral": use_motor_spectral,
            "use_csp": use_csp,
            "classes": list(model.classes_)
        }
        with open(os.path.join(sub_out_dir, "metadata.json"), "w") as f:
            json.dump(meta_dict, f, indent=2)

        # -------------------------------------------------------------
        # TRANSFORM & PREDICT UNTOUCHED R11 HOLDOUT DATA
        # -------------------------------------------------------------
        # Step 13B Baseline R11 Prediction
        X_eval_ts_base = ts_trans_base.transform(cov_est_base.transform(X_eval))
        X_eval_sc_base = scaler_base.transform(X_eval_ts_base)
        s13b_r11_preds = m_s13b.predict(X_eval_sc_base)

        # Step 14 Selected Model R11 Prediction
        covs_eval = cov_est.transform(X_eval)
        X_eval_ts = ts_trans.transform(covs_eval)
        feat_list_ev = [X_eval_ts]

        if csp_transformer is not None:
            X_eval_csp = csp_transformer.transform(X_eval)
            feat_list_ev.append(X_eval_csp)

        if use_motor_spectral:
            X_eval_spec = extract_spectral_features_batch(X_eval)
            feat_list_ev.append(X_eval_spec)

        X_eval_combined = np.hstack(feat_list_ev) if len(feat_list_ev) > 1 else feat_list_ev[0]
        X_eval_scaled = scaler.transform(X_eval_combined)

        s14_r11_raw_preds = model.predict(X_eval_scaled)
        s14_r11_smooth3_preds = compute_causal_smoothing_series(list(s14_r11_raw_preds), window_size=3)

        all_s13b_r11_actuals.extend(y_eval)
        all_s13b_r11_preds.extend(s13b_r11_preds)

        all_s14_r11_actuals.extend(y_eval)
        all_s14_r11_raw_preds.extend(s14_r11_raw_preds)
        all_s14_r11_smooth3_preds.extend(s14_r11_smooth3_preds)

        # Calculate subject metrics
        s13b_acc = accuracy_score(y_eval, s13b_r11_preds)
        s14_acc = accuracy_score(y_eval, s14_r11_raw_preds)
        s14_f1 = f1_score(y_eval, s14_r11_raw_preds, average="macro", zero_division=0)
        rep_s14 = classification_report(y_eval, s14_r11_raw_preds, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)

        subject_r11_results.append({
            "subject": sub_id,
            "windows": len(y_eval),
            "step13b_accuracy": round(s13b_acc * 100, 2),
            "step14_accuracy": round(s14_acc * 100, 2),
            "accuracy_change": round((s14_acc - s13b_acc) * 100, 2),
            "step14_macro_f1": round(s14_f1 * 100, 2),
            "rest_recall": round(rep_s14.get("REST", {}).get("recall", 0) * 100, 2),
            "left_recall": round(rep_s14.get("LEFT_FIST", {}).get("recall", 0) * 100, 2),
            "right_recall": round(rep_s14.get("RIGHT_FIST", {}).get("recall", 0) * 100, 2)
        })

    # Save Cross-Validation Grid Results
    df_grid = pd.DataFrame(grid_cv_results)
    df_grid.to_csv(os.path.join(out_dir, "experiment_results.csv"), index=False)

    # Save Subject R11 Results
    df_sub_r11 = pd.DataFrame(subject_r11_results)
    df_sub_r11.to_csv(os.path.join(out_dir, "subject_results.csv"), index=False)

    # -------------------------------------------------------------
    # OVERALL UNTOUCHED R11 BENCHMARK & COMPARISON
    # -------------------------------------------------------------
    s13b_acc = float(accuracy_score(all_s13b_r11_actuals, all_s13b_r11_preds) * 100)
    s13b_f1 = float(f1_score(all_s13b_r11_actuals, all_s13b_r11_preds, average="macro", zero_division=0) * 100)
    s13b_rep = classification_report(all_s13b_r11_actuals, all_s13b_r11_preds, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)

    s14_acc = float(accuracy_score(all_s14_r11_actuals, all_s14_r11_raw_preds) * 100)
    s14_prec = float(precision_score(all_s14_r11_actuals, all_s14_r11_raw_preds, average="macro", zero_division=0) * 100)
    s14_rec = float(recall_score(all_s14_r11_actuals, all_s14_r11_raw_preds, average="macro", zero_division=0) * 100)
    s14_f1 = float(f1_score(all_s14_r11_actuals, all_s14_r11_raw_preds, average="macro", zero_division=0) * 100)
    s14_rep = classification_report(all_s14_r11_actuals, all_s14_r11_raw_preds, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)

    # Check temporal smoothing
    s14_s3_rep = classification_report(all_s14_r11_actuals, all_s14_r11_smooth3_preds, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)
    if (s14_s3_rep["LEFT_FIST"]["recall"] < s14_rep["LEFT_FIST"]["recall"]) or (s14_s3_rep["RIGHT_FIST"]["recall"] < s14_rep["RIGHT_FIST"]["recall"]):
        selected_smoothing = "RAW (No smoothing - smoothing reduced motor recall)"
    else:
        selected_smoothing = "3-Window Causal Majority"

    comparison_data = [
        {"metric": "Overall Accuracy", "step13b_class_balanced": round(s13b_acc, 2), "step14_motor_rhythm": round(s14_acc, 2), "change": round(s14_acc - s13b_acc, 2)},
        {"metric": "Macro Precision", "step13b_class_balanced": round(precision_score(all_s13b_r11_actuals, all_s13b_r11_preds, average="macro", zero_division=0)*100, 2), "step14_motor_rhythm": round(s14_prec, 2), "change": round(s14_prec - precision_score(all_s13b_r11_actuals, all_s13b_r11_preds, average="macro", zero_division=0)*100, 2)},
        {"metric": "Macro Recall", "step13b_class_balanced": round(recall_score(all_s13b_r11_actuals, all_s13b_r11_preds, average="macro", zero_division=0)*100, 2), "step14_motor_rhythm": round(s14_rec, 2), "change": round(s14_rec - recall_score(all_s13b_r11_actuals, all_s13b_r11_preds, average="macro", zero_division=0)*100, 2)},
        {"metric": "Macro F1", "step13b_class_balanced": round(s13b_f1, 2), "step14_motor_rhythm": round(s14_f1, 2), "change": round(s14_f1 - s13b_f1, 2)},
        {"metric": "REST Recall", "step13b_class_balanced": round(s13b_rep["REST"]["recall"] * 100, 2), "step14_motor_rhythm": round(s14_rep["REST"]["recall"] * 100, 2), "change": round((s14_rep["REST"]["recall"] - s13b_rep["REST"]["recall"]) * 100, 2)},
        {"metric": "LEFT_FIST Recall", "step13b_class_balanced": round(s13b_rep["LEFT_FIST"]["recall"] * 100, 2), "step14_motor_rhythm": round(s14_rep["LEFT_FIST"]["recall"] * 100, 2), "change": round((s14_rep["LEFT_FIST"]["recall"] - s13b_rep["LEFT_FIST"]["recall"]) * 100, 2)},
        {"metric": "RIGHT_FIST Recall", "step13b_class_balanced": round(s13b_rep["RIGHT_FIST"]["recall"] * 100, 2), "step14_motor_rhythm": round(s14_rep["RIGHT_FIST"]["recall"] * 100, 2), "change": round((s14_rep["RIGHT_FIST"]["recall"] - s13b_rep["RIGHT_FIST"]["recall"]) * 100, 2)},
    ]
    df_comp = pd.DataFrame(comparison_data)
    df_comp.to_csv(os.path.join(out_dir, "comparison.csv"), index=False)

    # 3x3 Confusion Matrix
    cm_s14 = confusion_matrix(all_s14_r11_actuals, all_s14_r11_raw_preds, labels=EXPECTED_CLASSES)
    df_cm = pd.DataFrame(cm_s14, index=[f"True_{c}" for c in EXPECTED_CLASSES], columns=[f"Pred_{c}" for c in EXPECTED_CLASSES])
    df_cm.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

    # Statistical Comparison (Paired t-test across subjects)
    accs_13b = [r["step13b_accuracy"] for r in subject_r11_results]
    accs_14 = [r["step14_accuracy"] for r in subject_r11_results]

    if len(accs_14) > 1 and len(set(np.array(accs_14) - np.array(accs_13b))) > 1:
        t_stat, p_val = ttest_rel(accs_14, accs_13b)
        stat_test = {"test_name": "Paired Student t-test", "t_statistic": round(float(t_stat), 4), "p_value": round(float(p_val), 4), "is_statistically_significant": bool(p_val < 0.05)}
    else:
        stat_test = {"test_name": "Paired Student t-test", "t_statistic": 0.0, "p_value": 1.0, "is_statistically_significant": False}

    # Summary Statistics across S001-S010
    accs = [r["step14_accuracy"] for r in subject_r11_results]
    f1s = [r["step14_macro_f1"] for r in subject_r11_results]
    left_recs = [r["left_recall"] for r in subject_r11_results]
    right_recs = [r["right_recall"] for r in subject_r11_results]
    rest_recs = [r["rest_recall"] for r in subject_r11_results]

    summary_stats = {
        "accuracy": {"mean": round(np.mean(accs), 2), "median": round(np.median(accs), 2), "min": round(np.min(accs), 2), "max": round(np.max(accs), 2), "std": round(np.std(accs), 2)},
        "macro_f1": {"mean": round(np.mean(f1s), 2), "median": round(np.median(f1s), 2), "min": round(np.min(f1s), 2), "max": round(np.max(f1s), 2), "std": round(np.std(f1s), 2)},
        "left_recall": {"mean": round(np.mean(left_recs), 2), "median": round(np.median(left_recs), 2), "min": round(np.min(left_recs), 2), "max": round(np.max(left_recs), 2), "std": round(np.std(left_recs), 2)},
        "right_recall": {"mean": round(np.mean(right_recs), 2), "median": round(np.median(right_recs), 2), "min": round(np.min(right_recs), 2), "max": round(np.max(right_recs), 2), "std": round(np.std(right_recs), 2)},
        "rest_recall": {"mean": round(np.mean(rest_recs), 2), "median": round(np.median(rest_recs), 2), "min": round(np.min(rest_recs), 2), "max": round(np.max(rest_recs), 2), "std": round(np.std(rest_recs), 2)},
    }

    # Final Decision Rule
    passed_s14 = (s14_f1 >= s13b_f1) and (s14_acc >= s13b_acc)
    if passed_s14:
        final_decision = "ADOPT STEP 14"
    else:
        final_decision = "KEEP STEP 13B"

    summary_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "final_decision": final_decision,
        "winning_pipeline_configuration": best_config_name,
        "selected_smoothing_strategy": selected_smoothing,
        "statistical_significance_test": stat_test,
        "step13b_class_balanced_r11_metrics": {
            "accuracy": round(s13b_acc, 2),
            "macro_f1": round(s13b_f1, 2),
            "rest_recall": round(s13b_rep["REST"]["recall"] * 100, 2),
            "left_fist_recall": round(s13b_rep["LEFT_FIST"]["recall"] * 100, 2),
            "right_fist_recall": round(s13b_rep["RIGHT_FIST"]["recall"] * 100, 2),
        },
        "step14_motor_rhythm_r11_metrics": {
            "accuracy": round(s14_acc, 2),
            "macro_f1": round(s14_f1, 2),
            "rest_recall": round(s14_rep["REST"]["recall"] * 100, 2),
            "left_fist_recall": round(s14_rep["LEFT_FIST"]["recall"] * 100, 2),
            "right_fist_recall": round(s14_rep["RIGHT_FIST"]["recall"] * 100, 2),
        },
        "subject_statistics": summary_stats
    }

    with open(os.path.join(out_dir, "summary_report.json"), "w") as f:
        json.dump(summary_report, f, indent=2)

    print("Step 14 Experiments completed successfully.")
    print(f"Final Decision: {final_decision}")
    print(f"Step 13B Acc: {s13b_acc:.2f}%, F1: {s13b_f1:.2f}% | Step 14 Acc: {s14_acc:.2f}%, F1: {s14_f1:.2f}%")
    return summary_report


if __name__ == "__main__":
    run_step14_motor_rhythm_experiments()
