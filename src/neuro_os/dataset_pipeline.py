"""
NeuroOS Phase 6 — Multi-Dataset Preprocessing & Harmonization Pipeline

Provides reusable, leak-free loading, channel harmonization, resampling,
and feature extraction across heterogeneous motor-imagery EEG datasets:
1. PhysioNet EEGMMIDB Task 1 (Motor Execution: R03, R07, R11)
2. PhysioNet EEGMMIDB Task 2 (Pure Motor Imagery: R04, R08, R12)
3. BCI Competition IV Dataset 2a (GDF format, 22 EEG channels, 250 Hz -> 160 Hz)
"""

import os
import sys
import glob
import json
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import scipy.signal as signal
from collections import Counter

# Target 15-channel motor strip array for NeuroOS
TARGET_MONTAGE = [
    'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C3',  'C1',  'Cz',  'C2',  'C4',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4'
]

TARGET_SFREQ = 160.0  # NeuroOS standard sampling rate (Hz)
TARGET_WINDOW_SEC = 3.0
TARGET_SAMPLES = int(TARGET_SFREQ * TARGET_WINDOW_SEC)  # 480 samples

CLASSES = ["REST", "LEFT_FIST", "RIGHT_FIST"]

# True anatomical channel indices within TARGET_MONTAGE
FC3_IDX = 0
FC1_IDX = 1
FCZ_IDX = 2
FC2_IDX = 3
FC4_IDX = 4
C3_IDX  = 5
C1_IDX  = 6
CZ_IDX  = 7
C2_IDX  = 8
C4_IDX  = 9
CP3_IDX = 10
CP1_IDX = 11
CPZ_IDX = 12
CP2_IDX = 13
CP4_IDX = 14

logger = logging.getLogger("neuro_os.dataset_pipeline")


# ==============================================================================
# 1. Channel Harmonization and Resampling
# ==============================================================================

def clean_channel_name(ch: str) -> str:
    """Standardizes EEG channel name (removes dots, prefixes, whitespace, casing)."""
    ch_clean = ch.replace('.', '').replace('-', '').replace(' ', '').upper()
    if ch_clean.startswith('EEG'):
        ch_clean = ch_clean[3:]
    return ch_clean


def harmonize_montage(
    data: np.ndarray,
    source_channels: List[str],
    target_montage: List[str] = TARGET_MONTAGE
) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts and reorders channels from source EEG to match target montage exactly.
    Raises ValueError if any required channel is missing.
    """
    cleaned_sources = [clean_channel_name(c) for c in source_channels]
    cleaned_targets = [clean_channel_name(c) for c in target_montage]

    idx_map = []
    missing = []
    for tgt in cleaned_targets:
        if tgt in cleaned_sources:
            idx_map.append(cleaned_sources.index(tgt))
        else:
            missing.append(tgt)

    if missing:
        raise ValueError(
            f"Cannot harmonize montage: missing required channels {missing} "
            f"in source channels {cleaned_sources}"
        )

    harmonized_data = data[idx_map, :]
    return harmonized_data, target_montage


def resample_epoch(
    epoch: np.ndarray,
    orig_sfreq: float,
    target_sfreq: float = TARGET_SFREQ
) -> np.ndarray:
    """
    Resamples multi-channel epoch (C, T) to target_sfreq using polyphase/Fourier resampling.
    Guarantees output shape is (C, TARGET_SAMPLES).
    """
    if abs(orig_sfreq - target_sfreq) < 1e-3:
        if epoch.shape[-1] != TARGET_SAMPLES:
            return epoch[:, :TARGET_SAMPLES] if epoch.shape[-1] > TARGET_SAMPLES else np.pad(
                epoch, ((0, 0), (0, TARGET_SAMPLES - epoch.shape[-1]))
            )
        return epoch

    num_target_samples = int(round(epoch.shape[-1] * (target_sfreq / orig_sfreq)))
    resampled = signal.resample(epoch, num_target_samples, axis=-1)

    # Enforce exact TARGET_SAMPLES length
    if resampled.shape[-1] > TARGET_SAMPLES:
        resampled = resampled[:, :TARGET_SAMPLES]
    elif resampled.shape[-1] < TARGET_SAMPLES:
        pad_width = TARGET_SAMPLES - resampled.shape[-1]
        resampled = np.pad(resampled, ((0, 0), (0, pad_width)), mode='edge')

    return resampled.astype(np.float32)


# ==============================================================================
# 2. Spatial Filtering: True Anatomical Surface Laplacian & Contralateral Bipolar
# ==============================================================================

def apply_anatomical_surface_laplacian(X: np.ndarray) -> np.ndarray:
    """
    Computes local orthogonal discrete surface Laplacian for motor cortex electrodes:
    - C3 referenced to orthogonal ring: FC3, C1, CP3
    - C4 referenced to orthogonal ring: FC4, C2, CP4
    - Cz referenced to orthogonal ring: FCz, C1, C2, CPz
    - C1 referenced to orthogonal ring: FC1, C3, Cz, CP1
    - C2 referenced to orthogonal ring: FC2, Cz, C4, CP2
    """
    X_out = X.copy()
    c3_ring = [FC3_IDX, C1_IDX, CP3_IDX]
    c4_ring = [FC4_IDX, C2_IDX, CP4_IDX]
    cz_ring = [FCZ_IDX, C1_IDX, C2_IDX, CPZ_IDX]
    c1_ring = [FC1_IDX, C3_IDX, CZ_IDX, CP1_IDX]
    c2_ring = [FC2_IDX, CZ_IDX, C4_IDX, CP2_IDX]

    X_out[:, C3_IDX, :] = X[:, C3_IDX, :] - np.mean(X[:, c3_ring, :], axis=1)
    X_out[:, C4_IDX, :] = X[:, C4_IDX, :] - np.mean(X[:, c4_ring, :], axis=1)
    X_out[:, CZ_IDX, :] = X[:, CZ_IDX, :] - np.mean(X[:, cz_ring, :], axis=1)
    X_out[:, C1_IDX, :] = X[:, C1_IDX, :] - np.mean(X[:, c1_ring, :], axis=1)
    X_out[:, C2_IDX, :] = X[:, C2_IDX, :] - np.mean(X[:, c2_ring, :], axis=1)
    return X_out


# ==============================================================================
# 3. Feature Extraction (Advanced Contralateral Motor + ERD + Lateralization)
# ==============================================================================

def extract_advanced_motor_features(X: np.ndarray, sfreq: float = 160.0) -> np.ndarray:
    """
    Extracts 21 neurophysiologically grounded sensorimotor rhythm features per window:
    - 15 channel variances after anatomical surface Laplacian.
    - Contralateral motor asymmetry: log(Var(C4)) - log(Var(C3)).
    - Contralateral premotor asymmetry: log(Var(FC4)) - log(Var(FC3)).
    - Contralateral centroparietal asymmetry: log(Var(CP4)) - log(Var(CP3)).
    - Motor normalized ERD ratio: (Var(C3) - Var(C4)) / (Var(C3) + Var(C4)).
    - Paracentral lateral gradient: (Var(C1) - Var(C2)) / (Var(C1) + Var(C2)).
    - Central suppression index: Var(Cz) / (0.5 * (Var(C3) + Var(C4)) + 1e-6).
    """
    N, C, T = X.shape
    features = []

    for i in range(N):
        win = X[i]
        ch_vars = np.var(win, axis=-1)

        c3_v  = float(ch_vars[C3_IDX])
        c4_v  = float(ch_vars[C4_IDX])
        fc3_v = float(ch_vars[FC3_IDX])
        fc4_v = float(ch_vars[FC4_IDX])
        cp3_v = float(ch_vars[CP3_IDX])
        cp4_v = float(ch_vars[CP4_IDX])
        cz_v  = float(ch_vars[CZ_IDX])
        c1_v  = float(ch_vars[C1_IDX])
        c2_v  = float(ch_vars[C2_IDX])

        log_asym_motor = np.log(c4_v + 1e-8) - np.log(c3_v + 1e-8)
        log_asym_premotor = np.log(fc4_v + 1e-8) - np.log(fc3_v + 1e-8)
        log_asym_cp = np.log(cp4_v + 1e-8) - np.log(cp3_v + 1e-8)
        ratio_motor = (c3_v - c4_v) / (c3_v + c4_v + 1e-6)
        ratio_paracentral = (c1_v - c2_v) / (c1_v + c2_v + 1e-6)
        central_suppression = cz_v / (0.5 * (c3_v + c4_v) + 1e-6)

        row = list(ch_vars) + [
            log_asym_motor,
            log_asym_premotor,
            log_asym_cp,
            ratio_motor,
            ratio_paracentral,
            central_suppression
        ]
        features.append(row)

    return np.array(features, dtype=np.float32)


# ==============================================================================
# 4. Multi-Dataset Loaders
# ==============================================================================

def load_physionet_cohort(
    subjects: List[str],
    task: str = "imagery",
    data_dir: str = "data"
) -> Dict[str, Dict[str, Any]]:
    """
    Loads authentic PhysioNet EEGMMIDB runs for specified subjects:
    - task='imagery': Loads R04 & R08 (development calibration) and R12 (evaluation holdout).
    - task='execution': Loads R03 & R07 (development calibration) and R11 (evaluation holdout).
    - task='both': Loads all 6 runs, tagged with condition metadata.
    """
    import mne
    mne.set_log_level('WARNING')
    from src.step8_pipeline import find_subject_runs
    from src.step14_pipeline import extract_event_aligned_windows

    if task == "imagery":
        dev_runs = ["R04", "R08"]
        eval_run = "R12"
    elif task == "execution":
        dev_runs = ["R03", "R07"]
        eval_run = "R11"
    elif task == "both":
        dev_runs = ["R03", "R07", "R04", "R08"]
        eval_run = "R12"
    else:
        raise ValueError(f"Unknown task {task}. Must be 'imagery', 'execution', or 'both'.")

    cohort = {}
    for sub in subjects:
        runs_map = find_subject_runs(sub, data_dir=data_dir)
        sub_record = {"subject_id": sub, "task": task, "runs": {}}

        dev_windows, dev_labels = [], []
        for r_id in dev_runs:
            if r_id in runs_map:
                X_r, y_r, info_r = extract_event_aligned_windows(
                    runs_map[r_id], channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5
                )
                if len(X_r) > 0:
                    sub_record["runs"][r_id] = {"X": X_r, "y": y_r, "info": info_r}
                    dev_windows.append(X_r)
                    dev_labels.append(y_r)

        if dev_windows:
            sub_record["X_dev"] = np.concatenate(dev_windows, axis=0)
            sub_record["y_dev"] = np.concatenate(dev_labels, axis=0)
        else:
            sub_record["X_dev"] = np.empty((0, 15, int(TARGET_SFREQ * TARGET_WINDOW_SEC)))
            sub_record["y_dev"] = np.empty((0,), dtype=str)

        if eval_run in runs_map:
            X_eval, y_eval, info_eval = extract_event_aligned_windows(
                runs_map[eval_run], channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5
            )
            sub_record["runs"][eval_run] = {"X": X_eval, "y": y_eval, "info": info_eval}
            sub_record["X_eval"] = X_eval
            sub_record["y_eval"] = y_eval
        else:
            sub_record["X_eval"] = np.empty((0, 15, int(TARGET_SFREQ * TARGET_WINDOW_SEC)))
            sub_record["y_eval"] = np.empty((0,), dtype=str)

        cohort[sub] = sub_record

    return cohort


def load_bci_competition_iv_2a(
    gdf_path: str,
    target_montage: List[str] = TARGET_MONTAGE
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Parses authentic Graz BCI Competition IV Dataset 2a GDF file:
    - 22 EEG channels resampled to 160 Hz.
    - Extracts 3.0s motor imagery epochs:
      * Event 769 -> LEFT_FIST (Left hand imagery)
      * Event 770 -> RIGHT_FIST (Right hand imagery)
      * Event 276 / Pre-cue baseline [-3.0s, 0.0s] -> REST
    - Reorders channels to match TARGET_MONTAGE exactly.
    """
    import mne
    mne.set_log_level('WARNING')

    if not os.path.exists(gdf_path):
        raise FileNotFoundError(f"GDF file not found: {gdf_path}")

    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    orig_sfreq = raw.info['sfreq']

    # Identify channel names and map to target montage
    raw_ch_names = raw.ch_names
    # Graz 2a GDF standard channels are typically EEG-Fz, EEG-0, etc. or standard names
    standard_graz_22 = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2',
        'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'Oz'
    ]

    ch_map = {}
    if len(raw_ch_names) >= 22:
        for idx in range(min(22, len(raw_ch_names))):
            ch_map[raw_ch_names[idx]] = standard_graz_22[idx]
        raw.rename_channels(ch_map)

    # Pick only target montage
    raw.pick_channels([c for c in TARGET_MONTAGE if c in raw.ch_names])

    # Extract events
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Find codes for left hand (769) and right hand (770)
    # In MNE annotation description: '769' is Left, '770' is Right, '276' is Fixation/Rest
    code_left = None
    code_right = None
    code_rest = None
    for desc, code in event_id.items():
        if '769' in desc:
            code_left = code
        elif '770' in desc:
            code_right = code
        elif '276' in desc or 'fixation' in desc.lower():
            code_rest = code

    windows = []
    labels = []
    metadata = []

    for ev in events:
        sample_idx = ev[0]
        event_code = ev[2]

        label = None
        if event_code == code_left:
            label = "LEFT_FIST"
        elif event_code == code_right:
            label = "RIGHT_FIST"
        elif event_code == code_rest:
            label = "REST"

        if label is None:
            continue

        # Extract 3.0s window starting 0.5s after cue onset
        onset_sample = int(sample_idx + 0.5 * orig_sfreq)
        offset_sample = int(onset_sample + TARGET_WINDOW_SEC * orig_sfreq)

        if offset_sample > raw.n_times:
            continue

        epoch_data = raw.get_data(start=onset_sample, stop=offset_sample)
        if np.isnan(epoch_data).any() or np.isinf(epoch_data).any():
            continue

        # Harmonize montage order
        harmonized, _ = harmonize_montage(epoch_data, raw.ch_names, TARGET_MONTAGE)

        # Resample to 160 Hz
        resampled = resample_epoch(harmonized, orig_sfreq=orig_sfreq, target_sfreq=TARGET_SFREQ)

        windows.append(resampled)
        labels.append(label)
        metadata.append({
            "source": os.path.basename(gdf_path),
            "label": label,
            "onset_sec": round(onset_sample / orig_sfreq, 3),
            "orig_sfreq": orig_sfreq,
            "target_sfreq": TARGET_SFREQ
        })

    return np.array(windows, dtype=np.float32), np.array(labels), metadata
