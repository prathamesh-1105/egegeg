import os
import glob
import numpy as np
import pandas as pd
import mne

from src.load_eeg import load_eeg_raw
from src.events import get_eeg_events
from src.feature_extraction import compute_feature_vector


# Map PhysioNet event codes to required dataset labels
LABEL_MAPPING = {
    "T0": "REST",
    "T1": "LEFT_FIST",
    "T2": "RIGHT_FIST"
}


def find_target_channel(ch_names: list, preferred_prefixes: list = ["C3", "Cz", "C4"]) -> str:
    """
    Dynamically finds the best available motor cortex EEG channel name.

    Parameters:
        ch_names (list): List of channel names present in EDF file.
        preferred_prefixes (list): Order of channel label prefixes to search for.

    Returns:
        str: Selected channel name.
    """
    for pref in preferred_prefixes:
        for ch in ch_names:
            clean_ch = ch.strip().rstrip('.')
            if clean_ch.upper() == pref.upper() or ch.startswith(pref):
                return ch
    return ch_names[0]


def validate_window_quality(window_uv: np.ndarray, expected_samples: int) -> bool:
    """
    Performs data quality validation checks on an extracted signal window.

    Checks:
    - Signal non-empty
    - Exact sample count matches expected window length
    - Zero NaN values
    - Zero Infinity values

    Parameters:
        window_uv (np.ndarray): Microvolt signal window array.
        expected_samples (int): Expected window length in samples.

    Returns:
        bool: True if window passes all quality checks, False otherwise.
    """
    if window_uv is None or len(window_uv) == 0:
        return False
    if len(window_uv) != expected_samples:
        return False
    if np.isnan(window_uv).any():
        return False
    if np.isinf(window_uv).any():
        return False
    return True


def discover_edf_files(data_dir: str = "data", subjects: list = None) -> list:
    """
    Recursively discovers all EDF files matching specified subject IDs.

    Parameters:
        data_dir (str): Root search directory.
        subjects (list): List of subject IDs to include (e.g. ['S001', ..., 'S010']).

    Returns:
        list: Sorted list of absolute filepaths to matched EDF files.
    """
    if subjects is None:
        subjects = [f"S{i:03d}" for i in range(1, 11)]

    matched_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".edf"):
                filename = os.path.basename(f)
                for sub in subjects:
                    if filename.startswith(sub):
                        matched_files.append(os.path.abspath(os.path.join(root, f)))
                        break

    return sorted(list(set(matched_files)))


def build_eeg_feature_dataset(
    subjects: list = None,
    data_dir: str = "data",
    output_csv: str = "data/processed/eeg_features.csv",
    window_sec: float = 2.0,
    overlap: float = 0.5,
    target_channel: str = None
) -> tuple[pd.DataFrame, dict]:
    """
    Processes real PhysioNet EDF files, slices event segments into 2s overlapping windows,
    extracts 13 mathematical features per window, performs quality control checks,
    and exports data/processed/eeg_features.csv.

    Parameters:
        subjects (list): List of subject IDs (e.g. ['S001', 'S002', ..., 'S010']).
        data_dir (str): Root data directory.
        output_csv (str): Output CSV filepath.
        window_sec (float): Window duration in seconds (default 2.0s).
        overlap (float): Window overlap fraction (default 0.5 = 50%).
        target_channel (str): Override channel label or None to auto-detect motor channel.

    Returns:
        df_features (pd.DataFrame): Generated feature dataset DataFrame.
        stats (dict): Dataset generation summary metrics.
    """
    if subjects is None:
        subjects = [f"S{i:03d}" for i in range(1, 11)]

    edf_files = discover_edf_files(data_dir=data_dir, subjects=subjects)

    if not edf_files:
        raise FileNotFoundError(f"No EDF files found for subjects {subjects} in '{data_dir}'")

    rows = []
    rejected_count = 0
    recordings_processed = 0

    for edf_path in edf_files:
        filename = os.path.basename(edf_path)
        subject_id = filename[:4] if filename.startswith("S") else "S001"
        run_id = filename[4:7] if len(filename) >= 7 else "R03"

        try:
            raw = load_eeg_raw(edf_path)
        except Exception as e:
            print(f"Skipping unreadable file {edf_path}: {e}")
            continue

        recordings_processed += 1
        sfreq = raw.info['sfreq']
        df_events = get_eeg_events(raw)

        if df_events.empty:
            continue

        # Select target channel (auto-detect motor channel C3/Cz/C4 if not specified)
        ch_name = target_channel if (target_channel and target_channel in raw.ch_names) else find_target_channel(raw.ch_names)
        ch_idx = raw.ch_names.index(ch_name)

        # Windowing configuration
        win_samples = int(window_sec * sfreq)
        step_samples = int(win_samples * (1.0 - overlap))

        for _, ev in df_events.iterrows():
            code = ev['event_code']
            if code not in LABEL_MAPPING:
                continue

            label = LABEL_MAPPING[code]
            onset_sample = int(ev['onset_sec'] * sfreq)
            end_sample = int(ev['end_sec'] * sfreq)

            # Slide window across the event segment
            current_sample = onset_sample
            while current_sample + win_samples <= end_sample:
                win_end_sample = current_sample + win_samples
                win_start_time = round(current_sample / sfreq, 4)
                win_end_time = round(win_end_sample / sfreq, 4)

                # Extract window signal microvolts
                win_signal_uv = raw.get_data(picks=[ch_idx], start=current_sample, stop=win_end_sample)[0] * 1e6

                # Data Quality Check
                if not validate_window_quality(win_signal_uv, win_samples):
                    rejected_count += 1
                    current_sample += step_samples
                    continue

                # Compute 13 Mathematical Features
                feats = compute_feature_vector(win_signal_uv, sfreq)

                row_dict = {
                    "subject": subject_id,
                    "run": run_id,
                    "window_start": win_start_time,
                    "window_end": win_end_time,
                    "label": label,
                    "mean": feats["Mean (µV)"],
                    "variance": feats["Variance (µV²)"],
                    "std": feats["Std Dev (µV)"],
                    "rms": feats["RMS (µV)"],
                    "min": feats["Min (µV)"],
                    "max": feats["Max (µV)"],
                    "peak_to_peak": feats["Peak-to-Peak (µV)"],
                    "energy": feats["Energy (µV²)"],
                    "delta_power": feats["Delta Power (0.5–4 Hz)"],
                    "theta_power": feats["Theta Power (4–8 Hz)"],
                    "alpha_power": feats["Alpha Power (8–13 Hz)"],
                    "beta_power": feats["Beta Power (13–30 Hz)"],
                    "gamma_power": feats["Gamma Power (30–45 Hz)"]
                }
                rows.append(row_dict)
                current_sample += step_samples

    df_features = pd.DataFrame(rows)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_features.to_csv(output_csv, index=False)

    # Calculate summary metrics
    unique_subjects = df_features["subject"].nunique() if not df_features.empty else 0
    total_windows = len(df_features)
    rest_count = len(df_features[df_features["label"] == "REST"]) if not df_features.empty else 0
    left_count = len(df_features[df_features["label"] == "LEFT_FIST"]) if not df_features.empty else 0
    right_count = len(df_features[df_features["label"] == "RIGHT_FIST"]) if not df_features.empty else 0

    stats = {
        "subjects_processed": unique_subjects,
        "recordings_processed": recordings_processed,
        "total_windows": total_windows,
        "rest_windows": rest_count,
        "left_fist_windows": left_count,
        "right_fist_windows": right_count,
        "rejected_windows": rejected_count,
        "features_per_window": 13,
        "output_csv": output_csv
    }

    return df_features, stats


if __name__ == "__main__":
    df, stats = build_eeg_feature_dataset()
    print("Dataset generated successfully!")
    print(stats)
