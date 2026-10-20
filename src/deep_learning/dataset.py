import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional

from src.load_eeg import load_eeg_raw
from src.channel_mapper import ChannelMapper, EXPECTED_GROUP_B_CHANNELS
from src.events import get_eeg_events
from src.step7_pipeline import EXPECTED_CLASSES


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG Motor Imagery Windows.
    Input Tensor Shape: (1, n_channels=15, n_samples=480)
    Label: int64 (0=REST, 1=LEFT_FIST, 2=RIGHT_FIST)
    """

    def __init__(self, X_windows: np.ndarray, y_labels: np.ndarray, transform=None):
        """
        X_windows: (N, n_channels=15, n_samples=480)
        y_labels: (N,) string or int array
        """
        self.X = torch.tensor(X_windows, dtype=torch.float32)

        # Convert string labels to integer indices if necessary
        if len(y_labels) > 0 and isinstance(y_labels[0], str):
            label_to_idx = {cls: idx for idx, cls in enumerate(EXPECTED_CLASSES)}
            y_idx = np.array([label_to_idx.get(lbl, 0) for lbl in y_labels], dtype=np.int64)
        else:
            y_idx = np.array(y_labels, dtype=np.int64)

        self.y = torch.tensor(y_idx, dtype=torch.int64)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if x.dim() == 2:
            x = x.unsqueeze(0) # (1, 15, 480)
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]


class ChannelScaler:
    """
    Leakage-safe channel-wise standard scaler.
    Computes mean and std strictly on training set windows and applies to val/test sets.
    """
    def __init__(self):
        self.mean = None # (n_channels, 1)
        self.std = None  # (n_channels, 1)

    def fit(self, X_windows: np.ndarray):
        """
        X_windows shape: (N, n_channels=15, n_samples=480)
        """
        # Compute mean and std across all windows N and time samples T for each channel C
        # Axis 0: Windows, Axis 2: Time samples
        self.mean = np.mean(X_windows, axis=(0, 2), keepdims=True) # (1, 15, 1)
        self.std = np.std(X_windows, axis=(0, 2), keepdims=True)   # (1, 15, 1)
        self.std[self.std < 1e-6] = 1.0 # Prevent division by zero
        return self

    def transform(self, X_windows: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet!")
        return (X_windows - self.mean) / self.std

    def fit_transform(self, X_windows: np.ndarray) -> np.ndarray:
        self.fit(X_windows)
        return self.transform(X_windows)


def extract_windows_from_subject_runs(
    subjects: List[str],
    runs: List[str],
    data_dir: str = "data",
    window_sec: float = 3.0,
    sfreq_target: float = 160.0,
    step_sec: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts raw 15-channel EEG 3-second windows from specified PhysioNet subjects and runs.
    Returns:
        X_windows: (N, 15, 480)
        y_labels: (N,) string array ('REST', 'LEFT_FIST', 'RIGHT_FIST')
    """
    window_samples = int(window_sec * sfreq_target)
    step_samples = int(step_sec * sfreq_target)

    all_windows = []
    all_labels = []

    for sub in subjects:
        for run in runs:
            edf_file = os.path.join(data_dir, f"{sub}{run}.edf")
            if not os.path.exists(edf_file):
                edf_file = os.path.join(data_dir, "raw", f"{sub}{run}.edf")
            if not os.path.exists(edf_file):
                continue

            try:
                raw = load_eeg_raw(edf_file)

                # Select and reorder to 15 Group B channels FIRST to speed up filtering by 4x
                ch_map = ChannelMapper()
                res_map = ch_map.validate_channels(raw.ch_names)
                if res_map["is_compatible"]:
                    data = ch_map.reorder_channels(raw.get_data(), res_map["mapped_indices"])
                else:
                    norm_available = [ch_map.normalize_channel_name(c) for c in raw.ch_names]
                    indices = [norm_available.index(ch_map.normalize_channel_name(target)) for target in EXPECTED_GROUP_B_CHANNELS if ch_map.normalize_channel_name(target) in norm_available]
                    data = raw.get_data()[indices, :] # (15, n_times)

                # Apply 1–45 Hz bandpass filtering on 15 channels
                from scipy import signal
                b, a = signal.butter(4, [1.0 / (sfreq_target / 2.0), 45.0 / (sfreq_target / 2.0)], btype='band')
                data = signal.filtfilt(b, a, data, axis=1)

                # Get events
                df_events = get_eeg_events(raw)
                code_to_label = {"T0": "REST", "T1": "LEFT_FIST", "T2": "RIGHT_FIST"}

                # Slide windows across signal
                n_times = data.shape[1]
                for start in range(0, n_times - window_samples + 1, step_samples):
                    end = start + window_samples
                    win = data[:, start:end]

                    # Assign label based on event midpoint
                    mid_sec = (start + window_samples / 2.0) / sfreq_target
                    win_label = "REST"

                    if df_events is not None and not df_events.empty:
                        for _, row in df_events.iterrows():
                            t_start = row["onset_sec"]
                            t_end = row["end_sec"]
                            if t_start <= mid_sec <= t_end:
                                code = row.get("event_code", "T0")
                                win_label = code_to_label.get(code, "REST")
                                break

                    all_windows.append(win)
                    all_labels.append(win_label)
            except Exception as e:
                print(f"Warning: Could not process {sub}{run}: {e}")

    if not all_windows:
        return np.zeros((0, 15, window_samples)), np.array([])

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=str)
    return X, y


def create_leakage_safe_dataloaders(
    train_subs: List[str] = [f"S{i:03d}" for i in range(1, 8)],
    val_subs: List[str] = ["S008"],
    test_subs: List[str] = ["S009", "S010"],
    train_runs: List[str] = ["R03", "R07"],
    test_runs: List[str] = ["R11"],
    data_dir: str = "data",
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader, ChannelScaler, Dict]:
    """
    Builds leakage-safe PyTorch DataLoaders and ChannelScaler.
    Train statistics (mean, std) are computed strictly on (train_subs, train_runs).
    """
    # 1. Extract Train Windows
    X_train_raw, y_train = extract_windows_from_subject_runs(train_subs, train_runs, data_dir=data_dir)

    # 2. Extract Validation Windows
    X_val_raw, y_val = extract_windows_from_subject_runs(val_subs, train_runs, data_dir=data_dir)

    # 3. Extract Test Windows (Holdout R11 on S009-S010)
    X_test_raw, y_test = extract_windows_from_subject_runs(test_subs, test_runs, data_dir=data_dir)

    # 4. Leakage-safe normalization: Fit ONLY on Train
    scaler = ChannelScaler()
    X_train_norm = scaler.fit_transform(X_train_raw) if len(X_train_raw) > 0 else X_train_raw
    X_val_norm = scaler.transform(X_val_raw) if len(X_val_raw) > 0 else X_val_raw
    X_test_norm = scaler.transform(X_test_raw) if len(X_test_raw) > 0 else X_test_raw

    # 5. Create PyTorch Datasets
    ds_train = EEGDataset(X_train_norm, y_train)
    ds_val = EEGDataset(X_val_norm, y_val)
    ds_test = EEGDataset(X_test_norm, y_test)

    # 6. Create DataLoaders
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    metadata = {
        "n_train_windows": len(X_train_norm),
        "n_val_windows": len(X_val_norm),
        "n_test_windows": len(X_test_norm),
        "train_subs": train_subs,
        "val_subs": val_subs,
        "test_subs": test_subs,
        "train_runs": train_runs,
        "test_runs": test_runs
    }

    return dl_train, dl_val, dl_test, scaler, metadata
