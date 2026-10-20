import os
import mne
import numpy as np


class EEGFileNotFoundError(FileNotFoundError):
    """Custom exception raised when the required EDF file is not found."""
    pass


def load_eeg_raw(file_path: str = "data/S001R03.edf") -> mne.io.Raw:
    """
    Loads real EEG data from an EDF file using MNE-Python.

    Strict Rule: Does NOT create simulated or dummy data if the file is missing.
    Raises EEGFileNotFoundError with instructions on where to place the file.

    Parameters:
        file_path (str): Relative or absolute path to the .edf file.

    Returns:
        raw (mne.io.Raw): MNE Raw object containing preload=True raw EEG data.
    """
    if not os.path.exists(file_path):
        abs_path = os.path.abspath(file_path)
        raise EEGFileNotFoundError(
            f"\n[ERROR] REAL EEG DATASET FILE NOT FOUND!\n"
            f"Expected file at: '{abs_path}'\n"
            f"Please place 'S001R03.edf' and 'S001R03.edf.event' into the 'data/' folder:\n"
            f"  -> {os.path.dirname(abs_path)}\n"
            f"Do NOT use simulated data. This application requires real PhysioNet EEG recordings."
        )

    try:
        # Load raw EDF file using MNE with preload=True to store data in memory
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        return raw
    except Exception as e:
        raise RuntimeError(f"Failed to parse real EDF file '{file_path}': {str(e)}")


def get_eeg_info(raw: mne.io.Raw, file_path: str = "S001R03.edf") -> dict:
    """
    Extracts key metadata directly from the MNE Raw object.

    Parameters:
        raw (mne.io.Raw): Loaded MNE Raw object.
        file_path (str): Name or path of the EDF file.

    Returns:
        dict: Metadata dictionary containing exact properties of the EDF recording.
    """
    n_channels = len(raw.ch_names)
    ch_names = list(raw.ch_names)
    sfreq = float(raw.info['sfreq'])
    duration = float(raw.times[-1] if len(raw.times) > 0 else 0.0)
    n_samples = int(raw.n_times)

    return {
        "file_name": os.path.basename(file_path),
        "file_path": os.path.abspath(file_path),
        "n_channels": n_channels,
        "ch_names": ch_names,
        "sfreq": sfreq,
        "duration_sec": duration,
        "n_samples": n_samples,
    }
