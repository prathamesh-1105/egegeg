import os
import numpy as np
import mne

from src.eeg_source import EEGSource
from src.step7_pipeline import get_channel_indices
from src.step8_pipeline import find_subject_runs
from src.step9_pipeline import extract_raw_multichannel_windows


class DatasetEEGSource(EEGSource):
    """
    PhysioNet EDF Dataset EEG Source.
    Provides real multi-channel EEG windows sequentially from PhysioNet EDF recordings.
    """

    def __init__(self, subject_id: str = "S001", run_name: str = "R03", data_dir: str = "data", channel_group: str = "Group B"):
        self.subject_id = subject_id
        self.run_name = run_name
        self.data_dir = data_dir
        self.channel_group = channel_group
        self.connected = False
        self.streaming = False

        self.edf_path = None
        self.raw = None
        self.sfreq = 160.0
        self.ch_names = []

        self.windows = []
        self.labels = []
        self.infos = []
        self.current_idx = 0

    def connect(self) -> bool:
        runs = find_subject_runs(self.subject_id, data_dir=self.data_dir)
        self.edf_path = runs.get(self.run_name, None)

        if not self.edf_path or not os.path.exists(self.edf_path):
            self.connected = False
            return False

        try:
            self.raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            self.sfreq = float(self.raw.info['sfreq'])
            self.ch_names = self.raw.info['ch_names']

            self.windows, self.labels, self.infos = extract_raw_multichannel_windows(
                self.edf_path, channel_group=self.channel_group, window_sec=3.0, overlap=0.5
            )
            self.current_idx = 0
            self.connected = True
            return True
        except Exception as e:
            print(f"Error loading EDF dataset: {e}")
            self.connected = False
            return False

    def disconnect(self) -> None:
        self.connected = False
        self.streaming = False
        self.raw = None

    def start(self) -> None:
        if self.connected:
            self.streaming = True

    def stop(self) -> None:
        self.streaming = False

    def read_window(self, window_sec: float = 3.0) -> np.ndarray:
        if not self.connected or len(self.windows) == 0:
            return np.zeros((15, int(window_sec * self.sfreq)))

        win = self.windows[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.windows)
        return win

    def get_current_label(self) -> str:
        if len(self.labels) > 0:
            idx = (self.current_idx - 1) % len(self.labels)
            return self.labels[idx]
        return "REST"

    def get_channel_names(self) -> list:
        if self.raw:
            indices = get_channel_indices(self.raw.ch_names, group_type=self.channel_group)
            return [self.raw.ch_names[i] for i in indices]
        return self.ch_names

    def get_sampling_rate(self) -> float:
        return self.sfreq

    def is_connected(self) -> bool:
        return self.connected
