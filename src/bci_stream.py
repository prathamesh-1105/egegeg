import os
import numpy as np
import pandas as pd
import mne

from src.load_eeg import load_eeg_raw
from src.events import get_eeg_events
from src.build_dataset import find_target_channel, validate_window_quality, LABEL_MAPPING


class RealEEGStreamer:
    """
    Simulates a real-time BCI stream by sequentially reading 2-second windows from a real PhysioNet EDF file.
    No synthetic, fake, or randomized EEG data is used.
    """

    def __init__(
        self,
        edf_path: str = "data/raw/S009R03.edf",
        window_sec: float = 2.0,
        overlap: float = 0.5,
        target_channel: str = None
    ):
        if not os.path.exists(edf_path):
            # Fallback to root data folder if raw subfolder path doesn't exist directly
            alt_path = os.path.join("data", os.path.basename(edf_path))
            if os.path.exists(alt_path):
                edf_path = alt_path
            else:
                raise FileNotFoundError(f"EEG recording not found at '{edf_path}'")

        self.edf_path = edf_path
        self.filename = os.path.basename(edf_path)
        self.raw = load_eeg_raw(edf_path)
        self.sfreq = float(self.raw.info['sfreq'])
        self.df_events = get_eeg_events(self.raw)

        # Select target channel (auto-detect C3/Cz/C4 if not specified)
        self.ch_name = target_channel if (target_channel and target_channel in self.raw.ch_names) else find_target_channel(self.raw.ch_names)
        self.ch_idx = self.raw.ch_names.index(self.ch_name)

        self.window_sec = window_sec
        self.overlap = overlap
        self.win_samples = int(window_sec * self.sfreq)
        self.step_samples = int(self.win_samples * (1.0 - overlap))

        self.windows = self._slice_recording_windows()

    def _slice_recording_windows(self) -> list:
        """
        Slices the real EEG recording into sequential 2-second windows across annotated events.

        Returns:
            list: List of window dictionaries.
        """
        windows = []
        if self.df_events.empty:
            return windows

        for _, ev in self.df_events.iterrows():
            code = ev['event_code']
            if code not in LABEL_MAPPING:
                continue

            label = LABEL_MAPPING[code]
            onset_sample = int(ev['onset_sec'] * self.sfreq)
            end_sample = int(ev['end_sec'] * self.sfreq)

            current_sample = onset_sample
            while current_sample + self.win_samples <= end_sample:
                win_end_sample = current_sample + self.win_samples
                win_start_time = round(current_sample / self.sfreq, 4)
                win_end_time = round(win_end_sample / self.sfreq, 4)

                # Extract real microvolt array
                win_signal_uv = self.raw.get_data(picks=[self.ch_idx], start=current_sample, stop=win_end_sample)[0] * 1e6
                win_times = self.raw.times[current_sample:win_end_sample]

                if validate_window_quality(win_signal_uv, self.win_samples):
                    windows.append({
                        "window_index": len(windows),
                        "filename": self.filename,
                        "channel_name": self.ch_name,
                        "window_start": win_start_time,
                        "window_end": win_end_time,
                        "signal_uv": win_signal_uv,
                        "times": win_times,
                        "actual_label": label,
                        "event_code": code
                    })
                current_sample += self.step_samples

        return windows

    def get_total_windows(self) -> int:
        return len(self.windows)

    def get_window(self, index: int) -> dict:
        if not self.windows:
            raise ValueError("No valid EEG windows extracted from recording.")
        idx = max(0, min(index, len(self.windows) - 1))
        win = self.windows[idx].copy()
        win["total_windows"] = len(self.windows)
        return win
