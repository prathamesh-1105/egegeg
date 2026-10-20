import numpy as np
from src.eeg_source import EEGSource


class HardwareEEGSource(EEGSource):
    """
    Physical EEG Hardware Source Adapter.
    Interface template for hardware acquisition devices (e.g. LSL streams, OpenBCI, Cyton, Emotiv).
    Never fabricates fake EEG signals or claims connection unless real physical hardware is verified.
    """

    def __init__(self, device_name: str = "Generic EEG Hardware"):
        self.device_name = device_name
        self.connected = False
        self.streaming = False
        self.status_message = "Hardware source not connected"
        self.sfreq = 160.0
        self.ch_names = []

    def connect(self) -> bool:
        # Hardware sources must perform genuine hardware handshake or stream discovery
        self.connected = False
        self.status_message = f"Hardware source '{self.device_name}' not detected. Connect a physical EEG device or LSL stream."
        return False

    def disconnect(self) -> None:
        self.connected = False
        self.streaming = False
        self.status_message = "Hardware source disconnected"

    def start(self) -> None:
        if self.connected:
            self.streaming = True

    def stop(self) -> None:
        self.streaming = False

    def read_window(self, window_sec: float = 3.0) -> np.ndarray:
        # Return zeros or raise exception rather than fabricating fake signals
        return np.zeros((len(self.ch_names) if len(self.ch_names) > 0 else 15, int(window_sec * self.sfreq)))

    def get_channel_names(self) -> list:
        return self.ch_names

    def get_sampling_rate(self) -> float:
        return self.sfreq

    def is_connected(self) -> bool:
        return self.connected

    def get_status_message(self) -> str:
        return self.status_message
