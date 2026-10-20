import numpy as np
from typing import List, Dict, Optional
from src.eeg_source import EEGSource
from src.lsl_eeg_source import LSLEEGSource, PYLSL_AVAILABLE
from src.hardware.openbci_cyton_daisy import OpenBCICytonDaisyProfile


class OpenBCISource(EEGSource):
    """
    Dedicated OpenBCI Cyton + Daisy Hardware Source.
    Implements EEGSource interface via Lab Streaming Layer (LSL) connection adapter.
    """

    def __init__(self, stream_name: str = None):
        self.profile = OpenBCICytonDaisyProfile()
        self.stream_name = stream_name
        self.lsl_source = LSLEEGSource(stream_name=stream_name)
        self.connected = False
        self.streaming = False
        self.selected_stream_info: Optional[Dict] = None

    @staticmethod
    def is_lsl_available() -> bool:
        return PYLSL_AVAILABLE

    @classmethod
    def discover_openbci_streams(cls) -> List[Dict]:
        """
        Discovers active LSL EEG streams matching OpenBCI characteristics.
        """
        if not PYLSL_AVAILABLE:
            return []
        
        all_streams = LSLEEGSource.discover_streams()
        openbci_matched = []

        for st in all_streams:
            st_name = st.get("name", "").lower()
            st_ch_count = st.get("channel_count", 0)
            
            is_openbci_candidate = ("openbci" in st_name or st_ch_count >= 16 or "cyton" in st_name or "daisy" in st_name)
            openbci_matched.append({
                "name": st["name"],
                "type": st["type"],
                "channel_count": st["channel_count"],
                "sampling_rate": st["sampling_rate"],
                "source_id": st["source_id"],
                "is_openbci_candidate": is_openbci_candidate
            })
            
        return openbci_matched

    def connect(self, stream_name: str = None) -> bool:
        """
        Attempts to connect to an OpenBCI LSL stream.
        """
        if stream_name:
            self.stream_name = stream_name
            self.lsl_source = LSLEEGSource(stream_name=stream_name)

        success = self.lsl_source.connect()
        if success:
            self.connected = True
            self.selected_stream_info = {
                "name": self.lsl_source.stream_name or "OpenBCI_EEG",
                "sampling_rate": self.lsl_source.get_sampling_rate(),
                "channel_names": self.lsl_source.get_channel_names()
            }
        else:
            self.connected = False
            self.selected_stream_info = None

        return self.connected

    def disconnect(self) -> None:
        self.lsl_source.disconnect()
        self.connected = False
        self.streaming = False
        self.selected_stream_info = None

    def start(self) -> None:
        if self.connected:
            self.lsl_source.start()
            self.streaming = True

    def stop(self) -> None:
        self.lsl_source.stop()
        self.streaming = False

    def read_window(self, window_sec: float = 3.0) -> np.ndarray:
        if not self.connected:
            return np.zeros((16, int(window_sec * self.profile.hardware_sampling_rate_hz)))
        return self.lsl_source.read_window(window_sec=window_sec)

    def get_channel_names(self) -> list:
        if self.connected:
            return self.lsl_source.get_channel_names()
        return self.profile.default_device_channels

    def get_sampling_rate(self) -> float:
        if self.connected:
            return self.lsl_source.get_sampling_rate()
        return self.profile.hardware_sampling_rate_hz

    def is_connected(self) -> bool:
        return self.connected and self.lsl_source.is_connected()
