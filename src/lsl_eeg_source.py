import numpy as np
from src.eeg_source import EEGSource

try:
    import pylsl
    PYLSL_AVAILABLE = True
except ImportError:
    PYLSL_AVAILABLE = False


class LSLEEGSource(EEGSource):
    """
    Lab Streaming Layer (LSL) Hardware EEG Source.
    Discovers, connects to, and streams from real physical EEG acquisition devices broadcasting over LSL.
    """

    def __init__(self, stream_name: str = None):
        self.stream_name = stream_name
        self.connected = False
        self.streaming = False
        self.inlet = None
        self.info = None
        self.sfreq = 160.0
        self.ch_names = []
        self.buffer = []

    @staticmethod
    def is_lsl_available() -> bool:
        return PYLSL_AVAILABLE

    @staticmethod
    def discover_streams() -> list:
        if not PYLSL_AVAILABLE:
            return []
        try:
            streams = pylsl.resolve_streams(wait_time=1.0)
            stream_info_list = []
            for st in streams:
                if st.type().lower() == "eeg":
                    stream_info_list.append({
                        "name": st.name(),
                        "type": st.type(),
                        "channel_count": st.channel_count(),
                        "sampling_rate": st.nominal_srate(),
                        "source_id": st.source_id()
                    })
            return stream_info_list
        except Exception:
            return []

    def connect(self) -> bool:
        if not PYLSL_AVAILABLE:
            self.connected = False
            return False

        try:
            streams = pylsl.resolve_streams(wait_time=1.0)
            eeg_streams = [s for s in streams if s.type().lower() == "eeg"]

            if not eeg_streams:
                self.connected = False
                return False

            selected_stream = None
            if self.stream_name:
                for s in eeg_streams:
                    if s.name() == self.stream_name:
                        selected_stream = s
                        break
            if not selected_stream and len(eeg_streams) > 0:
                selected_stream = eeg_streams[0]

            if not selected_stream:
                self.connected = False
                return False

            self.inlet = pylsl.StreamInlet(selected_stream)
            self.info = self.inlet.info()
            self.sfreq = float(self.info.nominal_srate())
            n_ch = self.info.channel_count()

            # Attempt to extract channel names from LSL XML metadata
            ch_names = []
            try:
                chans = self.info.desc().child("channels").child("channel")
                for i in range(n_ch):
                    name = chans.child_value("label")
                    if name:
                        ch_names.append(name)
                    chans = chans.next_sibling("channel")
            except Exception:
                pass

            if len(chans_names := ch_names) == n_ch:
                self.ch_names = chans_names
            else:
                self.ch_names = [f"Ch{i+1}" for i in range(n_ch)]

            self.connected = True
            return True
        except Exception as e:
            print(f"LSL connection error: {e}")
            self.connected = False
            return False

    def disconnect(self) -> None:
        if self.inlet:
            try:
                self.inlet.close_stream()
            except Exception:
                pass
        self.inlet = None
        self.connected = False
        self.streaming = False

    def start(self) -> None:
        if self.connected:
            self.streaming = True

    def stop(self) -> None:
        self.streaming = False

    def read_window(self, window_sec: float = 3.0) -> np.ndarray:
        if not self.connected or not self.inlet:
            return np.zeros((len(self.ch_names) if self.ch_names else 15, int(window_sec * self.sfreq)))

        n_samples_target = int(window_sec * self.sfreq)
        chunk, timestamps = self.inlet.pull_chunk(timeout=0.1, max_samples=n_samples_target)

        if chunk:
            chunk_arr = np.array(chunk).T  # Shape: (n_channels, n_samples)
            for s_idx in range(chunk_arr.shape[1]):
                self.buffer.append(chunk_arr[:, s_idx])
            
            # Retain sliding window of target size
            if len(self.buffer) > n_samples_target:
                self.buffer = self.buffer[-n_samples_target:]

        if len(self.buffer) == 0:
            return np.zeros((len(self.ch_names) if self.ch_names else 15, n_samples_target))

        buf_arr = np.array(self.buffer).T
        if buf_arr.shape[1] < n_samples_target:
            # Pad with zeroes if buffer is filling up
            pad_width = n_samples_target - buf_arr.shape[1]
            buf_arr = np.pad(buf_arr, ((0, 0), (pad_width, 0)), mode='constant')

        return buf_arr

    def get_channel_names(self) -> list:
        return self.ch_names

    def get_sampling_rate(self) -> float:
        return self.sfreq

    def is_connected(self) -> bool:
        return self.connected
