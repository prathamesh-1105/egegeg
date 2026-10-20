import os
import json
import time
import datetime
import numpy as np


class HardwareSessionRecorder:
    """
    Non-Destructive EEG Hardware Session & Calibration Data Recorder.
    Saves raw multi-channel EEG signals and timestamped ground-truth event markers
    labeled 'USER-PROVIDED CALIBRATION LABEL' into data/hardware_sessions/.
    Never automatically overwrites Step 17B champion model artifacts.
    """

    def __init__(self, output_dir: str = "data/hardware_sessions"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.session_active = False
        self.session_id = None
        self.device_name = "Generic LSL EEG Device"
        self.sfreq = 160.0
        self.ch_names = []
        self.raw_samples = []
        self.events = []
        self.start_time = None

    def start_session(self, device_name: str = "LSL Hardware Device", sfreq: float = 160.0, ch_names: list = None) -> str:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{timestamp_str}"
        self.device_name = device_name
        self.sfreq = sfreq
        self.ch_names = ch_names if ch_names else [f"Ch{i+1}" for i in range(15)]
        self.raw_samples = []
        self.events = []
        self.start_time = time.time()
        self.session_active = True
        return self.session_id

    def log_event_marker(self, label: str) -> dict:
        """
        Record a timestamped event marker labeled 'USER-PROVIDED CALIBRATION LABEL'.
        """
        if not self.session_active:
            return {}

        rel_sec = round(time.time() - self.start_time, 4)
        sample_idx = int(rel_sec * self.sfreq)

        evt_entry = {
            "event_label": label,
            "label_type": "USER-PROVIDED CALIBRATION LABEL",
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "relative_sec": rel_sec,
            "sample_index": sample_idx
        }
        self.events.append(evt_entry)
        return evt_entry

    def append_raw_chunk(self, chunk_data: np.ndarray) -> None:
        """
        Append raw chunk data of shape (n_channels, n_samples).
        """
        if not self.session_active or chunk_data.size == 0:
            return
        
        for s_idx in range(chunk_data.shape[1]):
            self.raw_samples.append(chunk_data[:, s_idx].tolist())

    def stop_and_save_session(self) -> str:
        """
        Saves recorded raw EEG session and metadata into non-destructive JSON format.
        Returns saved file path.
        """
        if not self.session_active or not self.session_id:
            return ""

        self.session_active = False
        duration_sec = round(time.time() - self.start_time, 2)

        session_payload = {
            "session_id": self.session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "device_name": self.device_name,
            "sampling_rate_hz": self.sfreq,
            "channel_names": self.ch_names,
            "channel_count": len(self.ch_names),
            "total_samples": len(self.raw_samples),
            "duration_sec": duration_sec,
            "events_count": len(self.events),
            "events": self.events,
            "raw_eeg_uv": self.raw_samples
        }

        save_path = os.path.join(self.output_dir, f"{self.session_id}.json")
        with open(save_path, "w") as f:
            json.dump(session_payload, f, indent=2)

        print(f"Hardware Session saved to {save_path} ({len(self.raw_samples)} samples, {len(self.events)} event markers)")
        return save_path

    def get_calibration_summary(self) -> dict:
        event_counts = {}
        for evt in self.events:
            lbl = evt["event_label"]
            event_counts[lbl] = event_counts.get(lbl, 0) + 1

        return {
            "is_active": self.session_active,
            "session_id": self.session_id,
            "total_samples": len(self.raw_samples),
            "duration_sec": round(time.time() - self.start_time, 1) if self.start_time else 0.0,
            "event_counts": event_counts
        }
