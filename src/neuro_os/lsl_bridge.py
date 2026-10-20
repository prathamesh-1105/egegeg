"""
NeuroOS Real-Time EEG Source Abstraction & Robust LSL Bridge (Phase 2E)
Provides unified interface for:
1. Simulated PhysioNet EDF streaming (explicitly labeled as offline playback simulation)
2. 16-channel OpenBCI Cyton + Daisy (Direct 15-channel sensorimotor extraction, ZERO spatial interpolation)
3. 8-channel OpenBCI Cyton (3D spherical spline spatial reconstruction fallback)

HARDWARE SAFETY & CONTRACT INTEGRITY:
- Distinct connection states: DISCONNECTED, SEARCHING, CONNECTED_INVALID, CONNECTED_VALID.
- Incompatible montages (missing channels, wrong electrode locations) are strictly REJECTED, not silently fabricated.
- Resamples incoming non-160Hz streams (e.g. 125Hz nominal Cyton+Daisy or 250Hz Cyton 8ch) to exact 160.0 Hz.
- Buffer overflow protection and sample discontinuity tracking.
- Never generates fake live EEG when a device is disconnected.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import time
import os
import sys
from scipy import signal

from src.step33_pipeline import GROUP_B_CHANNELS
from src.step8_pipeline import find_subject_runs
from src.step14_pipeline import extract_event_aligned_windows
from src.step7_pipeline import load_and_preprocess_cached, LABEL_MAPPING
from src.neuro_os.channel_adapter import OpenBCIChannelAdapter
from src.neuro_os.hardware_profiles import (
    HardwareProfileRegistry,
    OPENBCI_16CH_PHYSICAL,
    OPENBCI_ULTRACORTEX_DEFAULT,
    STEP62_TARGET_CHANNELS,
    STEP60_TARGET_CHANNELS,
    validate_channel_montage,
    normalize_channel_name
)

# Try importing pylsl safely; handle missing dependency gracefully.
try:
    import pylsl
    HAS_PYLSL = True
except ImportError:
    HAS_PYLSL = False
    pylsl = None


class StreamState:
    DISCONNECTED = "DISCONNECTED"
    SEARCHING = "SEARCHING"
    CONNECTED_INVALID = "CONNECTED_INVALID"
    CONNECTED_VALID = "CONNECTED_VALID"


class EEGSource(ABC):
    """Abstract base class for all NeuroOS EEG input sources."""

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Returns metadata about the source (channels, rate, status, type)."""
        pass

    @abstractmethod
    def read_window(self) -> Optional[Dict[str, Any]]:
        """Pulls next window of EEG data (shape: 1 x channels x samples)."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if the source is actively connected and producing data."""
        pass


class DatasetEEGSource(EEGSource):
    """
    Simulated real-time EEG source streaming genuine PhysioNet EDF dataset windows.
    Guarantees scientific authenticity by replaying actual subject motor imagery trials.
    Explicitly labeled as simulation — not a live headset.
    """

    def __init__(
        self,
        data_dir: str = "data",
        default_subject: str = "S001",
        default_run: str = "R11",
        execution_mode: str = "RECORDED_TRIAL_REPLAY",
        stride_sec: float = 0.5
    ):
        self.data_dir = data_dir
        self.subject_id = default_subject
        self.run_id = default_run
        self.channels = GROUP_B_CHANNELS
        self.sampling_rate_hz = 160.0
        self.window_sec = 3.0
        self.stride_sec = float(stride_sec)
        self.samples_per_window = int(self.sampling_rate_hz * self.window_sec)  # 480
        self.current_trial_idx = 0
        self.continuous_idx = 0
        self.execution_mode = execution_mode
        self.is_simulated = True

        self._cache = {}
        self.continuous_windows: List[np.ndarray] = []
        self.continuous_labels: List[str] = []
        self._load_subject_data(self.subject_id, self.run_id)

    def _load_subject_data(self, subject_id: str, run_id: str):
        cache_key = f"{subject_id}_{run_id}"
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            self.X = cached_data["X"]
            self.y = cached_data["y"]
            self.info = cached_data["info"]
            self.continuous_windows = cached_data["continuous_windows"]
            self.continuous_labels = cached_data["continuous_labels"]
            self.subject_id = subject_id
            self.run_id = run_id
            self.current_trial_idx = 0
            self.continuous_idx = 0
            return

        runs_found = find_subject_runs(subject_id, data_dir=self.data_dir)
        edf_path = runs_found.get(run_id)
        if not edf_path or not os.path.exists(edf_path):
            raise FileNotFoundError(f"EDF file for {subject_id} {run_id} not found in {self.data_dir}.")

        # 1. Mode A: Discrete event-aligned trials (1 window per trial, typically 30 trials)
        X, y, info = extract_event_aligned_windows(edf_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
        self.X = X
        self.y = y
        self.info = info

        # 2. Mode B: Continuous sliding-window stream (stride_sec e.g. 0.5s -> ~244 overlapping windows)
        continuous_windows = []
        continuous_labels = []
        try:
            raw_sub, df_events = load_and_preprocess_cached(edf_path, "Group B")
            raw_data = raw_sub.get_data()  # shape (15, n_samples)
            sfreq = raw_sub.info['sfreq']
            win_samples = int(self.window_sec * sfreq)  # 480
            stride_samples = max(1, int(self.stride_sec * sfreq))  # 80 for 0.5s
            n_total = raw_data.shape[1]

            for start in range(0, n_total - win_samples + 1, stride_samples):
                end = start + win_samples
                win = raw_data[:, start:end]
                t_mid = (start + win_samples / 2.0) / sfreq
                label = "REST"
                if not df_events.empty:
                    for _, ev in df_events.iterrows():
                        if ev['onset_sec'] <= t_mid <= ev['end_sec']:
                            label = LABEL_MAPPING.get(ev['event_code'], "REST")
                            break
                continuous_windows.append(win[np.newaxis, :, :])
                continuous_labels.append(label)
        except Exception:
            # Fallback to discrete trials if continuous raw extraction fails
            continuous_windows = [X[i:i+1] for i in range(len(X))]
            continuous_labels = list(y)

        self.continuous_windows = continuous_windows
        self.continuous_labels = continuous_labels
        self.subject_id = subject_id
        self.run_id = run_id
        self.current_trial_idx = 0
        self.continuous_idx = 0

        self._cache[cache_key] = {
            "X": X,
            "y": y,
            "info": info,
            "continuous_windows": continuous_windows,
            "continuous_labels": continuous_labels
        }

    def set_subject_run(self, subject_id: str, run_id: str):
        """Switches active dataset subject/run stream."""
        if self.subject_id != subject_id or self.run_id != run_id:
            self._load_subject_data(subject_id, run_id)

    def set_execution_mode(self, mode: str, stride_sec: float = 0.5):
        """Switches between Mode A (RECORDED_TRIAL_REPLAY) and Mode B (CONTINUOUS_STREAM_SIMULATION)."""
        clean_mode = str(mode).upper()
        if "CONTINUOUS" in clean_mode or clean_mode in ["MODE_B", "STREAM"]:
            self.execution_mode = "CONTINUOUS_STREAM_SIMULATION"
        else:
            self.execution_mode = "RECORDED_TRIAL_REPLAY"
        self.stride_sec = float(stride_sec)
        self.current_trial_idx = 0
        self.continuous_idx = 0

    def get_info(self) -> Dict[str, Any]:
        mode_label = "OFFLINE REPLAY" if self.execution_mode == "RECORDED_TRIAL_REPLAY" else "SIMULATION (CONTINUOUS STREAM)"
        total_items = len(self.continuous_windows) if self.execution_mode == "CONTINUOUS_STREAM_SIMULATION" else (len(self.y) if hasattr(self, 'y') else 0)
        curr_idx = self.continuous_idx if self.execution_mode == "CONTINUOUS_STREAM_SIMULATION" else self.current_trial_idx

        return {
            "source_type": "SIMULATED_EDF",
            "name": f"PhysioNet EDF Simulation ({self.subject_id} {self.run_id})",
            "execution_mode": self.execution_mode,
            "mode_badge": mode_label,
            "connected": True,
            "status": "STREAMING",
            "stream_state": "SIMULATED",
            "is_simulated": True,
            "simulation_notice": "GENUINE_EDF_PLAYBACK_SIMULATION: Replaying recorded PhysioNet EEGMMIDB dataset. Not live hardware.",
            "hardware_mode": "SIMULATED_PHYSIONET",
            "hardware_profile_id": "SIMULATED_PHYSIONET",
            "channel_count": len(self.channels),
            "channel_names": self.channels,
            "sampling_rate_hz": self.sampling_rate_hz,
            "stride_sec": self.stride_sec if self.execution_mode == "CONTINUOUS_STREAM_SIMULATION" else 3.0,
            "samples_per_window": self.samples_per_window,
            "total_trials": total_items,
            "current_trial_idx": curr_idx,
            "hardware_ready": False,  # Never claim physical hardware ready for simulation
            "readiness_level": "SOFTWARE_READY",
            "adapter_telemetry": {
                "hardware_mode": "SIMULATED_PHYSIONET",
                "execution_mode": self.execution_mode,
                "input_channel_count": 15,
                "output_channel_count": 15,
                "interpolation_active": False,
                "adapter_status": "BYPASS_GENUINE_15CH_DATASET"
            },
            "error_detail": None
        }

    def is_connected(self) -> bool:
        return True

    def read_window(self) -> Optional[Dict[str, Any]]:
        """Fetches current window according to active execution mode (Mode A or Mode B)."""
        if self.execution_mode == "CONTINUOUS_STREAM_SIMULATION":
            if not self.continuous_windows:
                return None
            idx = self.continuous_idx
            win_data = self.continuous_windows[idx]
            true_label = self.continuous_labels[idx]
            self.continuous_idx = (self.continuous_idx + 1) % len(self.continuous_windows)
            active_idx = idx
            total_items = len(self.continuous_windows)
            stride = self.stride_sec
        else:
            if not hasattr(self, 'X') or len(self.X) == 0:
                return None
            trial_idx = self.current_trial_idx
            win_data = self.X[trial_idx:trial_idx+1]  # shape (1, 15, 480)
            true_label = self.y[trial_idx]
            self.current_trial_idx = (self.current_trial_idx + 1) % len(self.X)
            active_idx = trial_idx
            total_items = len(self.y)
            stride = 3.0

        return {
            "source_type": "SIMULATED_EDF",
            "stream_state": "SIMULATED",
            "execution_mode": self.execution_mode,
            "is_simulated": True,
            "subject_id": self.subject_id,
            "run_id": self.run_id,
            "trial_idx": active_idx,
            "total_trials": total_items,
            "stride_sec": stride,
            "true_label": true_label,
            "window_data": win_data,
            "channels": self.channels,
            "sampling_rate_hz": self.sampling_rate_hz,
            "timestamp": time.time(),
            "status": "STREAMING",
            "adapter_telemetry": {
                "hardware_mode": "SIMULATED_PHYSIONET",
                "execution_mode": self.execution_mode,
                "input_channel_count": 15,
                "output_channel_count": 15,
                "interpolation_active": False,
                "adapter_status": "BYPASS_GENUINE_15CH_DATASET"
            }
        }


class LSLEEGSource(EEGSource):
    """
    Robust Lab Streaming Layer (LSL) Hardware EEG Bridge (Phase 2E).
    Supports:
    1. 16-Channel OpenBCI Cyton + Daisy (Direct 15-channel sensorimotor extraction, ZERO spatial interpolation)
    2. 8-Channel OpenBCI Cyton (3D spherical spline spatial reconstruction fallback)
    3. 15-Channel Native Model Stream (Direct pass-through)

    Features:
    - 4 distinct states: DISCONNECTED, SEARCHING, CONNECTED_INVALID, CONNECTED_VALID.
    - Stream discovery with configurable timeout and graceful handling when pylsl is missing.
    - Strict channel montage validation (Ultracortex Mark IV default cap strictly REJECTED if missing motor grid).
    - Polyphase / sinc anti-aliased resampling from 125.0 Hz or 250.0 Hz to exact 160.0 Hz.
    - Buffer management, missing-sample discontinuity tracking, and freshness protection.
    - Clean disconnect() and reconnect() methods.
    - Never generates fake EEG data when disconnected.
    """

    def __init__(
        self,
        stream_name_keyword: str = "EEG",
        timeout_sec: float = 0.8,
        target_sfreq: float = 160.0,
        target_samples: int = 480,
        mock_inlet: Optional[Any] = None
    ):
        self.stream_name_keyword = stream_name_keyword
        self.timeout_sec = timeout_sec
        self.target_sfreq = target_sfreq
        self.target_samples = target_samples

        self.inlet = mock_inlet
        self.stream_info = None
        self.state = StreamState.DISCONNECTED
        self._error_msg: Optional[str] = None

        self.channel_count = 0
        self.channel_names: List[str] = []
        self.nominal_sfreq = 0.0
        self.measured_sfreq = 0.0

        # Montage validation results
        self.reorder_indices: List[int] = []
        self.missing_channels: List[str] = []
        self.extra_channels: List[str] = []
        self.hardware_profile_id: str = "OPENBCI_CYTON_DAISY_16CH"

        # Buffer & discontinuity tracking
        self.buffer: List[List[float]] = []  # Ring buffer of resampled samples (N, channels)
        self.max_buffer_samples = self.target_samples * 4  # 1920 samples max (~12s) to prevent memory growth
        self.last_chunk_timestamp: Optional[float] = None
        self.last_sample_time: float = 0.0
        self.discontinuity_count: int = 0
        self.dropped_samples_count: int = 0
        self.total_samples_received: int = 0

        # 8-channel adapter for fallback
        self.adapter = OpenBCIChannelAdapter(target_sfreq=self.target_sfreq, target_samples=self.target_samples)

        if mock_inlet is not None:
            self._setup_inlet_metadata(mock_inlet)
        else:
            self.attempt_connection()

    def disconnect(self):
        """Explicitly disconnects stream, clears buffers, and resets state."""
        if self.inlet is not None:
            try:
                if hasattr(self.inlet, "close_stream"):
                    self.inlet.close_stream()
            except Exception:
                pass
            self.inlet = None

        self.state = StreamState.DISCONNECTED
        self.buffer.clear()
        self.last_chunk_timestamp = None
        self._error_msg = "Device disconnected by user."

    def reconnect(self) -> bool:
        """Resets existing connection and re-attempts stream discovery."""
        self.disconnect()
        return self.attempt_connection()

    def _setup_inlet_metadata(self, inlet: Any):
        """Extracts stream metadata, sampling rate, and validates electrode montage."""
        try:
            self.stream_info = inlet.info()
            self.channel_count = self.stream_info.channel_count()
            self.nominal_sfreq = float(self.stream_info.nominal_srate())
            self.measured_sfreq = self.nominal_sfreq

            # 1. Parse channel labels from LSL XML metadata if available
            ch_names = []
            try:
                ch_xml = self.stream_info.desc().child("channels").child("channel")
                for _ in range(self.channel_count):
                    val = ch_xml.child_value("label")
                    if val:
                        ch_names.append(val)
                    ch_xml = ch_xml.next_sibling()
            except Exception:
                pass

            # If no XML metadata present, check if already configured or assign standard physical defaults
            if not ch_names or len(ch_names) != self.channel_count:
                if self.channel_names and len(self.channel_names) == self.channel_count:
                    ch_names = list(self.channel_names)
                elif self.channel_count == 16:
                    ch_names = list(OPENBCI_16CH_PHYSICAL)
                elif self.channel_count == 8:
                    ch_names = list(OpenBCIChannelAdapter.OPENBCI_8_CHANNELS)
                elif self.channel_count == 15:
                    ch_names = list(STEP62_TARGET_CHANNELS)
                else:
                    ch_names = [f"Ch_{i+1}" for i in range(self.channel_count)]

            self.channel_names = ch_names

            # 2. Montage Validation against Model Requirements
            if self.channel_count == 16:
                self.hardware_profile_id = "OPENBCI_CYTON_DAISY_16CH"
                # Check for standard Ultracortex Mark IV default cap layout (lacks frontocentral motor grid)
                norm_detected = [normalize_channel_name(c) for c in self.channel_names]
                norm_ultracortex = [normalize_channel_name(c) for c in OPENBCI_ULTRACORTEX_DEFAULT]

                # If detected montage matches Ultracortex default cap, reject:
                if set(norm_detected) == set(norm_ultracortex):
                    self.state = StreamState.CONNECTED_INVALID
                    self.missing_channels = [c for c in STEP62_TARGET_CHANNELS if normalize_channel_name(c) not in norm_detected]
                    self.extra_channels = [c for c in self.channel_names if normalize_channel_name(c) not in [normalize_channel_name(x) for x in STEP62_TARGET_CHANNELS]]
                    self._error_msg = (
                        "Montage Rejected: Detected Ultracortex Mark IV default cap montage. "
                        "Ultracortex default cap lacks required frontocentral and centroparietal motor grid positions "
                        f"({self.missing_channels}). Custom 10-20 motor strip placement required."
                    )
                    return

                is_valid, reorder_idx, missing, extra = validate_channel_montage(self.channel_names, STEP62_TARGET_CHANNELS)
                self.missing_channels = missing
                self.extra_channels = extra

                if is_valid:
                    self.state = StreamState.CONNECTED_VALID
                    self.reorder_indices = reorder_idx
                    self._error_msg = None
                else:
                    self.state = StreamState.CONNECTED_INVALID
                    self.reorder_indices = []
                    self._error_msg = (
                        f"Montage Rejected: 16-channel stream missing required motor channels: {missing}. "
                        "Step 62 model requires exact 15-channel sensorimotor grid."
                    )

            elif self.channel_count == 8:
                self.hardware_profile_id = "OPENBCI_CYTON_8CH"
                norm_detected = [normalize_channel_name(c) for c in self.channel_names]
                norm_8ch = [normalize_channel_name(c) for c in OpenBCIChannelAdapter.OPENBCI_8_CHANNELS]
                missing = [c for c, nc in zip(OpenBCIChannelAdapter.OPENBCI_8_CHANNELS, norm_8ch) if nc not in norm_detected]

                if not missing:
                    self.state = StreamState.CONNECTED_VALID
                    self.reorder_indices = [norm_detected.index(nc) for nc in norm_8ch]
                    self._error_msg = None
                else:
                    self.state = StreamState.CONNECTED_INVALID
                    self.missing_channels = missing
                    self._error_msg = f"Montage Rejected: 8-channel stream missing OpenBCI standard electrodes: {missing}."

            elif self.channel_count == 15:
                self.hardware_profile_id = "SIMULATED_PHYSIONET"
                is_valid, reorder_idx, missing, extra = validate_channel_montage(self.channel_names, STEP62_TARGET_CHANNELS)
                if is_valid:
                    self.state = StreamState.CONNECTED_VALID
                    self.reorder_indices = reorder_idx
                    self._error_msg = None
                else:
                    self.state = StreamState.CONNECTED_INVALID
                    self.missing_channels = missing
                    self._error_msg = f"Montage Rejected: 15-channel stream missing model channels: {missing}."

            else:
                self.state = StreamState.CONNECTED_INVALID
                self.hardware_profile_id = "UNSUPPORTED"
                self._error_msg = (
                    f"Unsupported LSL channel count: {self.channel_count}. "
                    "Expected 16-channel Cyton+Daisy (motor strip), 8-channel Cyton, or 15-channel native stream."
                )

        except Exception as e:
            self.state = StreamState.CONNECTED_INVALID
            self._error_msg = f"Failed to initialize LSL stream metadata: {str(e)}"

    def attempt_connection(self) -> bool:
        """Resolves LSL streams on local network with explicit timeout."""
        if not HAS_PYLSL:
            self.state = StreamState.DISCONNECTED
            self._error_msg = "pylsl python package is not installed. Install pylsl to stream live LSL hardware."
            return False

        self.state = StreamState.SEARCHING
        try:
            streams = pylsl.resolve_byprop('type', self.stream_name_keyword, timeout=self.timeout_sec)
            if not streams:
                streams = pylsl.resolve_streams(wait_time=self.timeout_sec)

            if not streams:
                self.state = StreamState.DISCONNECTED
                self._error_msg = f"No active LSL stream found (timeout={self.timeout_sec}s). Ensure OpenBCI GUI / LSL publisher is running."
                return False

            stream = streams[0]
            self.inlet = pylsl.StreamInlet(stream)
            self._setup_inlet_metadata(self.inlet)
            return self.state == StreamState.CONNECTED_VALID

        except Exception as e:
            self.state = StreamState.DISCONNECTED
            self._error_msg = f"LSL connection error: {str(e)}"
            return False

    def is_connected(self) -> bool:
        """Returns True if an LSL inlet is actively attached (regardless of valid/invalid montage)."""
        return self.state in [StreamState.CONNECTED_VALID, StreamState.CONNECTED_INVALID]

    def is_valid(self) -> bool:
        """Returns True if stream is actively connected AND passes full montage contract."""
        return self.state == StreamState.CONNECTED_VALID

    def _resample_chunk(self, chunk: np.ndarray, in_rate: float, out_rate: float = 160.0) -> np.ndarray:
        """
        Resamples chunk of shape (n_samples, n_channels) from in_rate to out_rate.
        Uses scipy.signal.resample along sample axis for anti-aliased reconstruction.
        """
        if abs(in_rate - out_rate) < 0.5 or chunk.shape[0] < 2:
            return chunk

        n_samples_in, n_channels = chunk.shape
        n_samples_out = int(round(n_samples_in * (out_rate / in_rate)))
        if n_samples_out < 1:
            return np.empty((0, n_channels))

        resampled = signal.resample(chunk, n_samples_out, axis=0)
        return resampled

    def get_info(self) -> Dict[str, Any]:
        """Provides comprehensive hardware diagnostics for UI and telemetry."""
        is_conn = self.is_connected()
        is_val = self.is_valid()

        # Determine readiness level
        if is_val:
            readiness = "READY_FOR_BENCH_TEST"
        elif is_conn:
            readiness = "HARDWARE_INVALID_CONFIG"
        else:
            readiness = "SOFTWARE_READY"

        stream_name = "None"
        if self.inlet and self.stream_info:
            try:
                stream_name = self.stream_info.name()
            except Exception:
                stream_name = "LSL_Stream"

        freshness_sec = round(time.time() - self.last_sample_time, 2) if self.last_sample_time > 0 else None

        return {
            "source_type": "LSL_HARDWARE",
            "name": f"LSL Hardware Stream ({stream_name})",
            "connected": is_conn,
            "status": self.state,
            "stream_state": self.state,
            "is_simulated": False,
            "hardware_mode": self.hardware_profile_id,
            "hardware_profile_id": self.hardware_profile_id,
            "readiness_level": readiness,
            "hardware_ready": is_val,  # Only True if physically connected AND valid configuration
            "channel_count": self.channel_count if is_conn else 0,
            "detected_channels": self.channel_names if is_conn else [],
            "expected_channels": STEP62_TARGET_CHANNELS,
            "missing_channels": self.missing_channels,
            "extra_channels": self.extra_channels,
            "nominal_sampling_rate_hz": self.nominal_sfreq if is_conn else 0.0,
            "measured_sampling_rate_hz": round(self.measured_sfreq, 1) if is_conn else 0.0,
            "target_sampling_rate_hz": self.target_sfreq,
            "sample_freshness_sec": freshness_sec,
            "samples_in_buffer": len(self.buffer),
            "discontinuity_count": self.discontinuity_count,
            "dropped_samples_count": self.dropped_samples_count,
            "total_samples_received": self.total_samples_received,
            "adapter_telemetry": {
                "hardware_mode": self.hardware_profile_id,
                "input_channel_count": self.channel_count if is_conn else 0,
                "output_channel_count": 15 if is_val else 0,
                "input_channel_names": self.channel_names if is_conn else [],
                "output_channel_names": STEP62_TARGET_CHANNELS if is_val else [],
                "interpolation_active": (self.channel_count == 8),
                "adapter_status": "VALID_MONTAGE" if is_val else (self.state),
                "scientific_notice": (
                    "16 physical channels captured. 15 sensorimotor channels selected directly with ZERO spatial interpolation."
                    if self.channel_count == 16 and is_val else (
                        "Reconstructed channels are 3D spherical spline spatial estimates, not physical electrode measurements."
                        if self.channel_count == 8 and is_val else None
                    )
                )
            },
            "error_detail": self._error_msg
        }

    def read_window(self) -> Optional[Dict[str, Any]]:
        """
        Pulls hardware chunk from LSL inlet, applies rate resampling, and buffers.
        Never generates fake live data when disconnected or invalid.
        """
        # If not connected or invalid configuration, reject immediately
        if not self.is_connected() or self.inlet is None:
            return {
                "source_type": "LSL_HARDWARE",
                "stream_state": StreamState.DISCONNECTED,
                "status": "DISCONNECTED",
                "is_simulated": False,
                "error_detail": self._error_msg or "LSL hardware stream disconnected.",
                "window_data": None,
                "timestamp": time.time()
            }

        if self.state == StreamState.CONNECTED_INVALID:
            return {
                "source_type": "LSL_HARDWARE",
                "stream_state": StreamState.CONNECTED_INVALID,
                "status": "CONNECTED_INVALID",
                "is_simulated": False,
                "error_detail": self._error_msg or "LSL stream has invalid montage or channel count.",
                "missing_channels": self.missing_channels,
                "window_data": None,
                "timestamp": time.time()
            }

        try:
            # Pull available chunk from inlet
            samples, timestamps = self.inlet.pull_chunk(timeout=0.05, max_samples=480)

            if samples and len(samples) > 0:
                now = time.time()
                self.last_sample_time = now
                self.total_samples_received += len(samples)

                # Check timestamp discontinuity (> 2 sample periods missing)
                if timestamps and len(timestamps) > 1 and self.nominal_sfreq > 0:
                    nominal_dt = 1.0 / self.nominal_sfreq
                    if self.last_chunk_timestamp is not None:
                        dt = timestamps[0] - self.last_chunk_timestamp
                        if dt > 2.5 * nominal_dt:
                            self.discontinuity_count += 1
                    self.last_chunk_timestamp = timestamps[-1]

                    # Measure dynamic rate
                    if len(timestamps) >= 10:
                        measured_dt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
                        if measured_dt > 0:
                            self.measured_sfreq = 0.9 * self.measured_sfreq + 0.1 * (1.0 / measured_dt)

                chunk_np = np.array(samples, dtype=np.float32)

                # Apply anti-alias resampling to 160 Hz if stream is non-160 Hz (e.g. 125 Hz Cyton+Daisy)
                effective_sfreq = self.nominal_sfreq if self.nominal_sfreq > 0 else 160.0
                if abs(effective_sfreq - self.target_sfreq) > 1.0:
                    resampled_chunk = self._resample_chunk(chunk_np, in_rate=effective_sfreq, out_rate=self.target_sfreq)
                else:
                    resampled_chunk = chunk_np

                for s in resampled_chunk:
                    self.buffer.append(s.tolist())

                # Buffer freshness protection (drop oldest if buffer exceeds maximum)
                if len(self.buffer) > self.max_buffer_samples:
                    overflow = len(self.buffer) - self.max_buffer_samples
                    self.dropped_samples_count += overflow
                    self.buffer = self.buffer[overflow:]

            # If enough samples accumulated for 1 full 3.0s window (480 samples)
            if len(self.buffer) >= self.target_samples:
                win_samples = np.array(self.buffer[:self.target_samples], dtype=np.float32).T  # (channels, 480)
                self.buffer = self.buffer[self.target_samples:]

                # Case A: 16-Channel OpenBCI Cyton + Daisy -> Direct 15 Physical Channel Selection
                if self.channel_count == 16:
                    win_15ch = win_samples[self.reorder_indices, :self.target_samples]  # shape (15, 480)
                    win_data = np.expand_dims(win_15ch, axis=0)  # shape (1, 15, 480)

                    return {
                        "source_type": "LSL_HARDWARE",
                        "stream_state": StreamState.CONNECTED_VALID,
                        "window_data": win_data,
                        "channels": STEP62_TARGET_CHANNELS,
                        "sampling_rate_hz": self.target_sfreq,
                        "nominal_sfreq": self.nominal_sfreq,
                        "measured_sfreq": self.measured_sfreq,
                        "timestamp": time.time(),
                        "status": "STREAMING",
                        "discontinuity_count": self.discontinuity_count,
                        "dropped_samples_count": self.dropped_samples_count,
                        "is_simulated": False,
                        "adapter_telemetry": {
                            "hardware_mode": "16CH_PHYSICAL",
                            "input_channel_count": 16,
                            "output_channel_count": 15,
                            "input_channel_names": self.channel_names,
                            "output_channel_names": STEP62_TARGET_CHANNELS,
                            "unused_channels": self.extra_channels,
                            "interpolation_active": False,
                            "adapter_status": "16CH_PHYSICAL_SELECTION",
                            "scientific_notice": "16 physical channels captured. 15 sensorimotor channels selected directly with ZERO spatial interpolation."
                        }
                    }

                # Case B: 8-Channel OpenBCI Cyton -> Apply 8->15 Spatial Adapter
                elif self.channel_count == 8:
                    reordered_8ch = win_samples[self.reorder_indices, :self.target_samples]
                    trans_res = self.adapter.transform(reordered_8ch, OpenBCIChannelAdapter.OPENBCI_8_CHANNELS, self.target_sfreq)
                    if not trans_res["success"]:
                        return {
                            "source_type": "LSL_HARDWARE",
                            "stream_state": StreamState.CONNECTED_INVALID,
                            "status": "ADAPTER_REJECTED",
                            "is_simulated": False,
                            "error_detail": trans_res["telemetry"].get("error_detail"),
                            "window_data": None,
                            "adapter_telemetry": {
                                "hardware_mode": "8CH_INTERPOLATED",
                                **trans_res["telemetry"]
                            },
                            "timestamp": time.time()
                        }

                    return {
                        "source_type": "LSL_HARDWARE",
                        "stream_state": StreamState.CONNECTED_VALID,
                        "window_data": trans_res["data_15ch"],
                        "channels": trans_res["channels"],
                        "sampling_rate_hz": self.target_sfreq,
                        "nominal_sfreq": self.nominal_sfreq,
                        "measured_sfreq": self.measured_sfreq,
                        "timestamp": time.time(),
                        "status": "STREAMING",
                        "discontinuity_count": self.discontinuity_count,
                        "dropped_samples_count": self.dropped_samples_count,
                        "is_simulated": False,
                        "adapter_telemetry": {
                            "hardware_mode": "8CH_INTERPOLATED",
                            **trans_res["telemetry"]
                        }
                    }

                # Case C: 15-Channel Native Stream -> Bypass adapter with reordering
                elif self.channel_count == 15:
                    win_15ch = win_samples[self.reorder_indices, :self.target_samples]
                    win_data = np.expand_dims(win_15ch, axis=0)  # shape (1, 15, 480)
                    return {
                        "source_type": "LSL_HARDWARE",
                        "stream_state": StreamState.CONNECTED_VALID,
                        "window_data": win_data,
                        "channels": STEP62_TARGET_CHANNELS,
                        "sampling_rate_hz": self.target_sfreq,
                        "nominal_sfreq": self.nominal_sfreq,
                        "measured_sfreq": self.measured_sfreq,
                        "timestamp": time.time(),
                        "status": "STREAMING",
                        "discontinuity_count": self.discontinuity_count,
                        "dropped_samples_count": self.dropped_samples_count,
                        "is_simulated": False,
                        "adapter_telemetry": {
                            "hardware_mode": "15CH_NATIVE",
                            "input_channel_count": 15,
                            "output_channel_count": 15,
                            "interpolation_active": False,
                            "adapter_status": "BYPASS_NATIVE_15CH"
                        }
                    }

            return {
                "source_type": "LSL_HARDWARE",
                "stream_state": self.state,
                "status": "BUFFERING",
                "is_simulated": False,
                "samples_in_buffer": len(self.buffer),
                "window_data": None,
                "timestamp": time.time()
            }

        except Exception as e:
            self.state = StreamState.DISCONNECTED
            self._error_msg = f"Error processing LSL stream: {str(e)}"
            return {
                "source_type": "LSL_HARDWARE",
                "stream_state": StreamState.DISCONNECTED,
                "status": "ERROR",
                "is_simulated": False,
                "error_detail": self._error_msg,
                "window_data": None,
                "timestamp": time.time()
            }
