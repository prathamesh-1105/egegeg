import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from scipy import signal

from src.channel_mapper import ChannelMapper, EXPECTED_GROUP_B_CHANNELS
from src.eeg_buffer import EEGBuffer
from src.lsl_eeg_source import LSLEEGSource
from src.step17b_pipeline import extract_fbcsp_features, FILTER_BANKS

# Hardware Safety States
STATE_DISCONNECTED = "DISCONNECTED"
STATE_SEARCHING = "SEARCHING"
STATE_CONNECTED = "CONNECTED"
STATE_BUFFERING = "BUFFERING"
STATE_READY = "READY"
STATE_SIGNAL_WARNING = "SIGNAL_WARNING"
STATE_INCOMPATIBLE = "INCOMPATIBLE"
STATE_ERROR = "ERROR"


def compute_signal_quality_metrics(data_2d: np.ndarray, channel_names: list, sfreq: float = 160.0) -> dict:
    """
    Computes per-channel signal quality metrics: RMS, variance, peak-to-peak amplitude,
    min, max, NaN count, and flatline detection.
    Assigns channel quality status: GOOD, WARNING, or BAD.
    """
    n_channels, n_samples = data_2d.shape
    ch_metrics = []
    has_warning = False
    has_bad = False

    for i in range(n_channels):
        ch_name = channel_names[i] if i < len(channel_names) else f"Ch{i+1}"
        sig = data_2d[i]

        nan_cnt = int(np.isnan(sig).sum()) + int(np.isinf(sig).sum())
        sig_min = float(np.nanmin(sig)) if nan_cnt < len(sig) else 0.0
        sig_max = float(np.nanmax(sig)) if nan_cnt < len(sig) else 0.0
        p2p = float(sig_max - sig_min)
        var_val = float(np.nanvar(sig)) if nan_cnt < len(sig) else 0.0
        rms_val = float(np.sqrt(np.nanmean(sig**2))) if nan_cnt < len(sig) else 0.0

        is_flatline = (p2p < 1e-3) or (var_val < 1e-4)
        is_noisy = (p2p > 500.0) or (var_val > 10000.0)

        if nan_cnt > 0 or is_flatline:
            status = "BAD"
            has_bad = True
        elif is_noisy:
            status = "WARNING"
            has_warning = True
        else:
            status = "GOOD"

        ch_metrics.append({
            "channel": ch_name,
            "status": status,
            "rms_uv": round(rms_val, 2),
            "variance": round(var_val, 2),
            "peak_to_peak_uv": round(p2p, 2),
            "min_uv": round(sig_min, 2),
            "max_uv": round(sig_max, 2),
            "nan_count": nan_cnt,
            "flatline": is_flatline
        })

    if has_bad:
        overall_quality = "BAD"
    elif has_warning:
        overall_quality = "WARNING"
    else:
        overall_quality = "GOOD"

    return {
        "overall_quality": overall_quality,
        "channels": ch_metrics
    }


def resample_eeg_data(data_2d: np.ndarray, orig_sfreq: float, target_sfreq: float = 160.0) -> tuple[np.ndarray, str]:
    """
    Controlled polyphase resampling layer (e.g., 250 Hz -> 160 Hz) with anti-aliasing filter.
    Returns:
        tuple: (resampled_data_2d, status_str)
    """
    if abs(orig_sfreq - target_sfreq) < 1.0:
        return data_2d, "PASS (Target Rate Match)"

    if orig_sfreq <= 0 or data_2d.shape[1] == 0:
        return data_2d, "INCOMPATIBLE (Invalid Source Rate)"

    try:
        # Compute exact integer rational fraction for polyphase resampling
        from math import gcd
        up = int(target_sfreq)
        down = int(orig_sfreq)
        common = gcd(up, down)
        up = up // common
        down = down // common

        resampled = signal.resample_poly(data_2d, up, down, axis=-1)
        return resampled, f"RESAMPLED ({orig_sfreq:.0f} Hz -> {target_sfreq:.0f} Hz)"
    except Exception as e:
        return data_2d, f"INCOMPATIBLE (Resampling Error: {e})"


class RealTimeHardwarePipelineManager:
    """
    Manages physical EEG hardware streaming, LSL discovery, stream validation,
    controlled resampling, 480-sample rolling buffer, safety state machine,
    and Step 17B FBCSP real-time inference.
    """

    def __init__(self, subject_id: str = "S001", models_dir: str = "models"):
        self.subject_id = subject_id
        self.models_dir = models_dir

        self.lsl_source = LSLEEGSource()
        self.channel_mapper = ChannelMapper(target_channels=EXPECTED_GROUP_B_CHANNELS)
        self.buffer = EEGBuffer(n_channels=15, sfreq=160.0, window_sec=3.0, step_sec=1.5)

        self.state = STATE_DISCONNECTED
        self.status_message = "No physical EEG hardware connected"
        self.mapped_indices = []

        # Load Step 17B model artifacts
        self.online_dir = os.path.join(models_dir, "step18", "selected_model", subject_id)
        if not os.path.exists(self.online_dir):
            self.online_dir = os.path.join(models_dir, "step17b", "selected_model", subject_id)
        if not os.path.exists(self.online_dir):
            self.online_dir = os.path.join(models_dir, "step17b", "selected_model", "S001")

        self.model_path = os.path.join(self.online_dir, "model.pkl")
        self.scaler_path = os.path.join(self.online_dir, "scaler.pkl")
        self.csp_path = os.path.join(self.online_dir, "csp.pkl")
        self.selector_path = os.path.join(self.online_dir, "selector.pkl")

        self.model = joblib.load(self.model_path) if os.path.exists(self.model_path) else None
        self.scaler = joblib.load(self.scaler_path) if os.path.exists(self.scaler_path) else None
        self.csp_dict = joblib.load(self.csp_path) if os.path.exists(self.csp_path) else None
        self.selector = joblib.load(self.selector_path) if os.path.exists(self.selector_path) else None

    def scan_lsl_streams(self) -> list:
        self.state = STATE_SEARCHING
        streams = LSLEEGSource.discover_streams()
        if len(streams) == 0:
            self.state = STATE_DISCONNECTED
            self.status_message = "No active LSL EEG streams found on local network"
        return streams

    def connect_hardware_stream(self, stream_name: str = None) -> tuple[bool, str]:
        """
        Attempts to connect to an LSL stream and performs compatibility verification.
        """
        self.lsl_source = LSLEEGSource(stream_name=stream_name)
        if not self.lsl_source.connect():
            self.state = STATE_DISCONNECTED
            self.status_message = "Failed to connect to LSL EEG stream"
            return False, self.status_message

        # Verify Stream Compatibility
        avail_chans = self.lsl_source.get_channel_names()
        val_res = self.channel_mapper.validate_channels(avail_chans)

        if not val_res["is_compatible"]:
            self.state = STATE_INCOMPATIBLE
            missing_str = ", ".join(val_res["missing_channels"])
            self.status_message = f"INCOMPATIBLE: Required channels missing: {missing_str}"
            self.lsl_source.disconnect()
            return False, self.status_message

        self.mapped_indices = val_res["mapped_indices"]
        self.lsl_source.start()
        self.buffer.reset()
        self.state = STATE_BUFFERING
        self.status_message = "Connected to stream. Buffering 480 samples @ 160 Hz..."
        return True, self.status_message

    def disconnect_hardware_stream(self) -> None:
        self.lsl_source.disconnect()
        self.buffer.reset()
        self.state = STATE_DISCONNECTED
        self.status_message = "Hardware stream disconnected"

    def process_realtime_chunk(self) -> dict:
        """
        Pulls latest samples from LSL inlet, resamples if needed, maps channels,
        appends to EEGBuffer, and performs inference when buffer is full.
        """
        if self.state == STATE_DISCONNECTED or not self.lsl_source.is_connected():
            return {
                "state": STATE_DISCONNECTED,
                "status_message": "Hardware disconnected",
                "buffer_status": self.buffer.get_fill_status(),
                "can_predict": False
            }

        # Pull raw window from LSL source
        raw_window = self.lsl_source.read_window(window_sec=3.0)
        orig_srate = self.lsl_source.get_sampling_rate()

        # Resample to 160 Hz target if needed
        resampled_data, resamp_status = resample_eeg_data(raw_window, orig_sfreq=orig_srate, target_sfreq=160.0)

        # Map to 15 Group B model channels
        if len(self.mapped_indices) == 15 and resampled_data.shape[0] >= 15:
            mapped_data = resampled_data[self.mapped_indices, :]
        else:
            mapped_data = resampled_data[:15, :] if resampled_data.shape[0] >= 15 else resampled_data

        # Append new samples to rolling buffer
        self.buffer.append_samples(mapped_data)
        buf_status = self.buffer.get_fill_status()

        # Compute Signal Quality
        sq_info = compute_signal_quality_metrics(mapped_data, EXPECTED_GROUP_B_CHANNELS, sfreq=160.0)

        if not buf_status["is_ready"]:
            self.state = STATE_BUFFERING
            self.status_message = f"Buffering... ({buf_status['current_samples']}/480 samples)"
            return {
                "state": self.state,
                "status_message": self.status_message,
                "buffer_status": buf_status,
                "resample_status": resamp_status,
                "signal_quality": sq_info,
                "can_predict": False
            }

        if sq_info["overall_quality"] == "BAD":
            self.state = STATE_SIGNAL_WARNING
            self.status_message = "⚠️ SIGNAL QUALITY TOO LOW: Excessive noise or flatline detected"

        else:
            self.state = STATE_READY
            self.status_message = "READY: Live hardware inference active"

        # Execute Step 17B FBCSP Inference on 480-sample window
        t0 = time.perf_counter()
        X_win_480 = self.buffer.get_window() # (15, 480)
        X_input = np.expand_dims(X_win_480, axis=0)

        X_fbcsp, _ = extract_fbcsp_features(X_input, csp_dict=self.csp_dict, is_train=False)

        if self.selector is not None:
            X_sel = self.selector.transform(X_fbcsp)
        else:
            X_sel = X_fbcsp[:, :self.scaler.n_features_in_]

        X_sc = self.scaler.transform(X_sel)
        raw_pred = str(self.model.predict(X_sc)[0])
        proba_arr = self.model.predict_proba(X_sc)[0]
        t1 = time.perf_counter()

        comp_latency_ms = (t1 - t0) * 1000.0
        classes = list(self.model.classes_)
        prob_dict = {cls: float(p) for cls, p in zip(classes, proba_arr)}
        confidence = float(np.max(proba_arr))

        return {
            "state": self.state,
            "status_message": self.status_message,
            "buffer_status": buf_status,
            "resample_status": resamp_status,
            "signal_quality": sq_info,
            "can_predict": True,
            "raw_prediction": raw_pred,
            "confidence": confidence,
            "probabilities": prob_dict,
            "latency_ms": round(comp_latency_ms, 2),
            "eeg_window_sec": 3.0,
            "single_channel_uv": X_win_480[5] * 1e6 # Channel C3
        }
