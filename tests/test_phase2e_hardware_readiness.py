"""
NeuroOS Phase 2E — Hardware Readiness, LSL Stream Integration & Regression Suite
Tests hardware contract validation, mock LSL streams, sampling rate resampling,
signal quality gates, calibration state machine, and multi-command profile regression.
"""

import pytest
import numpy as np
import time
import os
import json

from src.neuro_os.hardware_profiles import (
    STEP62_TARGET_CHANNELS,
    OPENBCI_16CH_PHYSICAL,
    OPENBCI_ULTRACORTEX_DEFAULT,
    validate_channel_montage,
    normalize_channel_name,
    HardwareProfileRegistry
)
from src.neuro_os.lsl_bridge import (
    StreamState,
    DatasetEEGSource,
    LSLEEGSource
)
from src.neuro_os.signal_quality import NeuroOSSignalQualityGate
from src.neuro_os.calibration_manager import NeuroOSCalibrationManager, CalibrationState
from src.neuro_os.session_manager import NeuroOSSessionManager, SessionState


# --------------------------------------------------------------------------
# Mock LSL Stream Helpers (Software Testing Only — Not Real Hardware)
# --------------------------------------------------------------------------
class MockChannelNode:
    def __init__(self, labels, index=0):
        self._labels = labels
        self._index = index

    def child_value(self, name):
        if name == "label" and self._index < len(self._labels):
            return self._labels[self._index]
        return None

    def next_sibling(self):
        if self._index + 1 < len(self._labels):
            return MockChannelNode(self._labels, self._index + 1)
        return None


class MockChannelsNode:
    def __init__(self, labels):
        self._labels = labels

    def child(self, name):
        if name == "channel" and self._labels:
            return MockChannelNode(self._labels, 0)
        return MockXMLNode()


class MockDescNode:
    def __init__(self, labels):
        self._labels = labels

    def child(self, name):
        if name == "channels":
            return MockChannelsNode(self._labels)
        return MockXMLNode()


class MockStreamInfo:
    def __init__(self, name: str, channel_count: int, sfreq: float, channel_names=None):
        self._name = name
        self._channel_count = channel_count
        self._sfreq = sfreq
        self._channel_names = channel_names or []

    def name(self):
        return self._name

    def channel_count(self):
        return self._channel_count

    def nominal_srate(self):
        return self._sfreq

    def desc(self):
        return MockDescNode(self._channel_names)


class MockInlet:
    """Mock LSL Inlet simulating hardware stream packet generation."""

    def __init__(self, stream_info: MockStreamInfo, chunk_size: int = 32, noise_amplitude_uv: float = 20.0):
        self._info = stream_info
        self.chunk_size = chunk_size
        self.noise_amplitude_uv = noise_amplitude_uv
        self.closed = False
        self.current_time = time.time()
        self.call_count = 0

    def info(self):
        return self._info

    def close_stream(self):
        self.closed = True

    def pull_chunk(self, timeout: float = 0.05, max_samples: int = 480):
        if self.closed:
            return [], []

        self.call_count += 1
        n_channels = self._info.channel_count()
        n_samples = min(self.chunk_size, max_samples)

        # Generate realistic EEG microvolt samples (mean 0, std ~20 uV)
        samples = (np.random.randn(n_samples, n_channels) * self.noise_amplitude_uv).tolist()

        # Generate linear timestamps based on sampling rate
        sfreq = self._info.nominal_srate() if self._info.nominal_srate() > 0 else 160.0
        dt = 1.0 / sfreq
        timestamps = [self.current_time + i * dt for i in range(n_samples)]
        self.current_time = timestamps[-1] + dt

        return samples, timestamps


# --------------------------------------------------------------------------
# Step 1: Hardware Contract & Montage Audit Tests
# --------------------------------------------------------------------------
def test_step62_target_channel_contract():
    """Verifies that Step 62 channel contract requires exact 15 sensorimotor electrodes."""
    assert len(STEP62_TARGET_CHANNELS) == 15
    expected = [
        "FC3", "FC1", "FCz", "FC2", "FC4",
        "C3", "C1", "Cz", "C2", "C4",
        "CP3", "CP1", "CPz", "CP2", "CP4"
    ]
    assert STEP62_TARGET_CHANNELS == expected


def test_montage_validation_exact_15_channels():
    """Tests that a stream containing exact 15 channels in standard order is accepted."""
    is_valid, reorder_idx, missing, extra = validate_channel_montage(
        detected_channels=STEP62_TARGET_CHANNELS,
        expected_channels=STEP62_TARGET_CHANNELS
    )
    assert is_valid is True
    assert missing == []
    assert extra == []
    assert reorder_idx == list(range(15))


def test_montage_validation_shuffled_15_channels():
    """Tests deterministic reordering when channel names are permuted."""
    shuffled = list(reversed(STEP62_TARGET_CHANNELS))
    is_valid, reorder_idx, missing, extra = validate_channel_montage(
        detected_channels=shuffled,
        expected_channels=STEP62_TARGET_CHANNELS
    )
    assert is_valid is True
    assert missing == []
    # Reordered indices should map back to original STEP62 order
    reordered_channels = [shuffled[idx] for idx in reorder_idx]
    assert reordered_channels == STEP62_TARGET_CHANNELS


def test_montage_validation_16ch_cyton_daisy():
    """Tests OpenBCI Cyton + Daisy 16-channel physical montage (15 motor + 1 auxiliary O1)."""
    is_valid, reorder_idx, missing, extra = validate_channel_montage(
        detected_channels=OPENBCI_16CH_PHYSICAL,
        expected_channels=STEP62_TARGET_CHANNELS
    )
    assert is_valid is True
    assert missing == []
    assert extra == ["O1"]
    assert len(reorder_idx) == 15


def test_montage_validation_ultracortex_default_rejected():
    """
    CRITICAL HARDWARE TEST:
    Verifies that the default Ultracortex Mark IV headset cap montage is strictly REJECTED
    because it lacks the required frontocentral and centroparietal motor grid positions.
    """
    is_valid, reorder_idx, missing, extra = validate_channel_montage(
        detected_channels=OPENBCI_ULTRACORTEX_DEFAULT,
        expected_channels=STEP62_TARGET_CHANNELS
    )
    assert is_valid is False
    assert len(missing) > 0
    # Must report missing frontocentral electrodes
    assert "FC3" in missing
    assert "FC1" in missing
    assert "FCz" in missing
    assert "CP3" in missing


def test_montage_validation_missing_channels_rejected():
    """Verifies that missing channels are strictly rejected instead of fabricated."""
    detected = ["C3", "Cz", "C4", "O1", "O2"]
    is_valid, reorder_idx, missing, extra = validate_channel_montage(
        detected_channels=detected,
        expected_channels=STEP62_TARGET_CHANNELS
    )
    assert is_valid is False
    assert "FC3" in missing
    assert "CP4" in missing
    assert reorder_idx == []


# --------------------------------------------------------------------------
# Step 2: Robust LSL Bridge & Stream Resampling Tests
# --------------------------------------------------------------------------
def test_lsl_source_disconnected_when_no_stream():
    """Tests that LSLEEGSource initializes cleanly into DISCONNECTED state without crashing."""
    source = LSLEEGSource(stream_name_keyword="NONEXISTENT_STREAM", timeout_sec=0.1)
    assert source.is_connected() is False
    assert source.is_valid() is False
    assert source.state == StreamState.DISCONNECTED

    win = source.read_window()
    assert win is not None
    assert win["window_data"] is None  # Never fake live EEG!
    assert win["source_type"] == "LSL_HARDWARE"
    assert win["status"] == "DISCONNECTED"


def test_lsl_source_with_mock_16ch_valid():
    """Tests mock 16-channel OpenBCI Cyton + Daisy inlet streaming at 160.0 Hz."""
    info = MockStreamInfo("OpenBCI_Cyton_Daisy_16Ch", 16, 160.0)
    inlet = MockInlet(info, chunk_size=120)
    source = LSLEEGSource(mock_inlet=inlet, target_sfreq=160.0, target_samples=480)

    assert source.is_connected() is True
    assert source.is_valid() is True
    assert source.state == StreamState.CONNECTED_VALID
    assert source.channel_count == 16
    assert source.hardware_profile_id == "OPENBCI_CYTON_DAISY_16CH"

    # Pull chunks until a full 480-sample window is assembled
    win = None
    for _ in range(5):
        res = source.read_window()
        if res and res["window_data"] is not None:
            win = res
            break

    assert win is not None
    assert win["window_data"].shape == (1, 15, 480)
    assert win["channels"] == STEP62_TARGET_CHANNELS
    assert win["status"] == "STREAMING"
    assert win["is_simulated"] is False


def test_lsl_source_ultracortex_default_rejection():
    """Tests that connecting an inlet configured with Ultracortex default cap transitions to CONNECTED_INVALID."""
    info = MockStreamInfo("OpenBCI_Ultracortex_MarkIV", 16, 125.0, channel_names=OPENBCI_ULTRACORTEX_DEFAULT)
    inlet = MockInlet(info, chunk_size=64)
    source = LSLEEGSource(mock_inlet=inlet)

    assert source.is_connected() is True
    assert source.is_valid() is False
    assert source.state == StreamState.CONNECTED_INVALID
    assert "Ultracortex" in source._error_msg
    assert len(source.missing_channels) > 0

    win = source.read_window()
    assert win["window_data"] is None
    assert win["status"] == "CONNECTED_INVALID"


def test_lsl_source_resampling_125hz_to_160hz():
    """
    CRITICAL HARDWARE TEST:
    Cyton + Daisy broadcasts at 125.0 Hz over RFDuino dongle.
    Tests that LSLEEGSource resamples from 125.0 Hz to 160.0 Hz accurately.
    """
    info = MockStreamInfo("Cyton_Daisy_125Hz", 16, 125.0)
    inlet = MockInlet(info, chunk_size=125)  # 1 second of data at 125 Hz
    source = LSLEEGSource(mock_inlet=inlet, target_sfreq=160.0, target_samples=480)

    assert source.state == StreamState.CONNECTED_VALID
    assert source.nominal_sfreq == 125.0
    assert source.target_sfreq == 160.0

    # Feeding 4 seconds of 125 Hz data (500 samples) should resample to ~640 samples at 160 Hz
    win = None
    for _ in range(6):
        res = source.read_window()
        if res and res["window_data"] is not None:
            win = res
            break

    assert win is not None
    assert win["window_data"].shape == (1, 15, 480)
    assert win["sampling_rate_hz"] == 160.0


def test_lsl_disconnect_and_reconnect():
    """Tests explicit disconnection and reconnection lifecycle."""
    info = MockStreamInfo("Mock_EEG", 16, 160.0)
    inlet = MockInlet(info, chunk_size=64)
    source = LSLEEGSource(mock_inlet=inlet)

    assert source.is_connected() is True
    source.disconnect()
    assert source.is_connected() is False
    assert source.state == StreamState.DISCONNECTED
    assert source.inlet is None
    assert len(source.buffer) == 0

    win = source.read_window()
    assert win["window_data"] is None


def test_dataset_source_explicitly_labeled_simulated():
    """Verifies that DatasetEEGSource is always labeled as simulation and never claims hardware ready."""
    src = DatasetEEGSource()
    info = src.get_info()
    assert info["source_type"] == "SIMULATED_EDF"
    assert info["is_simulated"] is True
    assert info["hardware_ready"] is False
    assert "SIMULATION" in info["simulation_notice"]

    win = src.read_window()
    assert win["is_simulated"] is True
    assert win["window_data"].shape == (1, 15, 480)


# --------------------------------------------------------------------------
# Step 3: Signal Quality Gate Tests
# --------------------------------------------------------------------------
def test_signal_quality_clean_data():
    """Tests that realistic EEG data passes the signal quality gate."""
    gate = NeuroOSSignalQualityGate()
    data = np.random.randn(1, 15, 480) * 15.0  # 15 uV std
    res = gate.evaluate_quality(data, STEP62_TARGET_CHANNELS, sfreq=160.0)

    assert res["passed"] is True
    assert res["signal_quality"] == "EXCELLENT"
    assert res["bad_channels"] == []
    assert res["impedance_measured"] is False
    assert "not physical impedance" in res["notice"]


def test_signal_quality_nan_inf_detection():
    """Tests strict rejection of NaN and Inf samples."""
    gate = NeuroOSSignalQualityGate()
    data = np.random.randn(1, 15, 480) * 15.0
    data[0, 2, 100] = np.nan
    res = gate.evaluate_quality(data, STEP62_TARGET_CHANNELS)

    assert res["passed"] is False
    assert res["quality_status"] == "NAN_INF_DETECTED"


def test_signal_quality_flatline_detection():
    """Tests flatline channel detection when variance is near zero."""
    gate = NeuroOSSignalQualityGate()
    data = np.random.randn(1, 15, 480) * 15.0
    data[0, 5, :] = 0.0  # Flatline channel C3
    data[0, 6, :] = 0.0  # Flatline channel C1
    data[0, 7, :] = 0.0  # Flatline channel Cz
    res = gate.evaluate_quality(data, STEP62_TARGET_CHANNELS)

    assert res["passed"] is False
    assert "C3" in res["bad_channels"]
    assert "Cz" in res["bad_channels"]


def test_signal_quality_extreme_amplitude_artifact():
    """Tests artifact rejection when amplitude exceeds configurable max threshold (500 uV)."""
    gate = NeuroOSSignalQualityGate(max_amplitude_uv=500.0)
    data = np.random.randn(1, 15, 480) * 15.0
    data[0, 0, 50] = 850.0  # Excessive spike in FC3
    data[0, 1, 50] = 900.0
    data[0, 2, 50] = 750.0
    res = gate.evaluate_quality(data, STEP62_TARGET_CHANNELS)

    assert res["passed"] is False
    assert "FC3" in res["bad_channels"]


def test_signal_quality_staleness_detection():
    """Tests rejection when incoming data timestamp is stale (> 2.5s)."""
    gate = NeuroOSSignalQualityGate(max_stale_age_sec=2.5)
    data = np.random.randn(1, 15, 480) * 15.0
    stale_time = time.time() - 5.0
    res = gate.evaluate_quality(data, STEP62_TARGET_CHANNELS, timestamp=stale_time)

    assert res["passed"] is False
    assert "Stale stream data detected" in res["quality_warning"]


# --------------------------------------------------------------------------
# Step 4: Real Hardware Calibration Workflow Tests
# --------------------------------------------------------------------------
def test_calibration_blocks_when_source_disconnected():
    """Verifies that calibration cannot initiate if EEG source is disconnected."""
    mgr = NeuroOSCalibrationManager()
    source_info = {"connected": False, "status": "DISCONNECTED", "source_type": "LSL_HARDWARE"}
    res = mgr.start_calibration(source_info=source_info)

    assert res["started"] is False
    assert mgr.state == CalibrationState.ERROR
    assert "disconnected" in res["error"].lower()


def test_calibration_blocks_when_source_has_invalid_montage():
    """Verifies that calibration cannot initiate if EEG source has an invalid montage."""
    mgr = NeuroOSCalibrationManager()
    source_info = {
        "connected": True,
        "status": "CONNECTED_INVALID",
        "stream_state": "CONNECTED_INVALID",
        "error_detail": "Missing FC3, FC1",
        "source_type": "LSL_HARDWARE"
    }
    res = mgr.start_calibration(source_info=source_info)

    assert res["started"] is False
    assert mgr.state == CalibrationState.ERROR
    assert "invalid montage" in res["error"].lower()


def test_calibration_lifecycle_and_rejection():
    """Tests progression through REST -> LEFT_FIST -> RIGHT_FIST and window rejection."""
    mgr = NeuroOSCalibrationManager(target_windows_per_class=2)
    source_info = {"connected": True, "status": "STREAMING", "source_type": "LSL_HARDWARE", "is_simulated": False}
    start_res = mgr.start_calibration(source_info=source_info, budget=2)

    assert start_res["started"] is True
    assert mgr.state == CalibrationState.COLLECTING_REST
    assert mgr.get_active_class() == "REST"

    valid_win = np.random.randn(1, 15, 480) * 15.0
    good_quality = {"passed": True, "bad_channels": []}
    bad_quality = {"passed": False, "quality_warning": "NaN detected"}

    # 1. Record bad window: must be rejected without advancing count
    rec1 = mgr.record_window(valid_win, bad_quality)
    assert rec1["window_recorded"] is False
    assert mgr.rejected_windows_count == 1
    assert len(mgr.collected_windows["REST"]) == 0

    # 2. Record 2 good REST windows
    mgr.record_window(valid_win, good_quality)
    mgr.record_window(valid_win, good_quality)
    assert mgr.state == CalibrationState.COLLECTING_LEFT
    assert mgr.get_active_class() == "LEFT_FIST"

    # 3. Record 2 good LEFT windows
    mgr.record_window(valid_win, good_quality)
    mgr.record_window(valid_win, good_quality)
    assert mgr.state == CalibrationState.COLLECTING_RIGHT

    # 4. Record 2 good RIGHT windows -> Transitions to COMPLETED
    mgr.record_window(valid_win, good_quality)
    res_final = mgr.record_window(valid_win, good_quality)
    assert mgr.state == CalibrationState.COMPLETED
    assert res_final["progress_pct"] == 100.0


def test_calibration_cancel():
    """Tests user cancellation of calibration session."""
    mgr = NeuroOSCalibrationManager()
    mgr.start_calibration()
    assert mgr.state == CalibrationState.COLLECTING_REST

    cancel_res = mgr.cancel_calibration()
    assert cancel_res["state"] == CalibrationState.CANCELLED


def test_calibration_never_overwrites_step62():
    """Verifies that calibration saves metadata separately without touching Step 62 models."""
    mgr = NeuroOSCalibrationManager()
    source_info = {"connected": True, "status": "STREAMING", "source_type": "SIMULATED_EDF", "is_simulated": True}
    mgr.start_calibration(source_info=source_info, budget=1)

    win = np.random.randn(1, 15, 480) * 15.0
    good_q = {"passed": True}
    mgr.record_window(win, good_q)  # REST
    mgr.record_window(win, good_q)  # LEFT
    mgr.record_window(win, good_q)  # RIGHT -> COMPLETED

    assert mgr.state == CalibrationState.COMPLETED
    meta_path = os.path.join(mgr.save_dir, f"{mgr.session_id}_meta.json")
    assert os.path.exists(meta_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["base_model_preserved"] is True
    assert meta["model_version"] == "Step62-SurfaceLaplacian-Ensemble"


# --------------------------------------------------------------------------
# Step 5 & 6: Session Manager & Multi-Command Profile Regression Tests
# --------------------------------------------------------------------------
def dummy_predictor(subject_id="S001", run_id="R11", trial_idx=0):
    return {
        "predicted_label": "LEFT_FIST",
        "confidence": 0.85,
        "probabilities": {"REST": 0.05, "LEFT_FIST": 0.85, "RIGHT_FIST": 0.10}
    }


def test_session_manager_hardware_status_contract():
    """Tests that session manager exposes unified hardware status with 6 distinct states."""
    session = NeuroOSSessionManager()
    hw_stat = session.get_hardware_status()

    assert "active_source_type" in hw_stat
    assert "active_source" in hw_stat
    assert "lsl_bridge" in hw_stat
    assert "quality_thresholds" in hw_stat


def test_regression_scrolling_profile():
    """Verifies that Phase 2D scrolling profile continues to function with Phase 2E pipeline."""
    session = NeuroOSSessionManager(default_profile_id="scrolling")
    res = session.process_pipeline_step(dummy_predictor, allow_unstarted=True)

    assert res["active_profile"]["id"] == "scrolling"
    assert res["active_profile"]["adapter_mode"] == "SCROLL_DEMO"
    assert res["source_info"]["connected"] is True


def test_regression_document_navigation_profile():
    """Verifies that document navigation profile functions properly."""
    session = NeuroOSSessionManager()
    session.set_profile("document_navigation")
    res = session.process_pipeline_step(dummy_predictor, allow_unstarted=True)

    assert res["active_profile"]["id"] == "document_navigation"
    assert res["active_profile"]["adapter_mode"] == "DOCUMENT_NAVIGATION"


def test_regression_presentation_profile():
    """Verifies that presentation profile functions properly."""
    session = NeuroOSSessionManager()
    session.set_profile("presentation")
    res = session.process_pipeline_step(dummy_predictor, allow_unstarted=True)

    assert res["active_profile"]["id"] == "presentation"
    assert res["active_profile"]["adapter_mode"] == "PRESENTATION"


def test_emergency_lock_suppression():
    """Verifies that engaging emergency lock blocks command execution in session manager."""
    session = NeuroOSSessionManager()
    session.set_emergency_lock(True)
    res = session.process_pipeline_step(dummy_predictor, allow_unstarted=True)

    assert res["safety_state"]["emergency_locked"] is True
    assert res["action_execution"]["executed"] is False
    assert session.session_counters["emergency_lock_blocks_count"] >= 1
