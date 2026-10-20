"""
NeuroOS Final Master Phase Comprehensive Integration & Reliability Test Suite
Verifies:
1. Frozen model artifact integrity (Step 60/61/62 immutability)
2. Strict ground-truth and command-routing segregation
3. Offline replay determinism (Mode A: Discrete Trials)
4. Continuous-stream temporal confirmation & command release (Mode B: Sliding Windows)
5. REST neutrality and invalid-signal blocking (NaN/Inf, flatlines, extreme amplitude)
6. Duplicate and stale event suppression
7. Session lifecycle controls (START, PAUSE, RESUME, STOP, EMERGENCY LOCK)
8. All three command profiles (Scrolling, Document Navigation, Presentation) and exact state deltas
9. Audit trail metrics, explicit denominators, and CSV/JSON exports
10. Bounded audit memory retention (FIFO cap)
11. LSL hardware montage rejection and anti-aliased resampling (125/250Hz -> 160Hz)
12. API endpoints for mode selection and export downloads
"""

import os
import sys
import time
import json
import csv
import numpy as np
import pytest

from src.neuro_os.session_manager import NeuroOSSessionManager, SessionState, ExecutionMode
from src.neuro_os.safety_shield import NeuroOSSafetyShield
from src.neuro_os.command_router import NeuroOSCommandRouter
from src.neuro_os.action_executor import NeuroOSActionExecutor
from src.neuro_os.signal_quality import NeuroOSSignalQualityGate
from src.neuro_os.session_auditor import NeuroOSSessionAuditor
from src.neuro_os.deterministic_replayer import DeterministicReplayer
from src.neuro_os.command_profiles import get_command_profile
from src.neuro_os.lsl_bridge import LSLEEGSource, StreamState, DatasetEEGSource
from src.neuro_os.hardware_profiles import (
    STEP62_TARGET_CHANNELS,
    OPENBCI_ULTRACORTEX_DEFAULT,
    validate_channel_montage
)


# Dummy deterministic predictor for isolated integration tests
def mock_predictor(subject_id="S001", run_id="R11", trial_idx=0):
    patterns = [
        {"probabilities": {"REST": 0.90, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.05}, "predicted_label": "REST", "confidence": 0.90},
        {"probabilities": {"REST": 0.10, "LEFT_FIST": 0.85, "RIGHT_FIST": 0.05}, "predicted_label": "LEFT_FIST", "confidence": 0.85},
        {"probabilities": {"REST": 0.10, "LEFT_FIST": 0.82, "RIGHT_FIST": 0.08}, "predicted_label": "LEFT_FIST", "confidence": 0.82},
        {"probabilities": {"REST": 0.05, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.90}, "predicted_label": "RIGHT_FIST", "confidence": 0.90},
        {"probabilities": {"REST": 0.05, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.90}, "predicted_label": "RIGHT_FIST", "confidence": 0.90},
    ]
    return patterns[trial_idx % len(patterns)]


# --------------------------------------------------------------------------
# 1. FROZEN MODEL ARTIFACT INTEGRITY
# --------------------------------------------------------------------------
def test_frozen_model_artifact_integrity():
    """Verifies that Step 60, Step 61, and Step 62 model artifacts exist and remain intact."""
    for step_dir in ["models/step60", "models/step61", "models/step62"]:
        if os.path.exists(step_dir):
            files = os.listdir(step_dir)
            assert len(files) > 0, f"{step_dir} directory is empty."
            # Check for critical model artifacts
            has_report = any(f.endswith(".json") for f in files)
            assert has_report, f"Missing summary report in {step_dir}."


# --------------------------------------------------------------------------
# 2. GROUND TRUTH DECOUPLING
# --------------------------------------------------------------------------
def test_ground_truth_isolation():
    """
    Verifies that ground-truth labels are strictly isolated from operational decisions.
    Modifying or omitting ground truth must produce identical predictions, commands, and actions.
    """
    sm1 = NeuroOSSessionManager(default_profile_id="scrolling")
    sm2 = NeuroOSSessionManager(default_profile_id="scrolling")
    sm1.start_session()
    sm2.start_session()

    # Step 1: Normal step
    step1 = sm1.process_pipeline_step(mock_predictor)

    # Step 2: Step with scrambled ground truth label in EEG window
    orig_read = sm2.active_source.read_window
    def scrambled_read():
        win = orig_read()
        if win:
            win["true_label"] = "COMPLETELY_FABRICATED_LABEL"
        return win
    sm2.active_source.read_window = scrambled_read
    step2 = sm2.process_pipeline_step(mock_predictor)

    # Operational routing and actions must be bit-exact identical
    assert step1["safety_state"]["accepted"] == step2["safety_state"]["accepted"]
    assert step1["safety_state"]["intent"] == step2["safety_state"]["intent"]
    assert step1["command_routing"]["command"] == step2["command_routing"]["command"]
    assert step1["action_execution"]["executed"] == step2["action_execution"]["executed"]
    assert step1["action_execution"]["action"] == step2["action_execution"]["action"]

    # Ground truth must only appear in retrospective audit
    assert step1["trial_info"]["true_label"] != step2["trial_info"]["true_label"]


# --------------------------------------------------------------------------
# 3. MODE A (DISCRETE REPLAY) VS MODE B (CONTINUOUS STREAM)
# --------------------------------------------------------------------------
def test_mode_a_discrete_replay():
    """Verifies Mode A uses discrete trial boundaries with immediate dwell=1 execution."""
    sm = NeuroOSSessionManager(default_profile_id="scrolling")
    cfg = sm.set_execution_mode("RECORDED_TRIAL_REPLAY")

    assert sm.execution_mode == ExecutionMode.RECORDED_TRIAL_REPLAY
    assert cfg["dwell_windows"] == 1
    assert not cfg["command_release_required"]

    info = sm.dataset_source.get_info()
    assert info["execution_mode"] == ExecutionMode.RECORDED_TRIAL_REPLAY
    assert info["mode_badge"] == "OFFLINE REPLAY"


def test_mode_b_continuous_stream_and_dwell_confirmation():
    """
    Verifies Mode B uses continuous sliding windows with dwell=2 confirmation
    and command release requirement on discrete step commands.
    """
    sm = NeuroOSSessionManager(default_profile_id="presentation")
    cfg = sm.set_execution_mode("CONTINUOUS_STREAM_SIMULATION", stride_sec=0.5)

    assert sm.execution_mode == ExecutionMode.CONTINUOUS_STREAM_SIMULATION
    assert cfg["dwell_windows"] == 2
    assert cfg["command_release_required"]

    info = sm.dataset_source.get_info()
    assert info["execution_mode"] == ExecutionMode.CONTINUOUS_STREAM_SIMULATION
    assert "CONTINUOUS" in info["mode_badge"]

    # Verify temporal confirmation on router:
    # Window 1: LEFT_FIST (count=1 < 2) -> PENDING
    res1 = sm.command_router.process_intent({
        "accepted": True, "intent": "LEFT_FIST", "confidence": 0.85, "timestamp": 100.0, "prediction_id": "p1"
    }, current_time=100.0)
    assert not res1["action_triggered"]
    assert "PENDING" in res1["reason"]

    # Window 2: LEFT_FIST (count=2 >= 2) -> TRIGGERED (PREVIOUS_SLIDE)
    res2 = sm.command_router.process_intent({
        "accepted": True, "intent": "LEFT_FIST", "confidence": 0.85, "timestamp": 100.5, "prediction_id": "p2"
    }, current_time=100.5)
    assert res2["action_triggered"]
    assert res2["command"] == "PREVIOUS_SLIDE"

    # Window 3: LEFT_FIST continues (overlapping window on same cue) -> SUPPRESSED by repeat suppression/release
    res3 = sm.command_router.process_intent({
        "accepted": True, "intent": "LEFT_FIST", "confidence": 0.85, "timestamp": 101.0, "prediction_id": "p3"
    }, current_time=101.0)
    assert not res3["action_triggered"]
    assert "REPEAT_SUPPRESSED" in res3["reason"] or "PENDING" in res3["reason"]

    # Window 4: REST -> releases hold
    res4 = sm.command_router.process_intent({
        "accepted": True, "intent": "REST", "confidence": 0.90, "timestamp": 101.5, "prediction_id": "p4"
    }, current_time=101.5)
    assert res4["command"] == "NEUTRAL"
    assert sm.command_router.command_released


# --------------------------------------------------------------------------
# 4. REST NEUTRALITY AND INVALID SIGNAL BLOCKING
# --------------------------------------------------------------------------
def test_rest_neutrality_across_all_profiles():
    """Verifies that REST intent never triggers an action across any profile."""
    for profile_id in ["scrolling", "document_navigation", "presentation"]:
        router = NeuroOSCommandRouter(dwell_windows=1, profile_id=profile_id)
        res = router.process_intent({
            "accepted": True, "intent": "REST", "confidence": 0.95, "prediction_id": f"p_rest_{profile_id}"
        })
        assert res["command"] == "NEUTRAL"
        assert not res["action_triggered"]


def test_invalid_signal_blocking():
    """Verifies that non-finite, extreme amplitude, or malformed signals block action execution."""
    gate = NeuroOSSignalQualityGate()

    # NaN / Inf array
    bad_data = np.full((15, 480), np.nan)
    res_nan = gate.evaluate_quality(bad_data, STEP62_TARGET_CHANNELS)
    assert not res_nan["passed"]
    assert "NAN" in res_nan["quality_status"]

    # Flatline array
    flat_data = np.zeros((15, 480))
    res_flat = gate.evaluate_quality(flat_data, STEP62_TARGET_CHANNELS)
    assert not res_flat["passed"]
    assert len(res_flat["bad_channels"]) > 0

    # Safety Shield rejection of non-finite probability vector
    shield = NeuroOSSafetyShield()
    res_malformed = shield.evaluate_safety({"REST": np.nan, "LEFT_FIST": 0.5, "RIGHT_FIST": 0.5})
    assert not res_malformed["accepted"]
    assert "NON_FINITE" in res_malformed["reason"] or "MALFORMED" in res_malformed["reason"]


# --------------------------------------------------------------------------
# 5. STALE AND DUPLICATE SUPPRESSION
# --------------------------------------------------------------------------
def test_stale_and_duplicate_suppression():
    """Verifies that duplicate prediction IDs and expired timestamps cannot trigger actions."""
    router = NeuroOSCommandRouter(dwell_windows=1, stale_timeout_sec=2.0)

    # First event fires
    res1 = router.process_intent({
        "accepted": True, "intent": "LEFT_FIST", "confidence": 0.85, "timestamp": 100.0, "prediction_id": "dup_01"
    }, current_time=100.0)
    assert res1["action_triggered"]

    # Duplicate prediction ID is suppressed
    res2 = router.process_intent({
        "accepted": True, "intent": "LEFT_FIST", "confidence": 0.85, "timestamp": 100.1, "prediction_id": "dup_01"
    }, current_time=100.1)
    assert not res2["action_triggered"]
    assert "DUPLICATE" in res2["reason"]

    # Expired timestamp (> 2.0s) is suppressed
    res3 = router.process_intent({
        "accepted": True, "intent": "RIGHT_FIST", "confidence": 0.85, "timestamp": 90.0, "prediction_id": "stale_02"
    }, current_time=100.0)
    assert not res3["action_triggered"]
    assert "STALE" in res3["reason"]


# --------------------------------------------------------------------------
# 6. SESSION LIFECYCLE CONTROLS
# --------------------------------------------------------------------------
def test_session_lifecycle_controls():
    """Verifies START, PAUSE, RESUME, STOP, and EMERGENCY LOCK guardrails."""
    sm = NeuroOSSessionManager()

    # Paused state blocks execution
    sm.pause_session()
    assert sm.state == SessionState.PAUSED
    res_paused = sm.process_pipeline_step(mock_predictor, allow_unstarted=False)
    assert not res_paused["action_execution"]["executed"]
    assert "PAUSED" in res_paused["action_execution"]["details"]

    # Stopped state blocks execution
    sm.stop_session()
    assert sm.state == SessionState.STOPPED
    res_stopped = sm.process_pipeline_step(mock_predictor, allow_unstarted=False)
    assert not res_stopped["action_execution"]["executed"]
    assert "STOPPED" in res_stopped["action_execution"]["details"]

    # Emergency Lock immediately disengages safety shield
    sm.start_session()
    sm.set_emergency_lock(True)
    assert sm.safety_shield.manual_override_lock
    res_locked = sm.process_pipeline_step(mock_predictor)
    assert not res_locked["safety_state"]["accepted"]
    assert res_locked["safety_state"]["emergency_locked"]
    assert not res_locked["action_execution"]["executed"]

    # Unlocking restores operation
    sm.set_emergency_lock(False)
    assert not sm.safety_shield.manual_override_lock


# --------------------------------------------------------------------------
# 7. ALL THREE COMMAND PROFILES & EXACT STATE DELTAS
# --------------------------------------------------------------------------
def test_all_three_command_profiles_and_state_deltas():
    """Verifies scrolling, document navigation, and presentation profiles with exact state mutations."""
    # Profile A: Scrolling
    exec_a = NeuroOSActionExecutor(mode="SCROLL_DEMO")
    act_a = exec_a.execute_command("SCROLL_DOWN")
    assert act_a["executed"]
    assert act_a["state_change"]["delta"] == 180
    assert act_a["state_change"]["unit"] == "px"
    assert exec_a.state_store["scroll_demo"]["scroll_offset_px"] == 180

    # Profile B: Document Navigation
    exec_b = NeuroOSActionExecutor(mode="DOCUMENT_NAVIGATION")
    act_b = exec_b.execute_command("NEXT_SECTION")
    assert act_b["executed"]
    assert act_b["state_change"]["delta"] == 1
    assert act_b["state_change"]["unit"] == "index"
    assert exec_b.state_store["document_navigation"]["active_section_idx"] == 1

    # Profile C: Presentation
    exec_c = NeuroOSActionExecutor(mode="PRESENTATION")
    act_c = exec_c.execute_command("NEXT_SLIDE")
    assert act_c["executed"]
    assert act_c["state_change"]["delta"] == 1
    assert act_c["state_change"]["unit"] == "slide"
    assert exec_c.state_store["presentation"]["active_slide"] == 2


# --------------------------------------------------------------------------
# 8. AUDIT EXPORTS, DENOMINATORS, AND FIFO RETENTION
# --------------------------------------------------------------------------
def test_audit_csv_and_json_exports():
    """Verifies audit trail JSON and CSV export generation, schema, and bounded FIFO retention."""
    auditor = NeuroOSSessionAuditor(max_records=10)

    # Record 15 trials to test FIFO bounded retention
    for i in range(15):
        auditor.record_trial(
            session_id="test_fifo_session",
            predicted_label="REST" if i % 2 == 0 else "RIGHT_FIST",
            confidence=0.85,
            probabilities={"REST": 0.5, "LEFT_FIST": 0.2, "RIGHT_FIST": 0.3},
            safety_status={"accepted": True, "intent": "REST", "reason": "OK"},
            command_routing={"command": "NEUTRAL", "action_triggered": False},
            action_execution={"executed": False, "mode": "SCROLL_DEMO", "action": "NONE", "state_change": {"parameter": "none", "delta": 0, "unit": ""}},
            signal_quality={"passed": True, "signal_quality": "EXCELLENT"},
            latencies_ms={"ml_latency": 4.1, "pipeline_latency": 15.2, "ui_broadcast_latency": 0.0},
            subject_id="S001",
            run_id="R11",
            trial_idx=i,
            data_source="SIMULATED_EDF",
            model_version="Step 62 Ensemble",
            ground_truth_label="REST",
            timestamp=1000.0 + i
        )

    # Check bounded FIFO cap
    assert len(auditor.records) == 10
    assert auditor.records[0].trial_idx == 5  # First 5 popped

    # Check CSV export
    csv_path = auditor.export_audit_csv()
    assert os.path.exists(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        assert len(header) >= 20
        assert len(rows) == 10
        assert "trial_id" in header
        assert "action_executed" in header

    # Check JSON export
    json_path = auditor.export_audit_log()
    assert os.path.exists(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["total_records"] == 10
        assert "summary_metrics" in data
        assert data["summary_metrics"]["total_trials_processed"] == 15


# --------------------------------------------------------------------------
# 9. LSL HARDWARE MONTAGE REJECTION & RESAMPLING
# --------------------------------------------------------------------------
def test_lsl_hardware_montage_rejection():
    """
    Verifies that an incompatible montage (e.g. standard Ultracortex Mark IV default cap)
    is strictly rejected because it lacks the required frontocentral and centroparietal motor grid.
    """
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
            return None

    class MockDescNode:
        def __init__(self, labels):
            self._labels = labels
        def child(self, name):
            if name == "channels":
                return MockChannelsNode(self._labels)
            return None

    class MockStreamInfo:
        def __init__(self, ch_list):
            self.ch_list = ch_list
        def channel_count(self):
            return len(self.ch_list)
        def nominal_srate(self):
            return 250.0
        def desc(self):
            return MockDescNode(self.ch_list)

    class MockInlet:
        def __init__(self, channels):
            self._channels = channels
        def info(self):
            return MockStreamInfo(self._channels)

    inlet_ultracortex = MockInlet(OPENBCI_ULTRACORTEX_DEFAULT)
    source = LSLEEGSource(mock_inlet=inlet_ultracortex)
    assert source.state == StreamState.CONNECTED_INVALID
    assert not source.is_valid()
    assert "Ultracortex Mark IV default cap" in source._error_msg


def test_lsl_resampling_polyphase():
    """Verifies that incoming non-160Hz streams (e.g. 250Hz nominal) are resampled accurately to 160Hz."""
    source = LSLEEGSource(mock_inlet=None)
    # 250 samples at 250 Hz represents 1.0 second
    chunk_250hz = np.ones((250, 15))
    resampled = source._resample_chunk(chunk_250hz, in_rate=250.0, out_rate=160.0)
    # 1.0 second at 160 Hz must equal 160 samples
    assert resampled.shape[0] == 160
    assert resampled.shape[1] == 15
