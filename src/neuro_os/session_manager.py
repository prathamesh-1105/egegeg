"""
NeuroOS Real-Time Session & Pipeline State Manager (Phase 2D)
Coordinates EEGSource streaming, Signal Quality Gate, Calibration Manager,
Step 62 Decoder inference, Safety Shield, Multi-Command Router, Action Executor,
and real-time event broadcasting.
"""

import time
import uuid
import numpy as np
from typing import Dict, Any, Optional, List

from src.neuro_os.lsl_bridge import EEGSource, DatasetEEGSource, LSLEEGSource
from src.neuro_os.safety_shield import NeuroOSSafetyShield
from src.neuro_os.command_router import NeuroOSCommandRouter
from src.neuro_os.action_executor import NeuroOSActionExecutor
from src.neuro_os.event_logger import NeuroOSEventLogger
from src.neuro_os.signal_quality import NeuroOSSignalQualityGate
from src.neuro_os.calibration_manager import NeuroOSCalibrationManager
from src.neuro_os.command_profiles import get_command_profile, DEFAULT_PROFILE_ID
from src.neuro_os.session_auditor import NeuroOSSessionAuditor


class SessionState:
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class ExecutionMode:
    RECORDED_TRIAL_REPLAY = "RECORDED_TRIAL_REPLAY"  # Mode A (Discrete PhysioNet trials)
    CONTINUOUS_STREAM_SIMULATION = "CONTINUOUS_STREAM_SIMULATION"  # Mode B (Continuous sliding-window stream)


class NeuroOSSessionManager:
    """
    Central real-time controller for NeuroOS pipeline (Phase 2D/2E/2F/Master).
    Maintains clean separation between EEG streaming source, ML decoder, safety, and action logic.
    Enforces explicit user session initiation, emergency stop lockout, and stale prediction suppression.
    """

    def __init__(self, default_profile_id: str = DEFAULT_PROFILE_ID):
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.state = SessionState.CONNECTED
        self.active_source_type = "SIMULATED_EDF"
        self.execution_mode = ExecutionMode.RECORDED_TRIAL_REPLAY

        # Instantiate sources
        self.dataset_source = DatasetEEGSource()
        self.lsl_source = LSLEEGSource()
        self.active_source: EEGSource = self.dataset_source

        # Instantiate pipeline components
        self.signal_quality_gate = NeuroOSSignalQualityGate()
        self.calibration_manager = NeuroOSCalibrationManager()
        self.safety_shield = NeuroOSSafetyShield(confidence_thresh=0.60, margin_thresh=0.12, smoothing_window_size=3)
        self.command_router = NeuroOSCommandRouter(
            dwell_windows=1,
            cooldown_sec=0.8,
            profile_id=default_profile_id
        )
        self.action_executor = NeuroOSActionExecutor(mode="SCROLL_DEMO")
        self.event_logger = NeuroOSEventLogger(max_history=100)
        self.auditor = NeuroOSSessionAuditor(max_records=1000)
        self.session_auditor = self.auditor  # Backwards compatibility alias

        # Active Model Identifier
        self.active_model_id = "Step 62 Surface Laplacian Multimodal Balanced Ensemble (Frozen Base)"

        # Session Telemetry Counters
        self.session_counters = {
            "total_windows_processed": 0,
            "accepted_predictions_count": 0,
            "rejected_predictions_count": 0,
            "rest_neutral_count": 0,
            "actions_executed_count": 0,
            "stale_suppressed_count": 0,
            "emergency_lock_blocks_count": 0
        }

        self.last_event_timestamp = time.time()
        self.event_sequence_num = 0

        # Initialize default profile mapping
        self.set_profile(default_profile_id)

    def set_model_version(self, model_version_id: str) -> Dict[str, Any]:
        """
        Configures the active BCI decoder model version.
        Supports versioned selection between Step 62 (default champion) and Phase 4
        promoted research models (e.g. Phase4-AnatomicalLaplacian-Ensemble, Phase4-K28-Ensemble).
        """
        self.active_model_id = model_version_id
        self.event_logger.log_event(
            event_type="MODEL_VERSION_SWITCHED",
            intent="SYSTEM",
            confidence=1.0,
            safety_status={"passed": True},
            command="SET_MODEL",
            action={"executed": True, "details": f"Switched model version to {model_version_id}"},
            details=f"Active model: {model_version_id}"
        )
        return {
            "status": "SUCCESS",
            "active_model_id": self.active_model_id,
            "session_id": self.session_id
        }

    def set_profile(self, profile_id: str) -> Dict[str, Any]:
        """
        Configures active command profile and synchronizes the action executor adapter mode.
        Safe and non-destructive: does not corrupt existing session state or counters.
        """
        profile = self.command_router.set_profile(profile_id)
        self.action_executor.set_adapter_mode(profile.adapter_mode)

        self.event_logger.log_event(
            event_type="PROFILE_SWITCHED",
            intent="SYSTEM",
            confidence=1.0,
            safety_status={"passed": True},
            command="SET_PROFILE",
            action={"executed": True, "details": f"Switched profile to {profile.name}"},
            details=f"Active profile: {profile.id} -> Adapter: {profile.adapter_mode}"
        )

        return profile.to_dict()

    def set_source(self, source_type: str) -> Dict[str, Any]:
        """Switches between SIMULATED_EDF and LSL_HARDWARE."""
        if source_type in ["SIMULATED", "SIMULATED_EDF", "DATASET"]:
            self.active_source_type = "SIMULATED_EDF"
            self.active_source = self.dataset_source
            if self.state not in [SessionState.RUNNING, SessionState.PAUSED]:
                self.state = SessionState.CONNECTED
        elif source_type in ["LSL", "LSL_HARDWARE", "HARDWARE"]:
            self.active_source_type = "LSL_HARDWARE"
            self.active_source = self.lsl_source
            if not self.lsl_source.is_connected():
                self.state = SessionState.DISCONNECTED
            else:
                if self.state not in [SessionState.RUNNING, SessionState.PAUSED]:
                    self.state = SessionState.CONNECTED
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        info = self.active_source.get_info()
        self.event_logger.log_event(
            event_type="SOURCE_SWITCHED",
            intent="SYSTEM",
            confidence=1.0,
            safety_status={"passed": True},
            command="NEUTRAL",
            action={"executed": False},
            details=f"Switched source to {self.active_source_type} ({info['name']})"
        )
        return info

    def set_execution_mode(self, mode: str, stride_sec: float = 0.5) -> Dict[str, Any]:
        """
        Configures execution mode between Mode A (Recorded-Trial Replay) and Mode B (Continuous-Stream Simulation).
        Mode A: Discrete recorded trial boundaries, dwell_windows=1 default, immediate trial execution.
        Mode B: Continuous sliding-window stream, dwell_windows=2 default, temporal confirmation & command release.
        """
        clean_mode = str(mode).upper()
        if "CONTINUOUS" in clean_mode or clean_mode in ["MODE_B", "STREAM", "CONTINUOUS_STREAM_SIMULATION"]:
            self.execution_mode = ExecutionMode.CONTINUOUS_STREAM_SIMULATION
            self.dataset_source.set_execution_mode("CONTINUOUS_STREAM_SIMULATION", stride_sec=stride_sec)
            self.command_router.set_temporal_confirmation_windows(2)
            self.command_router.command_release_required = True
            self.command_router.cooldown_sec = 0.8
        else:
            self.execution_mode = ExecutionMode.RECORDED_TRIAL_REPLAY
            self.dataset_source.set_execution_mode("RECORDED_TRIAL_REPLAY")
            self.command_router.set_temporal_confirmation_windows(1)
            self.command_router.command_release_required = False
            self.command_router.repeat_suppression = True

        self.command_router.reset_confirmation()
        self.event_logger.log_event(
            event_type="EXECUTION_MODE_CHANGED",
            intent="SYSTEM",
            confidence=1.0,
            safety_status={"passed": True},
            command="SET_MODE",
            action={"executed": True},
            details=f"Switched execution mode to {self.execution_mode} (dwell={self.command_router.temporal_confirmation_windows})"
        )
        return {
            "execution_mode": self.execution_mode,
            "mode_badge": "OFFLINE REPLAY" if self.execution_mode == ExecutionMode.RECORDED_TRIAL_REPLAY else "SIMULATION (CONTINUOUS STREAM)",
            "dwell_windows": self.command_router.temporal_confirmation_windows,
            "command_release_required": self.command_router.command_release_required,
            "cooldown_sec": self.command_router.cooldown_sec,
            "source_info": self.active_source.get_info()
        }

    def reconnect_hardware(self) -> Dict[str, Any]:
        """Triggers hardware reconnection and stream rediscovery."""
        success = self.lsl_source.reconnect()
        if self.active_source_type == "LSL_HARDWARE":
            if not self.lsl_source.is_connected():
                self.state = SessionState.DISCONNECTED
            elif self.lsl_source.is_valid():
                self.state = SessionState.CONNECTED
        info = self.lsl_source.get_info()
        self.event_logger.log_event(
            event_type="HARDWARE_RECONNECT_ATTEMPTED",
            intent="SYSTEM",
            confidence=1.0,
            safety_status={"passed": success},
            command="RECONNECT",
            action={"executed": success},
            details=f"Hardware reconnect status: {info['status']} | Error: {info.get('error_detail')}"
        )
        return info

    def get_hardware_status(self) -> Dict[str, Any]:
        """Returns unified diagnostic status for active source and LSL bridge."""
        active_info = self.active_source.get_info()
        lsl_info = self.lsl_source.get_info()
        return {
            "active_source_type": self.active_source_type,
            "session_state": self.state,
            "active_source": active_info,
            "lsl_bridge": lsl_info,
            "quality_thresholds": self.signal_quality_gate.get_thresholds()
        }

    def start_session(self):
        """Starts real-time session streaming. Explicit user action."""
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.state = SessionState.RUNNING
        self.auditor.reset(session_id=self.session_id)
        self.event_logger.log_event(
            event_type="SESSION_STARTED",
            intent="SYSTEM",
            confidence=1.0,
            safety_status={"passed": True},
            command="START",
            action={"executed": True},
            details=f"Session {self.session_id} started."
        )

    def pause_session(self):
        """Pauses real-time session streaming."""
        self.state = SessionState.PAUSED
        self.command_router.reset_confirmation()
        self.event_logger.log_event(
            event_type="SESSION_PAUSED",
            intent="SYSTEM",
            confidence=1.0,
            safety_status={"passed": True},
            command="PAUSE",
            action={"executed": True},
            details=f"Session {self.session_id} paused."
        )

    def resume_session(self):
        """Resumes real-time session streaming from paused state."""
        if self.state == SessionState.PAUSED:
            self.state = SessionState.RUNNING
            self.command_router.reset_confirmation()
            self.event_logger.log_event(
                event_type="SESSION_RESUMED",
                intent="SYSTEM",
                confidence=1.0,
                safety_status={"passed": True},
                command="RESUME",
                action={"executed": True},
                details=f"Session {self.session_id} resumed."
            )

    def stop_session(self):
        """Stops real-time session streaming and disengages actions."""
        self.state = SessionState.STOPPED
        self.command_router.reset_confirmation()
        self.event_logger.log_event(
            event_type="SESSION_STOPPED",
            intent="SYSTEM",
            confidence=1.0,
            safety_status={"passed": True},
            command="STOP",
            action={"executed": True},
            details=f"Session {self.session_id} stopped."
        )

    def set_emergency_lock(self, locked: bool) -> bool:
        """Immediately locks or unlocks the safety shield."""
        self.safety_shield.set_manual_lock(locked)
        self.command_router.reset_confirmation()
        self.event_logger.log_event(
            event_type="EMERGENCY_LOCK_CHANGED",
            intent="SYSTEM",
            confidence=1.0,
            safety_status={"passed": not locked, "emergency_locked": locked},
            command="LOCK" if locked else "UNLOCK",
            action={"executed": True},
            details=f"Emergency Lock set to: {locked}"
        )
        return locked

    def reset_counters(self):
        """Resets telemetry session counters."""
        for k in self.session_counters:
            self.session_counters[k] = 0
        self.auditor.reset(session_id=self.session_id)

    def process_pipeline_step(self, predictor_func, allow_unstarted: bool = True) -> Dict[str, Any]:
        """
        Executes one full synchronous end-to-end iteration of the streaming pipeline:
        EEG Window Read -> Signal Quality Gate -> ML Decoder -> Safety Shield -> Command Router -> Action Executor
        Enforces stale prediction detection and explicit session state guardrails.
        """
        t_pipeline_start = time.perf_counter()
        self.event_sequence_num += 1
        now = time.time()

        # 1. Read Window from Active Source
        src_info = self.active_source.get_info()
        win_sample = self.active_source.read_window()

        if win_sample is None or win_sample.get("window_data") is None:
            t_pipeline_end = time.perf_counter()
            return {
                "event_id": f"evt_{int(now)}_{self.event_sequence_num}",
                "timestamp": now,
                "session_id": self.session_id,
                "session_state": self.state,
                "active_model_id": self.active_model_id,
                "active_profile": self.command_router.active_profile.to_dict(),
                "stage": "EEG_RECEIVED",
                "source_info": src_info,
                "adapter_telemetry": src_info.get("adapter_telemetry"),
                "signal_quality": {
                    "passed": False,
                    "signal_quality": "DISCONNECTED",
                    "quality_status": "NO_STREAM",
                    "bad_channels": [],
                    "quality_warning": src_info.get("error_detail") or "No active EEG data stream."
                },
                "calibration_status": self.calibration_manager.get_status(),
                "latencies_ms": {
                    "ml_latency": 0.0,
                    "pipeline_latency": round((t_pipeline_end - t_pipeline_start) * 1000.0, 2),
                    "ui_broadcast_latency": 0.0
                },
                "decoder_output": None,
                "safety_state": {
                    "accepted": False,
                    "passed": False,
                    "intent": "REST",
                    "confidence": 0.0,
                    "margin": 0.0,
                    "entropy": 0.0,
                    "rejection_reason": src_info.get("error_detail") or "NO_DATA_STREAM",
                    "dwell_progress": 0.0,
                    "cooldown_active": False,
                    "emergency_locked": self.safety_shield.system_emergency_locked
                },
                "command_routing": {"command": "NEUTRAL", "action_triggered": False},
                "action_execution": {"executed": False, "mode": self.action_executor.mode, "action": "NONE", "state_change": {"parameter": "none", "delta": 0}},
                "session_counters": dict(self.session_counters),
                "session_analytics": self.auditor.get_summary_metrics(),
                "latest_audit_record": None,
                "event_log": self.event_logger.get_recent_logs(limit=5)
            }

        self.session_counters["total_windows_processed"] += 1

        # 2. Signal Quality Check
        win_data = win_sample["window_data"]
        ch_names = win_sample.get("channels", [])
        sfreq = win_sample.get("sampling_rate_hz", 160.0)
        win_ts = win_sample.get("timestamp")

        expected_ch = ch_names if (self.active_source_type in ["DATASET", "SIMULATED_EDF"] and ch_names) else None
        quality_res = self.signal_quality_gate.evaluate_quality(
            win_data, ch_names, sfreq, timestamp=win_ts, expected_channels=expected_ch
        )

        # Record window if calibration session active
        if self.calibration_manager.state != "IDLE":
            self.calibration_manager.record_window(win_data, quality_res, timestamp=win_ts)

        # 3. Run ML Decoder safely with exception & schema validation guardrails
        t_ml_start = time.perf_counter()
        subject_id = win_sample.get("subject_id", "S001")
        run_id = win_sample.get("run_id", "R11")
        trial_idx = win_sample.get("trial_idx", 0)

        try:
            pred_res = predictor_func(subject_id=subject_id, run_id=run_id, trial_idx=trial_idx, model_version=self.active_model_id)
            if not isinstance(pred_res, dict) or "probabilities" not in pred_res:
                raise ValueError("Predictor returned invalid result schema (missing probabilities).")
            raw_probs = pred_res["probabilities"]
            predicted_label = pred_res.get("predicted_label", "REST")
            pred_confidence = pred_res.get("confidence", 0.0)
        except Exception as e:
            raw_probs = {"REST": 1.0, "LEFT_FIST": 0.0, "RIGHT_FIST": 0.0}
            predicted_label = "REST"
            pred_confidence = 0.0
            pred_res = {
                "probabilities": raw_probs,
                "predicted_label": "REST",
                "confidence": 0.0,
                "error": str(e)
            }
        t_ml_end = time.perf_counter()
        ml_latency = round((t_ml_end - t_ml_start) * 1000.0, 2)

        # 4. Safety Shield Evaluation
        safety_res = self.safety_shield.evaluate_safety(
            raw_probs=raw_probs,
            hardware_connected=src_info["connected"],
            signal_quality_ok=quality_res["passed"]
        )

        # Attach unique prediction ID and timestamp for freshness tracking
        prediction_id = f"pred_{subject_id}_{run_id}_{trial_idx}_{self.event_sequence_num}"
        safety_res["prediction_id"] = prediction_id
        safety_res["timestamp"] = now

        # Update Safety Counters
        if safety_res.get("emergency_locked") or self.safety_shield.manual_override_lock:
            self.session_counters["emergency_lock_blocks_count"] += 1

        if safety_res.get("accepted", False):
            self.session_counters["accepted_predictions_count"] += 1
            if safety_res.get("intent") == "REST":
                self.session_counters["rest_neutral_count"] += 1
        else:
            self.session_counters["rejected_predictions_count"] += 1

        # 5. Command Routing
        routing_res = self.command_router.process_intent(safety_res)

        if "STALE" in routing_res.get("reason", ""):
            self.session_counters["stale_suppressed_count"] += 1

        # 6. Action Execution (Enforcing Session Lifecycle: RUNNING or allow_unstarted single-step)
        action_res = {
            "executed": False,
            "command": routing_res.get("command", "NEUTRAL"),
            "action": "NONE",
            "action_name": "NONE",
            "details": "No action executed"
        }

        # Check if session allows action execution
        can_execute = (self.state == SessionState.RUNNING) or (allow_unstarted and self.state in [SessionState.CONNECTED, SessionState.RUNNING])

        if not can_execute:
            action_res["details"] = f"Action blocked: Session is {self.state} (Requires explicit Start/Resume)"
        elif routing_res.get("action_triggered", False):
            action_res = self.action_executor.execute_command(routing_res["command"])
            if action_res.get("executed", False):
                self.session_counters["actions_executed_count"] += 1

        t_pipeline_end = time.perf_counter()
        pipeline_latency = round((t_pipeline_end - t_pipeline_start) * 1000.0, 2)

        # 7. Event Logging
        log_entry = self.event_logger.log_event(
            event_type="PIPELINE_STEP",
            intent=safety_res["intent"],
            confidence=safety_res["confidence"],
            safety_status=safety_res,
            command=routing_res["command"],
            action=action_res,
            details=f"Source: {src_info['name']} | True: {win_sample.get('true_label', 'N/A')} | Profile: {self.command_router.profile_id}"
        )

        # 8. Decoupled Ground-Truth Session Audit Logging
        audit_record = self.auditor.record_trial(
            session_id=self.session_id,
            predicted_label=pred_res["predicted_label"],
            confidence=pred_res["confidence"],
            probabilities=raw_probs,
            safety_status=safety_res,
            command_routing=routing_res,
            action_execution=action_res,
            signal_quality=quality_res,
            latencies_ms={
                "ml_latency": ml_latency,
                "pipeline_latency": pipeline_latency,
                "ui_broadcast_latency": 0.0
            },
            subject_id=subject_id,
            run_id=run_id,
            trial_idx=trial_idx,
            data_source=src_info.get("source_type", "SIMULATED_EDF"),
            model_version=self.active_model_id,
            ground_truth_label=win_sample.get("true_label"),
            timestamp=now
        )

        stage_str = "ACTION_EXECUTED" if action_res.get("executed") else (
            "COMMAND_DECIDED" if routing_res.get("command") != "NEUTRAL" else "SAFETY_CHECK"
        )

        return {
            "event_id": f"evt_{int(now)}_{self.event_sequence_num}",
            "timestamp": now,
            "session_id": self.session_id,
            "session_state": self.state,
            "execution_mode": self.execution_mode,
            "mode_badge": "OFFLINE REPLAY" if self.execution_mode == ExecutionMode.RECORDED_TRIAL_REPLAY else "SIMULATION (CONTINUOUS STREAM)",
            "active_model_id": self.active_model_id,
            "active_profile": self.command_router.active_profile.to_dict(),
            "stage": stage_str,
            "source_info": src_info,
            "adapter_telemetry": win_sample.get("adapter_telemetry") or src_info.get("adapter_telemetry"),
            "signal_quality": quality_res,
            "calibration_status": self.calibration_manager.get_status(),
            "trial_info": {
                "subject_id": subject_id,
                "run_id": run_id,
                "trial_idx": trial_idx,
                "true_label": win_sample.get("true_label", "UNKNOWN"),
                "predicted_label": pred_res["predicted_label"]
            },
            "latencies_ms": {
                "ml_latency": ml_latency,
                "pipeline_latency": pipeline_latency,
                "ui_broadcast_latency": 0.0
            },
            "decoder_output": {
                "predicted_label": pred_res["predicted_label"],
                "confidence": pred_res["confidence"],
                "probabilities": raw_probs
            },
            "safety_state": safety_res,
            "command_routing": routing_res,
            "action_execution": {
                "mode": self.action_executor.mode,
                "executed": action_res.get("executed", False),
                "command": action_res.get("command", "NEUTRAL"),
                "action": action_res.get("action") or action_res.get("action_name", "NONE"),
                "action_name": action_res.get("action_name", "NONE"),
                "details": action_res.get("details", ""),
                "state_change": action_res.get("state_change", {"parameter": "none", "delta": 0})
            },
            "system_state": {
                "active_adapter_mode": self.action_executor.mode,
                **self.action_executor.state_store
            },
            "session_counters": dict(self.session_counters),
            "session_analytics": self.auditor.get_summary_metrics(),
            "latest_audit_record": audit_record.to_dict(),
            "event_log": self.event_logger.get_recent_logs(limit=10)
        }

    def get_audit_analytics(self) -> Dict[str, Any]:
        """Returns structured session reliability metrics & latency percentiles."""
        return self.auditor.get_summary_metrics()

    def get_audit_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent trial audit records."""
        return [r.to_dict() for r in self.auditor.records[-limit:]]

    def export_audit_log(self, filepath: Optional[str] = None) -> str:
        """Exports audit records to JSON file."""
        return self.auditor.export_audit_log(filepath)

    def export_audit_json(self, filepath: Optional[str] = None) -> str:
        """Exports audit records to JSON file."""
        return self.auditor.export_audit_log(filepath)

    def export_audit_csv(self, filepath: Optional[str] = None) -> str:
        """Exports audit records to CSV file."""
        return self.auditor.export_audit_csv(filepath)

