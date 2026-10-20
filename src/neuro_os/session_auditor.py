"""
NeuroOS Session Replay & Telemetry Audit Engine (Phase 2F)
Maintains structured, trial-by-trial audit logs for real-time BCI streaming and deterministic replay.

CRITICAL ARCHITECTURAL CONSTRAINTS:
- Ground-truth evaluation layer is strictly decoupled from the live command-dispatch layer.
- Ground-truth labels are NEVER used to determine or influence software commands.
- Measures command reliability (executed vs suppressed actions) independently of classifier accuracy.
- Records before-and-after state changes for every executed software action.
- Tracks genuine ML inference, safety shield, and pipeline latencies with P95/P99 analytics.
"""

import time
import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score


@dataclass
class SessionAuditRecord:
    trial_id: str
    session_id: str
    timestamp: float
    iso_time: str
    subject_id: Optional[str]
    run_id: Optional[str]
    trial_idx: Optional[int]
    data_source: str
    model_version: str
    ground_truth_label: Optional[str]  # Offline evaluation only; never used in dispatch
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    is_correct_classification: Optional[bool]
    signal_quality_passed: bool
    signal_quality_status: str
    safety_accepted: bool
    safety_intent: str
    safety_rejection_reason: Optional[str]
    command_profile_id: str
    mapped_command: str
    action_executed: bool
    action_name: str
    is_incorrect_action: Optional[bool]
    state_change: Dict[str, Any]
    latencies_ms: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NeuroOSSessionAuditor:
    """
    Comprehensive session auditor and analytics engine for NeuroOS (Phase 2F).
    Computes rigorous reliability metrics, latency distributions, and safety statistics
    with clearly documented denominators.
    """

    def __init__(self, max_records: int = 2000, save_dir: str = "data/audit_logs"):
        self.max_records = max_records
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.records: List[SessionAuditRecord] = []
        self.session_id: Optional[str] = None
        self.active_profile_id: str = "scrolling"
        self.model_version: str = "Step 62 Surface Laplacian Multimodal Balanced Ensemble (Frozen Base)"

        # Live session counters
        self.counters = {
            "total_processed": 0,
            "labeled_trials_count": 0,
            "classification_correct": 0,
            "classification_errors": 0,
            "safety_accepted": 0,
            "safety_rejected": 0,
            "accepted_correct": 0,
            "actions_executed": 0,
            "actions_correct": 0,
            "actions_incorrect": 0,
            "suppressed_errors": 0,
            "suppressed_correct": 0,
            "rest_trials_count": 0,
            "rest_neutral_count": 0,
            "stale_suppressed": 0,
            "emergency_lock_suppressed": 0
        }

    def reset(self, session_id: Optional[str] = None):
        """Clears audit records and resets counters."""
        self.records.clear()
        self.session_id = session_id or f"audit_{int(time.time())}"
        for k in self.counters:
            self.counters[k] = 0

    def record_trial(
        self,
        session_id: str,
        predicted_label: str,
        confidence: float,
        probabilities: Dict[str, float],
        safety_status: Dict[str, Any],
        command_routing: Dict[str, Any],
        action_execution: Dict[str, Any],
        signal_quality: Dict[str, Any],
        latencies_ms: Dict[str, float],
        subject_id: Optional[str] = None,
        run_id: Optional[str] = None,
        trial_idx: Optional[int] = None,
        data_source: str = "SIMULATED_EDF",
        model_version: Optional[str] = None,
        ground_truth_label: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> SessionAuditRecord:
        """
        Records full trial execution audit record.
        Strictly preserves separation: ground_truth_label is used ONLY for offline scoring.
        """
        now = timestamp or time.time()
        self.session_id = session_id
        trial_id = f"aud_{session_id}_{self.counters['total_processed'] + 1}"

        # 1. Classification Correctness (Offline Evaluation Only)
        is_correct_class = None
        if ground_truth_label is not None and ground_truth_label in ["REST", "LEFT_FIST", "RIGHT_FIST"]:
            is_correct_class = (predicted_label == ground_truth_label)

        # 2. Safety Decisions
        safety_acc = bool(safety_status.get("accepted", False))
        safety_intent = safety_status.get("intent", "REST")
        rejection_reason = safety_status.get("rejection_reason") or safety_status.get("reason")
        if not safety_acc and not rejection_reason:
            rejection_reason = "CONFIDENCE_OR_MARGIN_LOW"

        # 3. Command & Action Decisions
        mapped_cmd = command_routing.get("command", "NEUTRAL")
        action_exec = bool(action_execution.get("executed", False))
        action_name = action_execution.get("action") or action_execution.get("action_name", "NONE")
        state_change = action_execution.get("state_change") or {"parameter": "none", "delta": 0}

        # 4. Action Correctness (Offline Evaluation Only)
        is_incorrect_action = None
        if ground_truth_label is not None:
            if action_exec:
                # An action was executed. Was the user actually intending this action?
                # If ground truth was REST: any executed action is an error!
                if ground_truth_label == "REST":
                    is_incorrect_action = True
                # If ground truth was LEFT_FIST but right action executed (or vice versa):
                elif ground_truth_label == "LEFT_FIST" and ("NEXT" in mapped_cmd or "DOWN" in mapped_cmd):
                    is_incorrect_action = True
                elif ground_truth_label == "RIGHT_FIST" and ("PREVIOUS" in mapped_cmd or "UP" in mapped_cmd):
                    is_incorrect_action = True
                else:
                    is_incorrect_action = False

        # 5. Update Aggregation Counters
        self.counters["total_processed"] += 1

        if is_correct_class is not None:
            self.counters["labeled_trials_count"] += 1
            if is_correct_class:
                self.counters["classification_correct"] += 1
            else:
                self.counters["classification_errors"] += 1

        if safety_acc:
            self.counters["safety_accepted"] += 1
            if is_correct_class is True:
                self.counters["accepted_correct"] += 1
        else:
            self.counters["safety_rejected"] += 1

        if action_exec:
            self.counters["actions_executed"] += 1
            if is_incorrect_action is True:
                self.counters["actions_incorrect"] += 1
            elif is_incorrect_action is False:
                self.counters["actions_correct"] += 1

        # Track safety shield suppression effectiveness
        if ground_truth_label is not None:
            # Did safety shield suppress a misclassification?
            if not is_correct_class and not action_exec:
                self.counters["suppressed_errors"] += 1
            # Did safety rules suppress a correct intention (cooldown / margin / dwell cost)?
            elif is_correct_class and ground_truth_label in ["LEFT_FIST", "RIGHT_FIST"] and not action_exec:
                self.counters["suppressed_correct"] += 1

            # REST Neutrality tracking
            if ground_truth_label == "REST":
                self.counters["rest_trials_count"] += 1
                if not action_exec and mapped_cmd in ["NEUTRAL", "NONE"]:
                    self.counters["rest_neutral_count"] += 1

        if "STALE" in command_routing.get("reason", ""):
            self.counters["stale_suppressed"] += 1

        if safety_status.get("emergency_locked") or safety_status.get("manual_override_lock"):
            self.counters["emergency_lock_suppressed"] += 1

        # Create structured record
        record = SessionAuditRecord(
            trial_id=trial_id,
            session_id=session_id,
            timestamp=now,
            iso_time=time.strftime("%H:%M:%S", time.localtime(now)),
            subject_id=subject_id,
            run_id=run_id,
            trial_idx=trial_idx,
            data_source=data_source,
            model_version=model_version or self.model_version,
            ground_truth_label=ground_truth_label,
            predicted_label=predicted_label,
            confidence=round(float(confidence), 4),
            probabilities={k: round(float(v), 4) for k, v in probabilities.items()},
            is_correct_classification=is_correct_class,
            signal_quality_passed=bool(signal_quality.get("passed", True)),
            signal_quality_status=signal_quality.get("quality_status", "OK"),
            safety_accepted=safety_acc,
            safety_intent=safety_intent,
            safety_rejection_reason=rejection_reason,
            command_profile_id=command_routing.get("profile_id", "scrolling"),
            mapped_command=mapped_cmd,
            action_executed=action_exec,
            action_name=action_name,
            is_incorrect_action=is_incorrect_action,
            state_change=state_change,
            latencies_ms={k: round(float(v), 2) for k, v in latencies_ms.items()}
        )

        self.records.append(record)
        if len(self.records) > self.max_records:
            self.records.pop(0)

        return record

    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Calculates comprehensive reliability, accuracy, and latency metrics with exact denominators.
        """
        n_total = max(1, self.counters["total_processed"])
        n_labeled = max(1, self.counters["labeled_trials_count"])
        n_accepted = max(1, self.counters["safety_accepted"])
        n_executed = max(1, self.counters["actions_executed"])
        n_rest = max(1, self.counters["rest_trials_count"])

        # Latency distribution (pipeline latency)
        latencies = [r.latencies_ms.get("pipeline_latency", 0.0) for r in self.records if r.latencies_ms]
        if not latencies:
            latencies = [0.0]

        mean_lat = float(np.mean(latencies))
        p50_lat = float(np.percentile(latencies, 50))
        p95_lat = float(np.percentile(latencies, 95))
        p99_lat = float(np.percentile(latencies, 99))
        min_lat = float(np.min(latencies))
        max_lat = float(np.max(latencies))

        # Raw classification accuracy
        raw_acc = (self.counters["classification_correct"] / n_labeled) * 100.0 if self.counters["labeled_trials_count"] > 0 else None

        # Accepted prediction accuracy
        accepted_acc = (self.counters["accepted_correct"] / n_accepted) * 100.0 if self.counters["safety_accepted"] > 0 else None

        # Safety rejection rate
        rejection_rate = (self.counters["safety_rejected"] / n_total) * 100.0

        # Command execution rate
        exec_rate = (self.counters["actions_executed"] / n_total) * 100.0

        # Incorrect command rate (relative to all trials)
        incorrect_rate_total = (self.counters["actions_incorrect"] / n_total) * 100.0

        # Incorrect command rate (relative to executed commands)
        incorrect_rate_executed = (self.counters["actions_incorrect"] / n_executed) * 100.0 if self.counters["actions_executed"] > 0 else 0.0

        # REST neutrality rate
        rest_neutrality_rate = (self.counters["rest_neutral_count"] / n_rest) * 100.0 if self.counters["rest_trials_count"] > 0 else 100.0

        # Macro F1 and per-class stats if labeled records exist
        labeled_records = [r for r in self.records if r.ground_truth_label is not None]
        macro_f1 = None
        per_class_metrics = {}

        if len(labeled_records) >= 5:
            y_true = [r.ground_truth_label for r in labeled_records]
            y_pred = [r.predicted_label for r in labeled_records]
            classes_present = sorted(list(set(y_true)))

            try:
                macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) * 100.0
                prec, rec, f1_vals, sup = precision_recall_fscore_support(y_true, y_pred, labels=classes_present, zero_division=0)
                for idx, c in enumerate(classes_present):
                    per_class_metrics[c] = {
                        "precision": round(float(prec[idx]) * 100.0, 2),
                        "recall": round(float(rec[idx]) * 100.0, 2),
                        "f1": round(float(f1_vals[idx]) * 100.0, 2),
                        "support": int(sup[idx])
                    }
            except Exception:
                pass

        return {
            "session_id": self.session_id,
            "total_trials_processed": self.counters["total_processed"],
            "labeled_trials_count": self.counters["labeled_trials_count"],
            "classification_accuracy_pct": round(raw_acc, 2) if raw_acc is not None else None,
            "classification_errors_count": self.counters["classification_errors"],
            "macro_f1_pct": round(macro_f1, 2) if macro_f1 is not None else None,
            "per_class_metrics": per_class_metrics,
            "safety_rejection_rate_pct": round(rejection_rate, 2),
            "safety_rejected_count": self.counters["safety_rejected"],
            "safety_accepted_count": self.counters["safety_accepted"],
            "accepted_trial_accuracy_pct": round(accepted_acc, 2) if accepted_acc is not None else None,
            "command_execution_rate_pct": round(exec_rate, 2),
            "actions_executed_count": self.counters["actions_executed"],
            "actions_correct_count": self.counters["actions_correct"],
            "actions_incorrect_count": self.counters["actions_incorrect"],
            "incorrect_command_rate_total_pct": round(incorrect_rate_total, 2),
            "incorrect_command_rate_executed_pct": round(incorrect_rate_executed, 2),
            "shield_suppressed_errors_count": self.counters["suppressed_errors"],
            "shield_suppressed_correct_count": self.counters["suppressed_correct"],
            "rest_neutrality_rate_pct": round(rest_neutrality_rate, 2),
            "rest_neutral_trials_count": self.counters["rest_neutral_count"],
            "total_rest_trials_count": self.counters["rest_trials_count"],
            "stale_suppressed_count": self.counters["stale_suppressed"],
            "emergency_lock_suppressed_count": self.counters["emergency_lock_suppressed"],
            "latencies_ms": {
                "mean": round(mean_lat, 2),
                "p50": round(p50_lat, 2),
                "p95": round(p95_lat, 2),
                "p99": round(p99_lat, 2),
                "min": round(min_lat, 2),
                "max": round(max_lat, 2)
            },
            "metric_definitions": {
                "classification_accuracy": "Correct predictions / Labeled trials (offline test only)",
                "safety_rejection_rate": "Safety-rejected predictions / Total processed trials",
                "accepted_trial_accuracy": "Correct predictions within accepted / Safety-accepted trials",
                "command_execution_rate": "Executed actions / Total processed trials",
                "incorrect_command_rate_total": "Incorrectly executed actions / Total processed trials",
                "incorrect_command_rate_executed": "Incorrectly executed actions / Total executed actions",
                "rest_neutrality_rate": "True REST trials producing ZERO action / Total true REST trials"
            }
        }

    def export_audit_log(self, filepath: Optional[str] = None) -> str:
        """Exports audit session trail to structured JSON file."""
        sess_id = self.session_id or f"audit_{int(time.time())}"
        out_path = filepath or os.path.join(self.save_dir, f"{sess_id}_audit.json")

        summary = self.get_summary_metrics()
        export_payload = {
            "session_id": sess_id,
            "export_timestamp": time.time(),
            "export_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary_metrics": summary,
            "total_records": len(self.records),
            "records_count": len(self.records),
            "audit_records": [r.to_dict() for r in self.records]
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)

        return out_path

    def export_audit_csv(self, filepath: Optional[str] = None) -> str:
        """Exports flattened audit trail records to standard CSV file."""
        import csv
        sess_id = self.session_id or f"audit_{int(time.time())}"
        out_path = filepath or os.path.join(self.save_dir, f"{sess_id}_audit.csv")

        fieldnames = [
            "trial_id", "session_id", "timestamp", "iso_time", "subject_id", "run_id", "trial_idx",
            "data_source", "model_version", "ground_truth_label", "predicted_label", "confidence",
            "prob_REST", "prob_LEFT_FIST", "prob_RIGHT_FIST", "is_correct_classification",
            "signal_quality_passed", "signal_quality_status", "safety_accepted", "safety_intent",
            "safety_rejection_reason", "command_profile_id", "mapped_command", "action_executed",
            "action_name", "is_incorrect_action", "state_parameter", "state_delta", "state_unit",
            "ml_latency_ms", "pipeline_latency_ms"
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                row = {
                    "trial_id": r.trial_id,
                    "session_id": r.session_id,
                    "timestamp": r.timestamp,
                    "iso_time": r.iso_time,
                    "subject_id": r.subject_id or "UNKNOWN",
                    "run_id": r.run_id or "UNKNOWN",
                    "trial_idx": r.trial_idx if r.trial_idx is not None else -1,
                    "data_source": r.data_source,
                    "model_version": r.model_version,
                    "ground_truth_label": r.ground_truth_label or "UNLABELED",
                    "predicted_label": r.predicted_label,
                    "confidence": r.confidence,
                    "prob_REST": r.probabilities.get("REST", 0.0),
                    "prob_LEFT_FIST": r.probabilities.get("LEFT_FIST", 0.0),
                    "prob_RIGHT_FIST": r.probabilities.get("RIGHT_FIST", 0.0),
                    "is_correct_classification": r.is_correct_classification if r.is_correct_classification is not None else "",
                    "signal_quality_passed": r.signal_quality_passed,
                    "signal_quality_status": r.signal_quality_status,
                    "safety_accepted": r.safety_accepted,
                    "safety_intent": r.safety_intent,
                    "safety_rejection_reason": r.safety_rejection_reason or "NONE",
                    "command_profile_id": r.command_profile_id,
                    "mapped_command": r.mapped_command,
                    "action_executed": r.action_executed,
                    "action_name": r.action_name,
                    "is_incorrect_action": r.is_incorrect_action if r.is_incorrect_action is not None else "",
                    "state_parameter": r.state_change.get("parameter", "none"),
                    "state_delta": r.state_change.get("delta", 0),
                    "state_unit": r.state_change.get("unit", ""),
                    "ml_latency_ms": r.latencies_ms.get("ml_latency", 0.0),
                    "pipeline_latency_ms": r.latencies_ms.get("pipeline_latency", 0.0)
                }
                writer.writerow(row)

        return out_path

    def load_audit_log(self, filepath: str) -> Dict[str, Any]:
        """Loads historical audit log for inspection or deterministic replay."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
