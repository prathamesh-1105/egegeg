import time
import json
from typing import Dict, Any, List, Optional


class NeuroOSEventLogger:
    """
    NeuroOS Telemetry & Explainable Audit Logger.
    Records structured JSON logs for all predictions, safety shield decisions,
    dwell confirmation states, and executed software actions.
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.logs: List[Dict[str, Any]] = []

    def log_event(
        self,
        event_type: str,
        intent: str,
        confidence: float,
        safety_status: Dict[str, Any],
        command: Optional[str] = None,
        action: Optional[Dict[str, Any]] = None,
        details: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates a structured, timestamped audit log entry.
        """
        entry = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%H:%M:%S"),
            "event_type": event_type,
            "raw_intent": intent,
            "confidence": round(confidence, 4),
            "safety_accepted": safety_status.get("accepted", False),
            "safety_reason": safety_status.get("reason", "OK"),
            "command": command or "NONE",
            "action_executed": action or {"name": "NONE", "executed": False},
            "details": details or ""
        }

        self.logs.append(entry)
        if len(self.logs) > self.max_history:
            self.logs.pop(0)

        return entry

    def get_recent_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns recent audit log entries."""
        return self.logs[-limit:]

    def clear(self):
        """Clears stored logs."""
        self.logs = []
