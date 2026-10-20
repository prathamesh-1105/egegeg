import time
from typing import Dict, Any, List, Optional
from src.neuro_os.command_profiles import (
    CommandProfile,
    COMMAND_PROFILES,
    get_command_profile,
    DEFAULT_PROFILE_ID
)


class NeuroOSCommandRouter:
    """
    NeuroOS Intent-to-Command Router & Multi-Profile Dispatch Engine (Phase 2D).
    
    Translates safety-verified brain intents into high-level BCI software commands
    according to configurable application profiles:
    - Profile A — Scrolling (Default): LEFT_FIST -> SCROLL_UP, RIGHT_FIST -> SCROLL_DOWN, REST -> NEUTRAL
    - Profile B — Document Navigation: LEFT_FIST -> PREVIOUS_SECTION, RIGHT_FIST -> NEXT_SECTION, REST -> NEUTRAL
    - Profile C — Presentation Demo: LEFT_FIST -> PREVIOUS_SLIDE, RIGHT_FIST -> NEXT_SLIDE, REST -> NEUTRAL

    Safety & Guardrail Features:
    1. REST Neutrality: REST is mathematically constrained to NEUTRAL across all profiles.
    2. Stale Prediction Guard: Deduplicates prediction IDs / timestamps to prevent duplicate triggers from stale data.
    3. Configurable Cooldown Guardrail: Enforces minimum time delta between consecutive action triggers.
    4. Optional Temporal Confirmation: Requires N consecutive consistent intent windows before triggering.
    5. Repeat Suppression: Prevents accidental rapid repeating of the same command.
    6. Complete Command Audit Trail: Maintains chronological record of routing decisions.
    """

    COMMAND_MAPPING = {
        "REST": "NEUTRAL",
        "LEFT_FIST": "PREVIOUS",
        "RIGHT_FIST": "NEXT",
        "SUSTAINED_LEFT": "SELECT",
        "SUSTAINED_RIGHT": "BACK"
    }

    def __init__(
        self,
        dwell_windows: int = 3,
        cooldown_sec: float = 0.8,
        repeat_suppression: bool = True,
        max_history: int = 50,
        command_mode: str = "SCROLL",
        profile_id: str = "scrolling",
        temporal_confirmation_windows: int = 1,
        stale_timeout_sec: float = 3.0
    ):
        self.dwell_windows = dwell_windows
        self.cooldown_sec = cooldown_sec
        self.repeat_suppression = repeat_suppression
        self.max_history = max_history
        self.command_mode = command_mode
        self.stale_timeout_sec = stale_timeout_sec

        # Profile Configuration (Default: Profile A — Scrolling)
        self.active_profile = get_command_profile(profile_id or command_mode)
        self.profile_id = self.active_profile.id

        # Temporal Confirmation Configuration
        self.temporal_confirmation_windows = max(1, int(temporal_confirmation_windows))

        # Internal State Tracking
        self.consecutive_left_count = 0
        self.consecutive_right_count = 0
        self.consecutive_intent_count = 0
        self.active_intent: Optional[str] = None
        self.last_action_time = 0.0
        self.last_fired_command = "NEUTRAL"
        self.command_history: List[Dict[str, Any]] = []

        # Command Release & Overlap Protection
        self.command_release_required = False
        self.command_released = True

        # Stale Prediction Protection
        self.last_processed_prediction_id: Optional[str] = None
        self.last_prediction_timestamp: float = 0.0

    def set_profile(self, profile_id: str) -> CommandProfile:
        """
        Switches the active command profile cleanly without corrupting router or session state.
        Resets transient dwell/confirmation counters for the new profile.
        """
        self.active_profile = get_command_profile(profile_id)
        self.profile_id = self.active_profile.id
        self.command_mode = "SCROLL" if self.profile_id == "scrolling" else self.profile_id.upper()

        # Reset transient intent counters to avoid cross-profile leakage
        self.consecutive_left_count = 0
        self.consecutive_right_count = 0
        self.consecutive_intent_count = 0
        self.active_intent = None

        return self.active_profile

    def set_command_mode(self, mode: str):
        """
        Backwards-compatible mode switcher.
        Maps mode strings (SCROLL, DEFAULT, etc.) to canonical profiles.
        """
        mode_clean = str(mode).upper()
        if mode_clean in ["SCROLL", "SCROLL_DEMO"]:
            self.set_profile("scrolling")
        elif mode_clean in ["DEFAULT", "NAVIGATION"]:
            self.set_profile("scrolling")
        elif mode_clean in ["DOCUMENTS", "DOCUMENT", "DOC_NAV"]:
            self.set_profile("document_navigation")
        elif mode_clean in ["PRESENTATION", "SLIDES"]:
            self.set_profile("presentation")
        else:
            self.set_profile(mode.lower())

    def set_temporal_confirmation_windows(self, windows: int):
        """Configures optional temporal confirmation window count."""
        self.temporal_confirmation_windows = max(1, int(windows))

    def reset_confirmation(self):
        """
        Resets temporal confirmation counters, dwell hold, and restores command release state.
        Called on REST, invalid signal quality, disconnect, stale input, pause, or emergency lock.
        """
        self.consecutive_left_count = 0
        self.consecutive_right_count = 0
        self.consecutive_intent_count = 0
        self.active_intent = None
        self.command_released = True

    def reset(self):
        """Resets command router state."""
        self.reset_confirmation()
        self.last_action_time = 0.0
        self.last_fired_command = "NEUTRAL"
        self.last_processed_prediction_id = None
        self.last_prediction_timestamp = 0.0
        self.command_history = []

    def process_intent(
        self,
        safety_result: Dict[str, Any],
        current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes a safety-shielded intent result and routes it according to active profile.
        Enforces stale prediction detection, REST neutrality, cooldown, and temporal confirmation.
        Returns command routing decision dictionary.
        """
        now = current_time if current_time is not None else time.time()
        accepted = safety_result.get("accepted", False)
        intent = safety_result.get("intent", "REST")
        confidence = safety_result.get("confidence", 0.0)
        prediction_id = safety_result.get("prediction_id") or safety_result.get("event_id")
        pred_timestamp = safety_result.get("timestamp")
        is_stale = safety_result.get("is_stale", False)

        # 1. Stale Prediction Guardrail
        if is_stale:
            return {
                "command": "NEUTRAL",
                "action_triggered": False,
                "profile_id": self.profile_id,
                "dwell_progress": 0.0,
                "confirmation_progress": 0.0,
                "reason": "STALE_PREDICTION_SUPPRESSED (Flagged stale)"
            }

        if prediction_id and prediction_id == self.last_processed_prediction_id:
            return {
                "command": "NEUTRAL",
                "action_triggered": False,
                "profile_id": self.profile_id,
                "dwell_progress": 0.0,
                "confirmation_progress": 0.0,
                "reason": f"STALE_DUPLICATE_PREDICTION_SUPPRESSED (Duplicate ID: {prediction_id})"
            }

        if pred_timestamp and (now - pred_timestamp) > self.stale_timeout_sec:
            return {
                "command": "NEUTRAL",
                "action_triggered": False,
                "profile_id": self.profile_id,
                "dwell_progress": 0.0,
                "confirmation_progress": 0.0,
                "reason": f"STALE_TIMESTAMP_EXPIRED ({now - pred_timestamp:.2f}s > {self.stale_timeout_sec}s)"
            }

        # 2. Safety Rejection or REST Neutrality Check
        # REST is strictly neutral across all profiles; rejected predictions trigger no action
        if not accepted or intent == "REST":
            self.consecutive_left_count = 0
            self.consecutive_right_count = 0
            self.consecutive_intent_count = 0
            self.active_intent = None
            self.command_released = True
            if prediction_id:
                self.last_processed_prediction_id = prediction_id

            return {
                "command": "NEUTRAL",
                "action_triggered": False,
                "profile_id": self.profile_id,
                "dwell_progress": 0.0,
                "confirmation_progress": 0.0,
                "reason": "REST_NEUTRAL_IDLE" if intent == "REST" else f"SAFETY_REJECTED ({safety_result.get('reason', '')})"
            }

        # 3. Track Consecutive Windows for Active Intent
        if intent == self.active_intent:
            self.consecutive_intent_count += 1
        else:
            self.active_intent = intent
            self.consecutive_intent_count = 1
            self.command_released = True

        if intent == "LEFT_FIST":
            self.consecutive_left_count += 1
            self.consecutive_right_count = 0
        elif intent == "RIGHT_FIST":
            self.consecutive_right_count += 1
            self.consecutive_left_count = 0

        # Calculate dwell / confirmation progress
        confirmation_target = max(1, self.temporal_confirmation_windows)
        confirmation_progress = min(1.0, float(self.consecutive_intent_count) / float(confirmation_target))

        current_hold = max(self.consecutive_left_count, self.consecutive_right_count)
        dwell_progress = min(1.0, float(current_hold) / float(self.dwell_windows))

        # 4. Map Intent via Active Profile
        candidate_command = self.active_profile.map_intent(intent)

        # Legacy sustained command support for DEFAULT mode if dwell_windows met
        if self.command_mode == "DEFAULT" and self.profile_id == "scrolling":
            if self.consecutive_left_count >= self.dwell_windows:
                candidate_command = "SELECT"
            elif self.consecutive_right_count >= self.dwell_windows:
                candidate_command = "BACK"

        if candidate_command == "NEUTRAL":
            return {
                "command": "NEUTRAL",
                "action_triggered": False,
                "profile_id": self.profile_id,
                "dwell_progress": round(dwell_progress, 2),
                "confirmation_progress": round(confirmation_progress, 2),
                "reason": "MAPPED_TO_NEUTRAL"
            }

        # 5. Check Temporal Confirmation Constraint (Optional)
        if self.temporal_confirmation_windows > 1 and self.consecutive_intent_count < self.temporal_confirmation_windows:
            return {
                "command": candidate_command,
                "action_triggered": False,
                "profile_id": self.profile_id,
                "dwell_progress": round(dwell_progress, 2),
                "confirmation_progress": round(confirmation_progress, 2),
                "reason": f"TEMPORAL_CONFIRMATION_PENDING ({self.consecutive_intent_count}/{self.temporal_confirmation_windows} windows)"
            }

        # 6. Check Cooldown Constraint
        time_since_last = now - self.last_action_time
        if time_since_last < self.cooldown_sec:
            return {
                "command": candidate_command,
                "action_triggered": False,
                "profile_id": self.profile_id,
                "dwell_progress": round(dwell_progress, 2),
                "confirmation_progress": round(confirmation_progress, 2),
                "reason": f"COOLDOWN_ACTIVE ({time_since_last:.2f}s < {self.cooldown_sec:.2f}s)"
            }

        # 7. Check Repeat Command Suppression and Command-Release Behavior
        DISCRETE_STEP_COMMANDS = ["SELECT", "BACK", "NEXT_SLIDE", "PREV_SLIDE", "NEXT_SECTION", "PREV_SECTION"]
        if (self.command_release_required or self.repeat_suppression) and not self.command_released:
            if candidate_command in DISCRETE_STEP_COMMANDS and candidate_command == self.last_fired_command:
                return {
                    "command": candidate_command,
                    "action_triggered": False,
                    "profile_id": self.profile_id,
                    "dwell_progress": round(dwell_progress, 2),
                    "confirmation_progress": round(confirmation_progress, 2),
                    "reason": f"REPEAT_SUPPRESSED (Release to REST required before repeating {candidate_command})"
                }

        # 8. Command Validated & Action Triggered
        self.last_action_time = now
        self.last_fired_command = candidate_command
        self.command_released = False
        if prediction_id:
            self.last_processed_prediction_id = prediction_id
        self.last_prediction_timestamp = pred_timestamp or now

        # In multi-window confirmation mode, reset intent counter upon firing so that
        # overlapping sliding windows do not trigger multiple times on the same sustained cue.
        if self.temporal_confirmation_windows > 1:
            self.consecutive_intent_count = 0

        # Reset sustained counters after firing sustained command
        if candidate_command in ["SELECT", "BACK"]:
            self.consecutive_left_count = 0
            self.consecutive_right_count = 0

        cmd_record = {
            "timestamp": now,
            "iso_time": time.strftime("%H:%M:%S"),
            "profile_id": self.profile_id,
            "command": candidate_command,
            "intent": intent,
            "confidence": confidence,
            "dwell_progress": round(dwell_progress, 2),
            "confirmation_progress": 1.0
        }

        self.command_history.append(cmd_record)
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)

        return {
            "command": candidate_command,
            "action_triggered": True,
            "profile_id": self.profile_id,
            "dwell_progress": 1.0,
            "confirmation_progress": 1.0,
            "reason": f"COMMAND_VALIDATED ({candidate_command})"
        }
