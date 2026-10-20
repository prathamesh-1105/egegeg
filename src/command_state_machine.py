import numpy as np
from typing import Dict, Any, List, Optional


class BCICommandStateMachine:
    """
    Step 28 Real-Time Causal Command State Machine & Temporal Reliability Engine.
    Filters raw model predictions through:
    1. Hardware & Signal Quality Check
    2. Step 26 Confidence & Abstention Layer (P_max, Margin, Entropy)
    3. Causal Temporal Confirmation & Hysteresis Activation Rules
    """

    STATES = [
        "DISCONNECTED",
        "SIGNAL_WARNING",
        "WAITING",
        "REST",
        "LEFT_FIST",
        "RIGHT_FIST",
        "UNCERTAIN"
    ]

    STRATEGIES = [
        "NO_SMOOTHING",
        "2_CONSECUTIVE",
        "3_CONSECUTIVE",
        "2_OF_3_MAJORITY",
        "3_OF_5_CONFIRMATION",
        "PROB_ACCUMULATION",
        "HYSTERESIS"
    ]

    def __init__(
        self,
        strategy: str = "HYSTERESIS",
        max_prob_thresh: float = 0.70,
        margin_thresh: float = 0.05,
        entropy_thresh: float = 1.20,
        activation_thresh: float = 0.70,
        deactivation_thresh: float = 0.45,
        window_duration_sec: float = 3.0
    ):
        self.strategy = strategy if strategy in self.STRATEGIES else "HYSTERESIS"
        self.max_prob_thresh = max_prob_thresh
        self.margin_thresh = margin_thresh
        self.entropy_thresh = entropy_thresh
        self.activation_thresh = activation_thresh
        self.deactivation_thresh = deactivation_thresh
        self.window_duration_sec = window_duration_sec

        self.classes = ["REST", "LEFT_FIST", "RIGHT_FIST"]
        
        # State tracking
        self.current_state = "WAITING"
        self.raw_history: List[str] = []
        self.conf_history: List[str] = []
        self.prob_history: List[np.ndarray] = []
        
        # Leaky accumulation vector
        self.accumulated_probs = np.array([1/3, 1/3, 1/3], dtype=float)
        self.leak_factor = 0.6  # Memory retention factor

        # Duration & transition stats
        self.current_command_windows = 0
        self.total_windows_processed = 0

    def reset(self):
        """Reset state machine for a new EEG stream."""
        self.current_state = "WAITING"
        self.raw_history = []
        self.conf_history = []
        self.prob_history = []
        self.accumulated_probs = np.array([1/3, 1/3, 1/3], dtype=float)
        self.current_command_windows = 0
        self.total_windows_processed = 0

    @staticmethod
    def calculate_entropy(probs: np.ndarray) -> float:
        """Calculate Shannon entropy H = -sum(p * log2(p))."""
        probs_clean = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(probs_clean * np.log2(probs_clean)))

    def process_window(
        self,
        raw_probs: Dict[str, float],
        hardware_connected: bool = True,
        signal_quality_ok: bool = True
    ) -> Dict[str, Any]:
        """
        Processes a single chronological window's probabilities.
        Returns full diagnostic status and updated final command state.
        """
        self.total_windows_processed += 1

        p_vec = np.array([raw_probs.get(c, 0.0) for c in self.classes], dtype=float)
        if p_vec.sum() > 0:
            p_vec /= p_vec.sum()
        else:
            p_vec = np.array([1/3, 1/3, 1/3], dtype=float)

        raw_pred_idx = np.argmax(p_vec)
        raw_pred = self.classes[raw_pred_idx]

        self.prob_history.append(p_vec)
        self.raw_history.append(raw_pred)

        # -------------------------------------------------------------
        # 1. HARDWARE & SIGNAL SAFETY CHECKS
        # -------------------------------------------------------------
        if not hardware_connected:
            self._update_state("DISCONNECTED")
            return self._build_result(raw_pred, "DISCONNECTED", False, p_vec, "Hardware disconnected")

        if not signal_quality_ok:
            self._update_state("SIGNAL_WARNING")
            return self._build_result(raw_pred, "SIGNAL_WARNING", False, p_vec, "Signal artifact detected")

        # -------------------------------------------------------------
        # 2. STEP 26 CONFIDENCE LAYER
        # -------------------------------------------------------------
        max_prob = float(np.max(p_vec))
        sorted_p = np.sort(p_vec)[::-1]
        margin = float(sorted_p[0] - sorted_p[1])
        entropy = self.calculate_entropy(p_vec)

        cond_prob = max_prob >= self.max_prob_thresh
        cond_margin = margin >= self.margin_thresh
        cond_entropy = entropy <= self.entropy_thresh
        conf_accepted = cond_prob and cond_margin and cond_entropy

        conf_state = raw_pred if conf_accepted else "UNCERTAIN"
        self.conf_history.append(conf_state)

        # -------------------------------------------------------------
        # 3. TEMPORAL LOGIC & HYSTERESIS ACTIVATION RULE
        # -------------------------------------------------------------
        final_action = self._eval_temporal_logic(conf_accepted, conf_state, p_vec)
        self._update_state(final_action)

        return self._build_result(
            raw_pred=raw_pred,
            final_action=final_action,
            conf_accepted=conf_accepted,
            p_vec=p_vec,
            reason="Confirmed by temporal state machine" if conf_accepted else "Abstained / Unconfirmed"
        )

    def _eval_temporal_logic(self, conf_accepted: bool, conf_state: str, p_vec: np.ndarray) -> str:
        """Evaluate temporal strategy to decide final command state."""
        # Safety rule: Low confidence or UNCERTAIN immediately aborts or holds REST
        if not conf_accepted or conf_state == "UNCERTAIN":
            # If motor command active, de-escalate to REST/UNCERTAIN safely
            if self.current_state in ["LEFT_FIST", "RIGHT_FIST"]:
                return "REST"
            return "UNCERTAIN"

        if self.strategy == "NO_SMOOTHING":
            return conf_state

        elif self.strategy == "2_CONSECUTIVE":
            if len(self.conf_history) >= 2:
                if self.conf_history[-1] == self.conf_history[-2] and self.conf_history[-1] in self.classes:
                    return self.conf_history[-1]
            return self.current_state if self.current_state in self.classes else "REST"

        elif self.strategy == "3_CONSECUTIVE":
            if len(self.conf_history) >= 3:
                if (self.conf_history[-1] == self.conf_history[-2] == self.conf_history[-3]) and (self.conf_history[-1] in self.classes):
                    return self.conf_history[-1]
            return self.current_state if self.current_state in self.classes else "REST"

        elif self.strategy == "2_OF_3_MAJORITY":
            recent = [c for c in self.conf_history[-3:] if c in self.classes]
            if len(recent) >= 2:
                from collections import Counter
                counts = Counter(recent)
                most_common, count = counts.most_common(1)[0]
                if count >= 2:
                    return most_common
            return self.current_state if self.current_state in self.classes else "REST"

        elif self.strategy == "3_OF_5_CONFIRMATION":
            recent = [c for c in self.conf_history[-5:] if c in self.classes]
            if len(recent) >= 3:
                from collections import Counter
                counts = Counter(recent)
                most_common, count = counts.most_common(1)[0]
                if count >= 3:
                    return most_common
            return self.current_state if self.current_state in self.classes else "REST"

        elif self.strategy == "PROB_ACCUMULATION":
            self.accumulated_probs = self.leak_factor * self.accumulated_probs + (1.0 - self.leak_factor) * p_vec
            self.accumulated_probs /= self.accumulated_probs.sum()
            acc_top_idx = np.argmax(self.accumulated_probs)
            if self.accumulated_probs[acc_top_idx] >= self.activation_thresh:
                return self.classes[acc_top_idx]
            return self.current_state if self.current_state in self.classes else "REST"

        elif self.strategy == "HYSTERESIS":
            p_rest, p_left, p_right = p_vec[0], p_vec[1], p_vec[2]

            # If current state is REST or UNCERTAIN/WAITING, require activation threshold
            if self.current_state not in ["LEFT_FIST", "RIGHT_FIST"]:
                if p_left >= self.activation_thresh and p_left > p_right and p_left > p_rest:
                    return "LEFT_FIST"
                elif p_right >= self.activation_thresh and p_right > p_left and p_right > p_rest:
                    return "RIGHT_FIST"
                else:
                    return "REST"
            
            # If current state is LEFT_FIST
            elif self.current_state == "LEFT_FIST":
                if p_left < self.deactivation_thresh or p_rest > p_left or p_right >= self.activation_thresh:
                    if p_right >= self.activation_thresh:
                        return "RIGHT_FIST"
                    return "REST"
                return "LEFT_FIST"

            # If current state is RIGHT_FIST
            elif self.current_state == "RIGHT_FIST":
                if p_right < self.deactivation_thresh or p_rest > p_right or p_left >= self.activation_thresh:
                    if p_left >= self.activation_thresh:
                        return "LEFT_FIST"
                    return "REST"
                return "RIGHT_FIST"

        return conf_state

    def _update_state(self, new_state: str):
        """Update current state and duration counter."""
        if new_state == self.current_state:
            self.current_command_windows += 1
        else:
            self.current_state = new_state
            self.current_command_windows = 1

    def _build_result(
        self,
        raw_pred: str,
        final_action: str,
        conf_accepted: bool,
        p_vec: np.ndarray,
        reason: str
    ) -> Dict[str, Any]:
        """Format complete result payload."""
        sorted_p = np.sort(p_vec)[::-1]
        max_p = float(sorted_p[0])
        margin = float(sorted_p[0] - sorted_p[1])
        entropy = self.calculate_entropy(p_vec)

        smoothed_3w = np.mean(self.prob_history[-3:], axis=0) if len(self.prob_history) > 0 else p_vec
        smoothed_pred = self.classes[np.argmax(smoothed_3w)]

        return {
            "window_index": self.total_windows_processed - 1,
            "raw_prediction": raw_pred,
            "smoothed_prediction": smoothed_pred,
            "confidence_accepted": conf_accepted,
            "confidence_state": self.conf_history[-1] if self.conf_history else "UNCERTAIN",
            "final_action": final_action,
            "current_state": self.current_state,
            "probabilities": {
                "REST": float(p_vec[0]),
                "LEFT_FIST": float(p_vec[1]),
                "RIGHT_FIST": float(p_vec[2])
            },
            "max_probability": max_p,
            "margin": margin,
            "entropy": entropy,
            "command_duration_sec": round(self.current_command_windows * self.window_duration_sec, 2),
            "command_windows": self.current_command_windows,
            "reason": reason
        }
