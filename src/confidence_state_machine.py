import numpy as np
from typing import Dict, Any, List, Optional


class BCISafetyStateMachine:
    """
    Step 26 Confidence-Aware Safety State Machine with Abstention & Error Rejection.
    Ensures BCI never triggers motor actions during uncertain predictions, poor signal, or disconnects.
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

    def __init__(
        self,
        max_prob_thresh: float = 0.65,
        margin_thresh: float = 0.15,
        entropy_thresh: float = 1.20,
        use_temporal_smoothing: bool = True,
        buffer_len: int = 3
    ):
        self.max_prob_thresh = max_prob_thresh
        self.margin_thresh = margin_thresh
        self.entropy_thresh = entropy_thresh
        self.use_temporal_smoothing = use_temporal_smoothing
        self.buffer_len = buffer_len
        self.prob_buffer: List[np.ndarray] = []
        self.current_state = "DISCONNECTED"

    def reset(self):
        self.prob_buffer = []
        self.current_state = "DISCONNECTED"

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
        Process a single EEG window probability dictionary.
        Returns state machine decision, metrics, and state flags.
        """
        classes = ["REST", "LEFT_FIST", "RIGHT_FIST"]
        p_vec = np.array([raw_probs.get(c, 0.0) for c in classes], dtype=float)
        
        # Ensure proper normalization
        if p_vec.sum() > 0:
            p_vec /= p_vec.sum()
        else:
            p_vec = np.array([1/3, 1/3, 1/3])

        # 1. Hardware Connection Check
        if not hardware_connected:
            self.current_state = "DISCONNECTED"
            return {
                "current_state": "DISCONNECTED",
                "accepted": False,
                "top_prediction": "NONE",
                "max_prob": 0.0,
                "margin": 0.0,
                "entropy": 1.58,
                "smoothed_probs": dict(zip(classes, p_vec)),
                "reason": "Hardware disconnected"
            }

        # 2. Signal Quality Check
        if not signal_quality_ok:
            self.current_state = "SIGNAL_WARNING"
            return {
                "current_state": "SIGNAL_WARNING",
                "accepted": False,
                "top_prediction": "NONE",
                "max_prob": 0.0,
                "margin": 0.0,
                "entropy": 1.58,
                "smoothed_probs": dict(zip(classes, p_vec)),
                "reason": "Signal artifacts detected (RMS/Peak out of range)"
            }

        # 3. Causal Temporal Smoothing
        if self.use_temporal_smoothing:
            self.prob_buffer.append(p_vec)
            if len(self.prob_buffer) > self.buffer_len:
                self.prob_buffer.pop(0)
            eval_p = np.mean(self.prob_buffer, axis=0)
        else:
            eval_p = p_vec

        # Calculate metrics
        sorted_indices = np.argsort(eval_p)[::-1]
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1]

        top_pred = classes[top1_idx]
        max_prob = float(eval_p[top1_idx])
        margin = float(eval_p[top1_idx] - eval_p[top2_idx])
        entropy = self.calculate_entropy(eval_p)

        # 4. Abstention Rule Check
        cond_prob = max_prob >= self.max_prob_thresh
        cond_margin = margin >= self.margin_thresh
        cond_entropy = entropy <= self.entropy_thresh

        accepted = cond_prob and cond_margin and cond_entropy

        if accepted:
            self.current_state = top_pred
            reason = "Confidence criteria met"
        else:
            self.current_state = "UNCERTAIN"
            reasons = []
            if not cond_prob:
                reasons.append(f"Max Prob {max_prob:.2f} < {self.max_prob_thresh:.2f}")
            if not cond_margin:
                reasons.append(f"Margin {margin:.2f} < {self.margin_thresh:.2f}")
            if not cond_entropy:
                reasons.append(f"Entropy {entropy:.2f} > {self.entropy_thresh:.2f}")
            reason = "Abstained: " + ", ".join(reasons)

        return {
            "current_state": self.current_state,
            "accepted": accepted,
            "top_prediction": top_pred,
            "max_prob": max_prob,
            "margin": margin,
            "entropy": entropy,
            "smoothed_probs": dict(zip(classes, eval_p)),
            "reason": reason
        }
