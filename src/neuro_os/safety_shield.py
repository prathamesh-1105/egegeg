import numpy as np
from typing import Dict, Any, List, Optional


class NeuroOSSafetyShield:
    """
    NeuroOS Safety & Artifact Guardrail Shield.
    Filters raw Step 60 model probability predictions to ensure that noisy single-window
    spikes or low-confidence predictions never trigger accidental OS actions.
    
    Safety Features:
    1. Hardware Connection & Signal Quality Guard (SQI check)
    2. Causal Temporal Probability Smoothing (moving window average over N predictions)
    3. Confidence & Margin Thresholding (max_prob >= 0.60, entropy <= 1.30)
    4. Continuous REST Emergency Disengage (Auto-locks system into safe state after 3.0s continuous REST)
    """

    CLASSES = ["REST", "LEFT_FIST", "RIGHT_FIST"]

    def __init__(
        self,
        confidence_thresh: float = 0.60,
        margin_thresh: float = 0.12,
        entropy_thresh: float = 1.30,
        smoothing_window_size: int = 3,
        rest_emergency_windows: int = 5
    ):
        self.confidence_thresh = confidence_thresh
        self.margin_thresh = margin_thresh
        self.entropy_thresh = entropy_thresh
        self.smoothing_window_size = smoothing_window_size
        self.rest_emergency_windows = rest_emergency_windows

        self.prob_history: List[np.ndarray] = []
        self.consecutive_rest_count: int = 0
        self.system_emergency_locked: bool = False
        self.manual_override_lock: bool = False

    def reset(self):
        """Resets safety shield buffers and locks."""
        self.prob_history = []
        self.consecutive_rest_count = 0
        self.system_emergency_locked = False
        self.manual_override_lock = False

    @staticmethod
    def calculate_entropy(probs: np.ndarray) -> float:
        """Calculates Shannon entropy H = -sum(p * log2(p))."""
        probs_clean = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(probs_clean * np.log2(probs_clean)))

    def set_manual_lock(self, locked: bool):
        """Sets manual emergency override lock."""
        self.manual_override_lock = locked

    def evaluate_safety(
        self,
        raw_probs: Dict[str, float],
        hardware_connected: bool = True,
        signal_quality_ok: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates a raw probability prediction dictionary against all safety rules.
        Returns safety assessment dictionary with status, smoothed probabilities, and explainable reason.
        """
        # Validate probability vector input
        if not isinstance(raw_probs, dict) or not all(c in raw_probs for c in self.CLASSES):
            return {
                "accepted": False,
                "intent": "REST",
                "confidence": 0.0,
                "margin": 0.0,
                "entropy": 1.58,
                "smoothed_probs": {"REST": 1.0, "LEFT_FIST": 0.0, "RIGHT_FIST": 0.0},
                "emergency_locked": False,
                "reason": "MALFORMED_PROBABILITY_INPUT (Missing required class keys)"
            }

        try:
            p_vals = [float(raw_probs[c]) for c in self.CLASSES]
            if not all(np.isfinite(v) and v >= 0.0 for v in p_vals):
                return {
                    "accepted": False,
                    "intent": "REST",
                    "confidence": 0.0,
                    "margin": 0.0,
                    "entropy": 1.58,
                    "smoothed_probs": {"REST": 1.0, "LEFT_FIST": 0.0, "RIGHT_FIST": 0.0},
                    "emergency_locked": False,
                    "reason": "NON_FINITE_OR_NEGATIVE_PROBABILITIES_DETECTED"
                }
            p_vec = np.array(p_vals, dtype=float)
            total = float(p_vec.sum())
            if total > 0.0:
                p_vec /= total
            else:
                p_vec = np.array([1/3, 1/3, 1/3], dtype=float)
        except (ValueError, TypeError) as e:
            return {
                "accepted": False,
                "intent": "REST",
                "confidence": 0.0,
                "margin": 0.0,
                "entropy": 1.58,
                "smoothed_probs": {"REST": 1.0, "LEFT_FIST": 0.0, "RIGHT_FIST": 0.0},
                "emergency_locked": False,
                "reason": f"PROBABILITY_PARSING_FAILED ({str(e)})"
            }

        # 1. Manual Lock Check
        if self.manual_override_lock:
            return {
                "accepted": False,
                "intent": "REST",
                "confidence": 1.0,
                "margin": 1.0,
                "entropy": 0.0,
                "smoothed_probs": {"REST": 1.0, "LEFT_FIST": 0.0, "RIGHT_FIST": 0.0},
                "emergency_locked": True,
                "reason": "SAFETY_SHIELD_MANUAL_LOCK_ACTIVE"
            }

        # 2. Hardware Connection Check
        if not hardware_connected:
            return {
                "accepted": False,
                "intent": "REST",
                "confidence": 0.0,
                "margin": 0.0,
                "entropy": 1.58,
                "smoothed_probs": dict(zip(self.CLASSES, p_vec)),
                "emergency_locked": False,
                "reason": "HARDWARE_DISCONNECTED"
            }

        # 3. Signal Quality Index Check
        if not signal_quality_ok:
            return {
                "accepted": False,
                "intent": "REST",
                "confidence": 0.0,
                "margin": 0.0,
                "entropy": 1.58,
                "smoothed_probs": dict(zip(self.CLASSES, p_vec)),
                "emergency_locked": False,
                "reason": "POOR_SIGNAL_QUALITY_ARTIFACT_DETECTED"
            }

        # 4. Temporal Probability Smoothing
        self.prob_history.append(p_vec)
        if len(self.prob_history) > self.smoothing_window_size:
            self.prob_history.pop(0)

        smoothed_p = np.mean(self.prob_history, axis=0)
        smoothed_dict = dict(zip(self.CLASSES, [round(float(v), 4) for v in smoothed_p]))

        top_idx = int(np.argmax(smoothed_p))
        top_intent = self.CLASSES[top_idx]
        max_prob = float(smoothed_p[top_idx])

        sorted_p = np.sort(smoothed_p)[::-1]
        margin = float(sorted_p[0] - sorted_p[1])
        entropy = self.calculate_entropy(smoothed_p)

        # 5. Continuous REST Emergency Tracking
        if top_intent == "REST":
            self.consecutive_rest_count += 1
            if self.consecutive_rest_count >= self.rest_emergency_windows:
                self.system_emergency_locked = True
        else:
            self.consecutive_rest_count = 0
            self.system_emergency_locked = False

        # 6. Confidence & Entropy Threshold Evaluation
        if max_prob < self.confidence_thresh:
            return {
                "accepted": False,
                "intent": "REST",
                "confidence": round(max_prob, 4),
                "margin": round(margin, 4),
                "entropy": round(entropy, 4),
                "smoothed_probs": smoothed_dict,
                "emergency_locked": self.system_emergency_locked,
                "reason": f"LOW_CONFIDENCE ({max_prob:.2f} < {self.confidence_thresh:.2f})"
            }

        if margin < self.margin_thresh:
            return {
                "accepted": False,
                "intent": "REST",
                "confidence": round(max_prob, 4),
                "margin": round(margin, 4),
                "entropy": round(entropy, 4),
                "smoothed_probs": smoothed_dict,
                "emergency_locked": self.system_emergency_locked,
                "reason": f"LOW_CLASS_MARGIN ({margin:.2f} < {self.margin_thresh:.2f})"
            }

        if entropy > self.entropy_thresh:
            return {
                "accepted": False,
                "intent": "REST",
                "confidence": round(max_prob, 4),
                "margin": round(margin, 4),
                "entropy": round(entropy, 4),
                "smoothed_probs": smoothed_dict,
                "emergency_locked": self.system_emergency_locked,
                "reason": f"HIGH_ENTROPY_UNCERTAINTY ({entropy:.2f} > {self.entropy_thresh:.2f})"
            }

        # Prediction Accepted Safely
        return {
            "accepted": True,
            "intent": top_intent,
            "confidence": round(max_prob, 4),
            "margin": round(margin, 4),
            "entropy": round(entropy, 4),
            "smoothed_probs": smoothed_dict,
            "emergency_locked": self.system_emergency_locked,
            "reason": "SAFETY_CHECKS_PASSED"
        }
