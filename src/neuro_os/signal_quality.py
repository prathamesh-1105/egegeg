"""
NeuroOS Hardware Signal Quality Gate (Phase 2E)
Validates raw incoming EEG windows before ML decoder execution using configurable engineering thresholds:
- NaN / Inf value detection
- Flatline detection (variance < min_variance uV^2)
- Extreme amplitude bounds (min_amplitude_uv < amplitude < max_amplitude_uv)
- Sudden sample-to-sample gradient artifact detection (max_gradient_uv)
- Channel montage integrity & missing electrode detection
- Insufficient sample count validation
- Sampling rate compatibility check
- Timestamp staleness & stream disconnection detection

SCIENTIFIC DISCLAIMER:
- Amplitude limits are configurable engineering sanity checks, NOT universal clinical thresholds.
- Software amplitude checks DO NOT measure physical electrode impedance.
- True electrode impedance requires compatible hardware with active current stimulation (e.g. TI ADS1299 lead-off excitation).
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.neuro_os.hardware_profiles import (
    STEP62_TARGET_CHANNELS,
    validate_channel_montage,
    normalize_channel_name
)


class NeuroOSSignalQualityGate:
    """
    Quality gate component that inspects incoming EEG signal windows
    and produces explainable bad channel reports, signal metrics, and disengage warnings.
    """

    def __init__(
        self,
        min_variance: float = 1e-12,
        max_amplitude_uv: float = 500.0,
        min_amplitude_uv: float = 0.1,
        max_gradient_uv: float = 150.0,
        target_sfreq: float = 160.0,
        target_samples: int = 480,
        max_stale_age_sec: float = 2.5
    ):
        self.min_variance = min_variance
        self.max_amplitude_uv = max_amplitude_uv
        self.min_amplitude_uv = min_amplitude_uv
        self.max_gradient_uv = max_gradient_uv
        self.target_sfreq = target_sfreq
        self.target_samples = target_samples
        self.max_stale_age_sec = max_stale_age_sec

    def get_thresholds(self) -> Dict[str, Any]:
        """Returns currently active engineering thresholds."""
        return {
            "min_variance": self.min_variance,
            "max_amplitude_uv": self.max_amplitude_uv,
            "min_amplitude_uv": self.min_amplitude_uv,
            "max_gradient_uv": self.max_gradient_uv,
            "target_sfreq": self.target_sfreq,
            "target_samples": self.target_samples,
            "max_stale_age_sec": self.max_stale_age_sec
        }

    def set_thresholds(self, **kwargs):
        """Updates configurable engineering thresholds."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def evaluate_quality(
        self,
        data: Optional[np.ndarray],
        channel_names: List[str],
        sfreq: Optional[float] = None,
        timestamp: Optional[float] = None,
        expected_channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates EEG window quality and returns detailed diagnostic status.
        Data shape expected: (1, n_channels, n_samples) or (n_channels, n_samples).
        """
        bad_channels = []
        warnings = []
        channel_diagnostics = {}
        passed = True

        expected_ch = expected_channels or STEP62_TARGET_CHANNELS

        # 0. Check data existence
        if data is None or not isinstance(data, np.ndarray):
            return {
                "passed": False,
                "signal_quality": "REJECTED",
                "quality_status": "NO_DATA",
                "bad_channels": [],
                "quality_warning": "No EEG signal data provided (stream disconnected or empty buffer).",
                "impedance_measured": False,
                "channel_diagnostics": {},
                "notice": "Software checks are engineering sanity filters, not physical impedance measurements."
            }

        # 1. Staleness Check
        if timestamp is not None and timestamp > 0:
            age_sec = time.time() - timestamp
            if age_sec > self.max_stale_age_sec:
                passed = False
                warnings.append(f"Stale stream data detected ({age_sec:.2f}s > {self.max_stale_age_sec}s).")

        if data.ndim == 3:
            data = data[0]

        if data.ndim != 2:
            return {
                "passed": False,
                "signal_quality": "REJECTED",
                "quality_status": "INVALID_DIMENSIONS",
                "bad_channels": [],
                "quality_warning": f"Expected 2D array, got ndim={data.ndim}.",
                "impedance_measured": False,
                "channel_diagnostics": {},
                "notice": "Software checks are engineering sanity filters, not physical impedance measurements."
            }

        n_channels, n_samples = data.shape

        # 2. Finite Check (NaN / Inf)
        if not np.all(np.isfinite(data)):
            return {
                "passed": False,
                "signal_quality": "REJECTED",
                "quality_status": "NAN_INF_DETECTED",
                "bad_channels": list(channel_names),
                "quality_warning": "Signal contains invalid numerical values (NaN or Infinity).",
                "impedance_measured": False,
                "channel_diagnostics": {},
                "notice": "Software checks are engineering sanity filters, not physical impedance measurements."
            }

        # 3. Channel Montage Integrity Check
        norm_det = [normalize_channel_name(c) for c in channel_names]
        norm_exp = [normalize_channel_name(c) for c in expected_ch]
        missing = [exp for exp, nexp in zip(expected_ch, norm_exp) if nexp not in norm_det]

        if missing:
            passed = False
            warnings.append(f"Missing required model channels: {missing}.")

        # 4. Sample Count Check
        if n_samples < self.target_samples:
            passed = False
            warnings.append(f"Insufficient samples ({n_samples} < {self.target_samples}).")

        # 5. Sampling Rate Check
        if sfreq is not None and abs(sfreq - self.target_sfreq) > 5.0:
            warnings.append(f"Sampling rate mismatch: {sfreq}Hz vs target {self.target_sfreq}Hz.")

        # 6. Signal Scale Detection (uV vs V)
        peak_amp = float(np.max(np.abs(data))) if data.size > 0 else 0.0
        scale_to_uv = 1e6 if peak_amp < 0.05 else 1.0

        # 7. Per-channel Quality Check (Flatline, Amplitude Bounds, Sudden Artifacts)
        for idx, ch_name in enumerate(channel_names[:n_channels]):
            ch_signal_uv = data[idx] * scale_to_uv
            ch_var_uv = float(np.var(ch_signal_uv))
            max_amp_uv = float(np.max(np.abs(ch_signal_uv)))
            mean_amp_uv = float(np.mean(ch_signal_uv))

            # Sample-to-sample gradient (sudden pop artifact)
            diffs = np.abs(np.diff(ch_signal_uv))
            max_gradient = float(np.max(diffs)) if len(diffs) > 0 else 0.0

            ch_status = "OK"
            ch_reason = "Signal optimal"

            # Flatline check
            if ch_var_uv < self.min_variance:
                ch_status = "FLATLINE"
                ch_reason = f"Channel flatline detected (var={ch_var_uv:.2e} uV^2 < {self.min_variance:.2e})."
                bad_channels.append(ch_name)
                warnings.append(f"Channel {ch_name}: {ch_reason}")

            # Excessive amplitude artifact (> 500 uV)
            elif max_amp_uv > self.max_amplitude_uv:
                ch_status = "ARTIFACT_HIGH"
                ch_reason = f"High amplitude artifact ({max_amp_uv:.1f}uV > {self.max_amplitude_uv}uV)."
                bad_channels.append(ch_name)
                warnings.append(f"Channel {ch_name}: {ch_reason}")

            # Dead / Disconnected electrode (< 0.1 uV)
            elif max_amp_uv < self.min_amplitude_uv:
                ch_status = "DISCONNECTED"
                ch_reason = f"Electrode disconnected or zeroed (< {self.min_amplitude_uv}uV)."
                bad_channels.append(ch_name)
                warnings.append(f"Channel {ch_name}: {ch_reason}")

            # Sudden step jump / pop artifact
            elif max_gradient > self.max_gradient_uv:
                ch_status = "GRADIENT_POP"
                ch_reason = f"Sudden electrode pop artifact (gradient={max_gradient:.1f}uV > {self.max_gradient_uv}uV)."
                bad_channels.append(ch_name)
                warnings.append(f"Channel {ch_name}: {ch_reason}")

            channel_diagnostics[ch_name] = {
                "status": ch_status,
                "reason": ch_reason,
                "variance_uv2": round(ch_var_uv, 4),
                "peak_amplitude_uv": round(max_amp_uv, 2),
                "mean_uv": round(mean_amp_uv, 2),
                "max_gradient_uv": round(max_gradient, 2)
            }

        if len(bad_channels) > 0 or missing:
            quality_str = "WARNING" if len(bad_channels) <= 2 and not missing else "REJECTED"
            if len(bad_channels) > 2 or missing:
                passed = False
        else:
            quality_str = "EXCELLENT"

        return {
            "passed": passed,
            "signal_quality": quality_str,
            "quality_status": "OK" if passed else "QUALITY_CHECK_FAILED",
            "bad_channels": bad_channels,
            "missing_channels": missing,
            "quality_warning": " | ".join(warnings) if warnings else "Signal quality optimal.",
            "channel_diagnostics": channel_diagnostics,
            "impedance_measured": False,
            "notice": "Software checks are engineering sanity filters, not physical impedance measurements. Physical impedance requires hardware ADS1299 excitation."
        }
