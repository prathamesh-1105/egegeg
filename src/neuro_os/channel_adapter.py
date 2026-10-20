"""
NeuroOS OpenBCI 8-Channel to 15-Channel Sensorimotor Spatial Adapter
Reconstructs missing motor cortex electrodes from standard 8-channel OpenBCI Cyton EEG placement
using MNE spherical spline 3D spatial interpolation.

Input Channel Contract (8 physical channels):
  F3, F4, C3, Cz, C4, P3, Pz, P4

Output Channel Contract (15 Step60 model channels):
  FC3, FC1, FCz, FC2, FC4,
  C3, C1, Cz, C2, C4,
  CP3, CP1, CPz, CP2, CP4
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
import mne


class AdapterValidationError(Exception):
    """Raised when incoming hardware EEG signal violates input contract specifications."""
    pass


class OpenBCIChannelAdapter:
    """
    Hardware preprocessing adapter that converts 8-channel OpenBCI Cyton LSL stream
    into 15-channel sensorimotor EEG format required by Step60 model.
    """

    OPENBCI_8_CHANNELS = ["F3", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]
    STEP60_15_CHANNELS = [
        "FC3", "FC1", "FCz", "FC2", "FC4",
        "C3", "C1", "Cz", "C2", "C4",
        "CP3", "CP1", "CPz", "CP2", "CP4"
    ]

    def __init__(self, target_sfreq: float = 160.0, target_samples: int = 480):
        self.target_sfreq = target_sfreq
        self.target_samples = target_samples

        # All channels involved in 10-20 montage
        self.all_channels = list(dict.fromkeys(self.OPENBCI_8_CHANNELS + self.STEP60_15_CHANNELS))
        
        # Build MNE Info and standard 10-20 montage
        self.mne_info = mne.create_info(ch_names=self.all_channels, sfreq=self.target_sfreq, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        self.mne_info.set_montage(montage)

        self.missing_channels = [c for c in self.all_channels if c not in self.OPENBCI_8_CHANNELS]
        self._precompute_interpolation_weights()

    def _precompute_interpolation_weights(self):
        """
        Precomputes the 15x8 spherical spline transformation matrix W.
        Allows real-time interpolation in < 0.1ms via single matrix multiplication X_15 = W @ X_8.
        """
        # Create identity excitation on 8 physical channels
        eye_8 = np.eye(len(self.OPENBCI_8_CHANNELS))
        full_eye = np.zeros((len(self.all_channels), len(self.OPENBCI_8_CHANNELS)))

        for i, name in enumerate(self.all_channels):
            if name in self.OPENBCI_8_CHANNELS:
                src_idx = self.OPENBCI_8_CHANNELS.index(name)
                full_eye[i] = eye_8[src_idx]

        # Use MNE RawArray to compute spherical spline interpolation weights
        raw = mne.io.RawArray(full_eye, self.mne_info, verbose=False)
        raw.info['bads'] = self.missing_channels
        raw.interpolate_bads(reset_bads=False, verbose=False)
        interpolated_data = raw.get_data() # shape (len(all_channels), 8)

        # Extract rows corresponding to STEP60_15_CHANNELS in exact required order
        W = np.zeros((len(self.STEP60_15_CHANNELS), len(self.OPENBCI_8_CHANNELS)))
        for i, target_ch in enumerate(self.STEP60_15_CHANNELS):
            ch_idx = self.all_channels.index(target_ch)
            W[i] = interpolated_data[ch_idx]

        self.W_interp = W # shape (15, 8)

    def validate_input(
        self,
        data: np.ndarray,
        channel_names: List[str],
        sfreq: Optional[float] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Strict quality checks on input EEG window:
        1. Channel count must be exactly 8
        2. Channel names must match OPENBCI_8_CHANNELS (or be deterministically reordered)
        3. Signal must contain only finite numbers (no NaN or Inf)
        4. Sample count must meet target_samples
        """
        if not isinstance(data, np.ndarray):
            raise AdapterValidationError("Input data must be a numpy ndarray.")

        # Squeeze batch dimension if shape (1, 8, N) -> (8, N)
        if data.ndim == 3:
            if data.shape[0] == 1:
                data = data[0]
            else:
                raise AdapterValidationError(f"Expected single trial window, got batch shape {data.shape}.")

        if data.ndim != 2:
            raise AdapterValidationError(f"Expected 2D signal array (channels, samples), got shape {data.shape}.")

        n_channels, n_samples = data.shape

        if n_channels != len(self.OPENBCI_8_CHANNELS):
            raise AdapterValidationError(
                f"Invalid channel count: Expected {len(self.OPENBCI_8_CHANNELS)} OpenBCI channels, got {n_channels}."
            )

        if n_samples < self.target_samples:
            raise AdapterValidationError(
                f"Insufficient sample window: Expected at least {self.target_samples} samples, got {n_samples}."
            )

        if not np.all(np.isfinite(data)):
            raise AdapterValidationError("Signal quality error: Input contains NaN or infinite values.")

        # Validate channel names and build permutation index
        norm_names = [ch.strip().upper() for ch in channel_names]
        expected_names = [ch.upper() for ch in self.OPENBCI_8_CHANNELS]

        if sorted(norm_names) != sorted(expected_names):
            missing = set(expected_names) - set(norm_names)
            unexpected = set(norm_names) - set(expected_names)
            raise AdapterValidationError(
                f"Channel name mismatch! Expected {expected_names}. Missing: {missing}, Unexpected: {unexpected}."
            )

        # Deterministic reordering index
        reorder_idx = [norm_names.index(ch) for ch in expected_names]
        reordered_data = data[reorder_idx, :self.target_samples]

        return reordered_data, expected_names

    def transform(
        self,
        data: np.ndarray,
        channel_names: List[str],
        sfreq: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Transforms validated 8-channel OpenBCI signal into 15-channel sensorimotor EEG window.
        Returns dictionary containing transformed data (1, 15, 480) and detailed adapter telemetry.
        """
        t0 = time.perf_counter()

        try:
            clean_8ch_data, clean_names = self.validate_input(data, channel_names, sfreq)
            
            # Fast spherical spline linear transformation: (15, 8) @ (8, 480) -> (15, 480)
            data_15ch = self.W_interp @ clean_8ch_data # shape (15, 480)
            data_15ch_batched = np.expand_dims(data_15ch, axis=0) # shape (1, 15, 480)

            t1 = time.perf_counter()
            adapter_latency_ms = round((t1 - t0) * 1000.0, 3)

            return {
                "success": True,
                "data_15ch": data_15ch_batched,
                "channels": self.STEP60_15_CHANNELS,
                "sampling_rate_hz": self.target_sfreq,
                "telemetry": {
                    "input_channel_count": 8,
                    "output_channel_count": 15,
                    "input_channel_names": clean_names,
                    "output_channel_names": self.STEP60_15_CHANNELS,
                    "interpolation_active": True,
                    "adapter_latency_ms": adapter_latency_ms,
                    "adapter_status": "SPHERICAL_SPLINE_INTERPOLATION_SUCCESS",
                    "scientific_notice": "Reconstructed channels (FC3..CP4) are 3D spherical spline spatial estimates, not physical electrode measurements."
                }
            }
        except AdapterValidationError as ve:
            t1 = time.perf_counter()
            return {
                "success": False,
                "data_15ch": None,
                "channels": [],
                "sampling_rate_hz": self.target_sfreq,
                "telemetry": {
                    "input_channel_count": len(channel_names) if channel_names else 0,
                    "output_channel_count": 0,
                    "input_channel_names": channel_names,
                    "output_channel_names": [],
                    "interpolation_active": False,
                    "adapter_latency_ms": round((t1 - t0) * 1000.0, 3),
                    "adapter_status": "ADAPTER_REJECTED",
                    "error_detail": str(ve)
                }
            }
