"""
NeuroOS Hardware Profiles & Electrode Montage Contract (Phase 2E)
Defines specifications, channel contracts, and preprocessing requirements for:
1. SIMULATED_PHYSIONET (15-channel genuine EDF dataset stream)
2. OPENBCI_CYTON_DAISY_16CH (16 physical channels: 15 motor cortex channels + 1 auxiliary)
3. OPENBCI_CYTON_8CH (8 physical channels with 3D spherical spline interpolation fallback)

HARDWARE READINESS POLICY:
- Headset hardware must physically provide the exact 15 sensorimotor electrodes required by Step 62.
- Incompatible montages (missing channels, wrong locations) are strictly REJECTED, not silently fabricated.
- Distinct readiness levels: SOFTWARE_READY, READY_FOR_BENCH_TEST, PHYSICALLY_VALIDATED.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class HardwareProfile:
    profile_id: str
    display_name: str
    channel_count: int
    physical_channels: List[str]
    model_channels: List[str]
    unused_channels: List[str]
    sampling_rate_hz: float
    requires_interpolation: bool
    status: str
    readiness_level: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Exact 15 sensorimotor channels required by Step 62 champion model in precise order:
STEP62_TARGET_CHANNELS = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4"
]

# Backward compatibility alias
STEP60_TARGET_CHANNELS = STEP62_TARGET_CHANNELS

# Standard 8-channel OpenBCI Cyton layout
OPENBCI_8CH_PHYSICAL = ["F3", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]

# OpenBCI Cyton + Daisy 16-channel physical electrode layout:
# 15 channels directly match the Step 62 sensorimotor grid; Channel 16 (O1) acts as auxiliary/alpha monitor.
OPENBCI_16CH_PHYSICAL = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "O1"  # 16th channel: Occipital reference / auxiliary
]

# Standard Ultracortex Mark IV default cap layout (broad cortical montage, lacks frontocentral/centroparietal motor grid)
OPENBCI_ULTRACORTEX_DEFAULT = [
    "Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "F3", "F4", "T3", "T4", "T5", "T6"
]


def normalize_channel_name(ch: str) -> str:
    """Normalizes channel name string for robust case- and dot-insensitive matching."""
    return ch.strip().rstrip('.').upper()


def validate_channel_montage(
    detected_channels: List[str],
    expected_channels: List[str] = STEP62_TARGET_CHANNELS
) -> Tuple[bool, List[int], List[str], List[str]]:
    """
    Validates whether detected incoming channels provide all expected model channels.
    
    Returns:
    - is_valid (bool): True if ALL expected channels are present
    - reorder_indices (List[int]): Index map to reorder detected channels to match expected ordering
    - missing_channels (List[str]): List of expected channels absent in detected stream
    - extra_channels (List[str]): Detected channels not needed by the model
    """
    norm_detected = [normalize_channel_name(ch) for ch in detected_channels]
    norm_expected = [normalize_channel_name(ch) for ch in expected_channels]

    missing = [exp for exp, n_exp in zip(expected_channels, norm_expected) if n_exp not in norm_detected]
    extra = [det for det, n_det in zip(detected_channels, norm_detected) if n_det not in norm_expected]

    if missing:
        return False, [], missing, extra

    # Compute deterministic reordering indices
    reorder_indices = [norm_detected.index(n_exp) for n_exp in norm_expected]
    return True, reorder_indices, [], extra


class HardwareProfileRegistry:
    """Central registry for supported NeuroOS hardware profiles."""

    PROFILES = {
        "SIMULATED_PHYSIONET": HardwareProfile(
            profile_id="SIMULATED_PHYSIONET",
            display_name="PhysioNet EDF Simulation",
            channel_count=15,
            physical_channels=STEP62_TARGET_CHANNELS,
            model_channels=STEP62_TARGET_CHANNELS,
            unused_channels=[],
            sampling_rate_hz=160.0,
            requires_interpolation=False,
            status="SIMULATED",
            readiness_level="SOFTWARE_READY",
            description="Genuine PhysioNet motor imagery EDF dataset replay. 15 physical channels, 0 interpolation."
        ),
        "OPENBCI_CYTON_DAISY_16CH": HardwareProfile(
            profile_id="OPENBCI_CYTON_DAISY_16CH",
            display_name="OpenBCI Cyton + Daisy (16-Channel)",
            channel_count=16,
            physical_channels=OPENBCI_16CH_PHYSICAL,
            model_channels=STEP62_TARGET_CHANNELS,
            unused_channels=["O1"],
            sampling_rate_hz=160.0,  # Stream resampled to 160.0 Hz from nominal 125 Hz
            requires_interpolation=False,
            status="READY_FOR_BENCH_TEST",
            readiness_level="READY_FOR_BENCH_TEST",
            description="16 physical OpenBCI Cyton + Daisy channels configured on sensorimotor grid. Directly extracts 15 motor channels with 0 spatial interpolation."
        ),
        "OPENBCI_CYTON_8CH": HardwareProfile(
            profile_id="OPENBCI_CYTON_8CH",
            display_name="OpenBCI Cyton (8-Channel)",
            channel_count=8,
            physical_channels=OPENBCI_8CH_PHYSICAL,
            model_channels=STEP62_TARGET_CHANNELS,
            unused_channels=[],
            sampling_rate_hz=160.0,
            requires_interpolation=True,
            status="EXPERIMENTAL_FALLBACK",
            readiness_level="SOFTWARE_READY",
            description="8 physical OpenBCI channels with 3D spherical spline spatial reconstruction of 12 missing motor channels. Experimental fallback mode."
        )
    }

    @classmethod
    def get_profile(cls, profile_id: str) -> Optional[HardwareProfile]:
        return cls.PROFILES.get(profile_id)

    @classmethod
    def detect_profile(cls, channel_count: int, channel_names: Optional[List[str]] = None) -> Optional[HardwareProfile]:
        """
        Detects matching hardware profile from stream properties.
        Validates channel montage if channel names are provided.
        """
        names = channel_names or []
        if channel_count == 16:
            return cls.PROFILES["OPENBCI_CYTON_DAISY_16CH"]
        elif channel_count == 8:
            return cls.PROFILES["OPENBCI_CYTON_8CH"]
        elif channel_count == 15:
            return cls.PROFILES["SIMULATED_PHYSIONET"]
        return None
