from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from src.channel_mapper import ChannelMapper, EXPECTED_GROUP_B_CHANNELS


DEFAULT_OPENBCI_CYTON_DAISY_CHANNELS = [
    "FC3", "FC1", "FCZ", "FC2", "FC4",
    "C3",  "C1",  "CZ",  "C2",  "C4",
    "CP3", "CP1", "CPZ", "CP2", "CP4",
    "PZ"
]


@dataclass
class OpenBCICytonDaisyProfile:
    """
    Hardware Profile for OpenBCI Cyton Board + Daisy Expansion Module (16-Channel EEG System).
    Targeted specifically for Motor-Imagery EEG acquisition matching Step 17B FBCSP requirements.
    """
    device_name: str = "OpenBCI Cyton + Daisy"
    expected_channels_count: int = 16
    expected_use: str = "Motor-imagery EEG"
    hardware_sampling_rate_hz: float = 125.0 # Native Cyton+Daisy 16-channel streaming rate over Bluetooth/Dongle
    target_model_sampling_rate_hz: float = 160.0 # Step 17B FBCSP target rate
    
    # 15 Group B Motor Strip Channels required by Step 17B model
    required_model_channels: List[str] = field(default_factory=lambda: list(EXPECTED_GROUP_B_CHANNELS))
    
    default_device_channels: List[str] = field(default_factory=lambda: list(DEFAULT_OPENBCI_CYTON_DAISY_CHANNELS))
    
    default_reference: str = "Ear Clips (A1/A2)"
    default_ground: str = "BIAS Pin"
    default_montage: str = "10-20 Standard Cap System"

    def validate_montage(self, connected_channels: List[str]) -> Dict:
        """
        Validates whether the connected electrode montage supplies all 15 required motor channels.
        Returns:
            dict: Validation results including status (PASS/INCOMPATIBLE), missing channels, mapped indices.
        """
        mapper = ChannelMapper(target_channels=self.required_model_channels)
        val_res = mapper.validate_channels(connected_channels)

        status = "PASS" if val_res["is_compatible"] else "INCOMPATIBLE"

        return {
            "device_name": self.device_name,
            "status": status,
            "is_compatible": val_res["is_compatible"],
            "available_channels_count": len(connected_channels),
            "required_channels_count": len(self.required_model_channels),
            "missing_channels": val_res["missing_channels"],
            "mapped_indices": val_res["mapped_indices"],
            "requires_resampling": True,
            "original_sampling_rate_hz": self.hardware_sampling_rate_hz,
            "target_sampling_rate_hz": self.target_model_sampling_rate_hz
        }

    def get_explicit_channel_mapping(self, connected_channels: List[str] = None) -> List[Dict]:
        """
        Returns an explicit mapping table: Model Channel -> Device Channel -> Physical Position -> Status.
        """
        chans = connected_channels if connected_channels else self.default_device_channels
        mapper = ChannelMapper(target_channels=self.required_model_channels)
        norm_available = [mapper.normalize_channel_name(c) for c in chans]

        mapping_table = []
        for target in self.required_model_channels:
            norm_target = mapper.normalize_channel_name(target)
            if norm_target in norm_available:
                idx = norm_available.index(norm_target)
                device_ch_name = chans[idx]
                mapping_table.append({
                    "model_channel": target,
                    "device_channel": device_ch_name,
                    "device_channel_index": idx + 1,
                    "physical_position": f"{target} Motor Cortex Electrode",
                    "status": "PASS"
                })
            else:
                mapping_table.append({
                    "model_channel": target,
                    "device_channel": "MISSING",
                    "device_channel_index": -1,
                    "physical_position": f"{target} Electrode Not Connected",
                    "status": "FAIL"
                })

        return mapping_table
