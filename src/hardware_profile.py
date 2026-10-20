import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np

from src.channel_mapper import ChannelMapper, EXPECTED_GROUP_B_CHANNELS


@dataclass
class HardwareProfile:
    """
    Generic hardware profile adapter interface for plugging physical EEG devices into the
    Step 17B FBCSP real-time pipeline.
    """
    device_name: str
    channel_names: List[str]
    sampling_rate: float
    montage: str
    reference: str
    lsl_stream_type: str
    lsl_stream_name: str
    requires_resampling: bool
    target_sampling_rate: float = 160.0
    motor_channels: List[str] = field(default_factory=lambda: list(EXPECTED_GROUP_B_CHANNELS))
    compatibility_status: str = "UNKNOWN"
    notes: str = ""

    def validate_compatibility(self) -> Dict:
        """
        Validates hardware profile compatibility against Step 17B 15-channel motor strip requirements.
        """
        mapper = ChannelMapper(target_channels=self.motor_channels)
        val_res = mapper.validate_channels(self.channel_names)
        
        is_motor_pass = val_res["is_compatible"]
        resample_needed = (abs(self.sampling_rate - self.target_sampling_rate) >= 1.0)

        status = "PASS" if is_motor_pass else "INCOMPATIBLE"
        
        return {
            "device_name": self.device_name,
            "status": status,
            "motor_cortex_coverage": "PASS" if is_motor_pass else "FAIL",
            "available_channels_count": len(self.channel_names),
            "required_channels_count": len(self.motor_channels),
            "missing_channels": val_res["missing_channels"],
            "mapped_indices": val_res["mapped_indices"],
            "sampling_rate_hz": self.sampling_rate,
            "requires_resampling": resample_needed,
            "target_sampling_rate_hz": self.target_sampling_rate
        }

    def get_channel_mapping(self) -> List[Tuple[str, str]]:
        """
        Returns list of tuples mapping (Model Channel, Device Channel).
        """
        mapper = ChannelMapper(target_channels=self.motor_channels)
        norm_available = [mapper.normalize_channel_name(c) for c in self.channel_names]
        mapping = []
        for target in self.motor_channels:
            norm_target = mapper.normalize_channel_name(target)
            if norm_target in norm_available:
                idx = norm_available.index(norm_target)
                mapping.append((target, self.channel_names[idx]))
            else:
                mapping.append((target, "MISSING"))
        return mapping

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "HardwareProfile":
        return cls(**data)
