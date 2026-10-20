import re


EXPECTED_GROUP_B_CHANNELS = [
    "FC3", "FC1", "FCZ", "FC2", "FC4",
    "C3",  "C1",  "CZ",  "C2",  "C4",
    "CP3", "CP1", "CPZ", "CP2", "CP4"
]


class ChannelMapper:
    """
    Channel Normalization & Compatibility Validation Layer.
    Verifies that incoming EEG streams supply the required 15 Group B motor channels in correct order.
    """

    def __init__(self, target_channels: list = None):
        self.target_channels = target_channels if target_channels else EXPECTED_GROUP_B_CHANNELS

    @staticmethod
    def normalize_channel_name(ch_name: str) -> str:
        """
        Normalize channel name (e.g. 'c3' -> 'C3', 'C3.' -> 'C3', 'EEG C3-REF' -> 'C3').
        """
        clean = ch_name.strip().upper()
        clean = re.sub(r'[^A-Z0-9]', '', clean)
        clean = clean.replace("EEG", "").replace("REF", "")
        return clean

    def validate_channels(self, available_channels: list) -> dict:
        """
        Validate incoming channel names against target 15 Group B motor channels.

        Returns:
            dict: {
                "status": "COMPATIBLE" or "INCOMPATIBLE",
                "available_count": int,
                "target_count": int,
                "missing_channels": list,
                "mapped_indices": list,
                "is_compatible": bool
            }
        """
        norm_available = [self.normalize_channel_name(c) for c in available_channels]
        mapped_indices = []
        missing = []

        for target in self.target_channels:
            norm_target = self.normalize_channel_name(target)
            if norm_target in norm_available:
                idx = norm_available.index(norm_target)
                mapped_indices.append(idx)
            else:
                missing.append(target)

        is_compat = (len(missing) == 0)
        status_str = "COMPATIBLE" if is_compat else "INCOMPATIBLE"

        return {
            "status": status_str,
            "available_count": len(available_channels),
            "target_count": len(self.target_channels),
            "missing_channels": missing,
            "mapped_indices": mapped_indices,
            "is_compatible": is_compat
        }

    def reorder_channels(self, data: np.ndarray if 'np' in locals() else None, mapped_indices: list) -> np.ndarray if 'np' in locals() else None:
        """
        Reorder data channels according to target Group B channel order.
        """
        import numpy as np
        if data is None or len(mapped_indices) != len(self.target_channels):
            return data
        return data[mapped_indices, :]
