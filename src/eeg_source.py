from abc import ABC, abstractmethod
import numpy as np


class EEGSource(ABC):
    """
    Abstract Base Class for EEG Data Sources.
    Separates EEG stream acquisition (PhysioNet Dataset vs LSL Hardware)
    from downstream Riemannian BCI signal processing and classification.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the EEG source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the EEG source."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start streaming data."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming data."""
        pass

    @abstractmethod
    def read_window(self, window_sec: float = 3.0) -> np.ndarray:
        """
        Read a window of raw EEG data.
        Returns:
            np.ndarray: Array of shape (n_channels, n_samples)
        """
        pass

    @abstractmethod
    def get_channel_names(self) -> list:
        """Get channel names provided by this source."""
        pass

    @abstractmethod
    def get_sampling_rate(self) -> float:
        """Get sampling rate (Hz) of this source."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if source is connected and streaming."""
        pass


# Re-export DatasetEEGSource for backwards compatibility
try:
    from src.dataset_eeg_source import DatasetEEGSource
except ImportError:
    pass
