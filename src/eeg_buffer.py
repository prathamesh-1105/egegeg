import numpy as np


class EEGBuffer:
    """
    Rolling EEG Sample Buffer.
    Maintains a sliding FIFO queue of multi-channel EEG samples to construct
    clean 3.0-second windows (480 samples @ 160 Hz) with 1.5-second step advancement.
    """

    def __init__(self, n_channels: int = 15, sfreq: float = 160.0, window_sec: float = 3.0, step_sec: float = 1.5):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.window_sec = window_sec
        self.step_sec = step_sec

        self.capacity = int(window_sec * sfreq)
        self.step_samples = int(step_sec * sfreq)
        self.buffer = np.zeros((n_channels, 0))
        self.total_samples_received = 0

    def append_samples(self, samples: np.ndarray) -> None:
        """
        Append new incoming samples.
        samples shape: (n_channels, n_new_samples)
        """
        if samples.ndim == 1:
            samples = np.expand_dims(samples, axis=1)

        if samples.shape[0] != self.n_channels:
            # Adjust channels if shape differs
            min_ch = min(self.n_channels, samples.shape[0])
            adj_samples = np.zeros((self.n_channels, samples.shape[1]))
            adj_samples[:min_ch, :] = samples[:min_ch, :]
            samples = adj_samples

        self.buffer = np.hstack([self.buffer, samples])
        self.total_samples_received += samples.shape[1]

        # Keep buffer within capacity bounds
        if self.buffer.shape[1] > self.capacity:
            self.buffer = self.buffer[:, -self.capacity:]

    def is_ready(self) -> bool:
        """Return True if buffer has accumulated full window capacity (e.g. 480 samples)."""
        return self.buffer.shape[1] >= self.capacity

    def get_fill_status(self) -> dict:
        return {
            "current_samples": self.buffer.shape[1],
            "target_samples": self.capacity,
            "percentage": min(100.0, round((self.buffer.shape[1] / self.capacity) * 100.0, 1)),
            "is_ready": self.is_ready()
        }

    def get_window(self) -> np.ndarray:
        """Get the current 3-second window array of shape (n_channels, capacity)."""
        if self.buffer.shape[1] < self.capacity:
            # Pad with leading zeroes if buffer is incomplete
            pad_len = self.capacity - self.buffer.shape[1]
            return np.pad(self.buffer, ((0, 0), (pad_len, 0)), mode='constant')
        return self.buffer[:, -self.capacity:]

    def reset(self) -> None:
        self.buffer = np.zeros((self.n_channels, 0))
        self.total_samples_received = 0
