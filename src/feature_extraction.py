import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.integrate as integrate
import mne


def extract_raw_samples_table(
    raw: mne.io.Raw,
    channel_name: str,
    start_time: float,
    end_time: float,
    n_samples: int = 15
) -> pd.DataFrame:
    """
    Extracts raw EEG numerical sample values directly from the MNE raw object.

    Parameters:
        raw (mne.io.Raw): MNE raw instance.
        channel_name (str): Selected channel label.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        n_samples (int): Number of consecutive samples to extract for preview table.

    Returns:
        pd.DataFrame: Table with Sample Number, Time (seconds), and Amplitude (uV).
    """
    sfreq = raw.info['sfreq']
    start_idx = int(max(0, start_time * sfreq))
    ch_idx = raw.ch_names.index(channel_name)
    
    # Calculate sample slice range
    stop_idx = min(len(raw.times), start_idx + n_samples)
    
    times = raw.times[start_idx:stop_idx]
    # Convert from Volts to uV
    data_uv = raw.get_data(picks=[ch_idx], start=start_idx, stop=stop_idx)[0] * 1e6

    sample_indices = np.arange(start_idx, stop_idx)

    df_samples = pd.DataFrame({
        "Sample Number": sample_indices,
        "Time (seconds)": np.round(times, 4),
        "Amplitude (µV)": np.round(data_uv, 4)
    })
    return df_samples


def calculate_time_domain_features(signal_uv: np.ndarray) -> dict:
    """
    Calculates 8 core time-domain mathematical metrics directly from real EEG microvolt samples.

    Parameters:
        signal_uv (np.ndarray): 1D array of EEG signal amplitudes in uV.

    Returns:
        dict: Calculated mathematical features.
    """
    if len(signal_uv) == 0:
        return {
            "mean": 0.0, "variance": 0.0, "std": 0.0, "rms": 0.0,
            "min": 0.0, "max": 0.0, "peak_to_peak": 0.0, "energy": 0.0
        }

    mean_val = float(np.mean(signal_uv))
    var_val = float(np.var(signal_uv))
    std_val = float(np.std(signal_uv))
    rms_val = float(np.sqrt(np.mean(signal_uv ** 2)))
    min_val = float(np.min(signal_uv))
    max_val = float(np.max(signal_uv))
    ptp_val = float(max_val - min_val)
    energy_val = float(np.sum(signal_uv ** 2))

    return {
        "mean": mean_val,
        "variance": var_val,
        "std": std_val,
        "rms": rms_val,
        "min": min_val,
        "max": max_val,
        "peak_to_peak": ptp_val,
        "energy": energy_val
    }


def calculate_fft_psd(signal_uv: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Power Spectral Density (PSD) using Welch's method to transform
    the time-domain EEG signal into frequency spectrum data.

    Parameters:
        signal_uv (np.ndarray): 1D array of EEG signal amplitudes in uV.
        sfreq (float): Sampling frequency in Hz.

    Returns:
        freqs (np.ndarray): Frequency vector in Hz.
        psd (np.ndarray): Power Spectral Density vector in uV^2/Hz.
    """
    if len(signal_uv) < 16:
        return np.array([0.0]), np.array([0.0])

    nperseg = min(len(signal_uv), int(sfreq * 2))  # 2-second window for resolution
    freqs, psd = signal.welch(signal_uv, fs=sfreq, nperseg=nperseg)
    return freqs, psd


def calculate_band_powers(freqs: np.ndarray, psd: np.ndarray) -> dict:
    """
    Calculates absolute band power for standard EEG frequency bands:
      - Delta: 0.5 - 4 Hz
      - Theta: 4 - 8 Hz
      - Alpha: 8 - 13 Hz
      - Beta: 13 - 30 Hz
      - Gamma: 30 - 45 Hz

    Parameters:
        freqs (np.ndarray): Frequency vector in Hz.
        psd (np.ndarray): Spectral power vector.

    Returns:
        dict: Band powers for each frequency band in uV^2.
    """
    bands = {
        "Delta": (0.5, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta": (13.0, 30.0),
        "Gamma": (30.0, 45.0)
    }

    band_powers = {}
    for band_name, (f_low, f_high) in bands.items():
        idx_band = np.logical_and(freqs >= f_low, freqs <= f_high)
        if np.any(idx_band) and len(freqs[idx_band]) > 1:
            try:
                # Integrate PSD over frequency band using Simpson's rule
                power = integrate.simpson(psd[idx_band], freqs[idx_band])
            except Exception:
                power = np.trapz(psd[idx_band], freqs[idx_band])
        else:
            power = 0.0
        band_powers[band_name] = float(max(0.0, power))

    return band_powers


def apply_beta_bandpass_filter(
    signal_uv: np.ndarray,
    sfreq: float,
    lowcut: float = 13.0,
    highcut: float = 30.0,
    order: int = 4
) -> np.ndarray:
    """
    Applies a 13-30 Hz Butterworth band-pass filter to isolate Beta band rhythms.

    Parameters:
        signal_uv (np.ndarray): 1D array of EEG signal amplitudes in uV.
        sfreq (float): Sampling rate in Hz.
        lowcut (float): Lower cutoff frequency (default 13 Hz).
        highcut (float): Upper cutoff frequency (default 30 Hz).
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered Beta-band EEG signal in uV.
    """
    if len(signal_uv) <= 12:
        return signal_uv.copy()

    nyquist = 0.5 * sfreq
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth filter
    b, a = signal.butter(order, [low, high], btype='bandpass')
    # Zero-phase digital filtering using filtfilt
    beta_signal = signal.filtfilt(b, a, signal_uv)
    return beta_signal


def extract_event_signals(
    raw: mne.io.Raw,
    channel_name: str,
    df_events: pd.DataFrame,
    event_code: str
) -> np.ndarray:
    """
    Extracts and concatenates all signal segments matching a specific event code (e.g. T0, T1, T2).

    Parameters:
        raw (mne.io.Raw): MNE raw data object.
        channel_name (str): Target channel label.
        df_events (pd.DataFrame): DataFrame containing parsed annotations.
        event_code (str): 'T0' (Rest), 'T1' (Left Fist), or 'T2' (Right Fist).

    Returns:
        np.ndarray: Concatenated microvolt array for all matching event segments.
    """
    if df_events.empty:
        return np.array([])

    ch_idx = raw.ch_names.index(channel_name)
    sfreq = raw.info['sfreq']
    event_sub = df_events[df_events['event_code'] == event_code]

    if event_sub.empty:
        return np.array([])

    segments = []
    for _, ev in event_sub.iterrows():
        start_idx = int(max(0, ev['onset_sec'] * sfreq))
        stop_idx = int(min(len(raw.times), ev['end_sec'] * sfreq))
        if stop_idx > start_idx:
            seg_uv = raw.get_data(picks=[ch_idx], start=start_idx, stop=stop_idx)[0] * 1e6
            segments.append(seg_uv)

    if len(segments) == 0:
        return np.array([])

    return np.concatenate(segments)


def compute_feature_vector(signal_uv: np.ndarray, sfreq: float) -> dict:
    """
    Computes complete mathematical feature dictionary & vector representation for a signal segment.

    Parameters:
        signal_uv (np.ndarray): Signal values in uV.
        sfreq (float): Sampling frequency in Hz.

    Returns:
        dict: Complete dictionary of 13 extracted features.
    """
    time_feats = calculate_time_domain_features(signal_uv)
    freqs, psd = calculate_fft_psd(signal_uv, sfreq)
    band_powers = calculate_band_powers(freqs, psd)

    full_features = {
        "Mean (µV)": time_feats["mean"],
        "Variance (µV²)": time_feats["variance"],
        "Std Dev (µV)": time_feats["std"],
        "RMS (µV)": time_feats["rms"],
        "Min (µV)": time_feats["min"],
        "Max (µV)": time_feats["max"],
        "Peak-to-Peak (µV)": time_feats["peak_to_peak"],
        "Energy (µV²)": time_feats["energy"],
        "Delta Power (0.5–4 Hz)": band_powers.get("Delta", 0.0),
        "Theta Power (4–8 Hz)": band_powers.get("Theta", 0.0),
        "Alpha Power (8–13 Hz)": band_powers.get("Alpha", 0.0),
        "Beta Power (13–30 Hz)": band_powers.get("Beta", 0.0),
        "Gamma Power (30–45 Hz)": band_powers.get("Gamma", 0.0)
    }

    return full_features


def compute_event_comparison(
    raw: mne.io.Raw,
    channel_name: str,
    df_events: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes mathematical features for Rest (T0), Left Fist (T1), and Right Fist (T2) event segments.

    Returns:
        pd.DataFrame: Side-by-side comparison table.
    """
    sfreq = raw.info['sfreq']
    t0_sig = extract_event_signals(raw, channel_name, df_events, "T0")
    t1_sig = extract_event_signals(raw, channel_name, df_events, "T1")
    t2_sig = extract_event_signals(raw, channel_name, df_events, "T2")

    f0 = compute_feature_vector(t0_sig, sfreq)
    f1 = compute_feature_vector(t1_sig, sfreq)
    f2 = compute_feature_vector(t2_sig, sfreq)

    rows = []
    for key in f0.keys():
        rows.append({
            "Feature": key,
            "Rest (T0)": round(f0[key], 4),
            "Left Fist (T1)": round(f1[key], 4),
            "Right Fist (T2)": round(f2[key], 4)
        })

    return pd.DataFrame(rows)
