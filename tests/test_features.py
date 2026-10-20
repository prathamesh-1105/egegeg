import os
import sys
import numpy as np
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath("."))

from src.load_eeg import load_eeg_raw, get_eeg_info
from src.events import get_eeg_events
from src.feature_extraction import (
    extract_raw_samples_table,
    calculate_time_domain_features,
    calculate_fft_psd,
    calculate_band_powers,
    apply_beta_bandpass_filter,
    extract_event_signals,
    compute_feature_vector,
    compute_event_comparison
)


def test_step3_feature_extraction():
    edf_path = "data/S001R03.edf"
    assert os.path.exists(edf_path), f"EDF file not found at {edf_path}"

    print("1. Loading raw EEG data...")
    raw = load_eeg_raw(edf_path)
    info = get_eeg_info(raw, edf_path)
    df_events = get_eeg_events(raw)

    ch_name = "C3.." if "C3.." in info["ch_names"] else info["ch_names"][0]
    print(f"Target channel for testing: {ch_name}")

    print("\n2. Testing extract_raw_samples_table...")
    df_samples = extract_raw_samples_table(raw, ch_name, start_time=0.0, end_time=5.0, n_samples=10)
    print(df_samples)
    assert len(df_samples) == 10
    assert "Sample Number" in df_samples.columns
    assert "Time (seconds)" in df_samples.columns
    assert "Amplitude (µV)" in df_samples.columns

    print("\n3. Testing calculate_time_domain_features...")
    data_uv = raw.get_data(picks=[info["ch_names"].index(ch_name)], start=0, stop=160)[0] * 1e6
    feats = calculate_time_domain_features(data_uv)
    print("Time domain features:", feats)
    assert "mean" in feats
    assert "variance" in feats
    assert "std" in feats
    assert "rms" in feats
    assert "min" in feats
    assert "max" in feats
    assert "peak_to_peak" in feats
    assert "energy" in feats

    print("\n4. Testing calculate_fft_psd...")
    freqs, psd = calculate_fft_psd(data_uv, info["sfreq"])
    print(f"Freqs shape: {freqs.shape}, PSD shape: {psd.shape}")
    assert len(freqs) > 0 and len(psd) > 0

    print("\n5. Testing calculate_band_powers...")
    band_powers = calculate_band_powers(freqs, psd)
    print("Band powers:", band_powers)
    for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
        assert band in band_powers
        assert band_powers[band] >= 0.0

    print("\n6. Testing apply_beta_bandpass_filter...")
    beta_sig = apply_beta_bandpass_filter(data_uv, info["sfreq"])
    assert len(beta_sig) == len(data_uv)
    print(f"Raw std: {np.std(data_uv):.4f}, Beta std: {np.std(beta_sig):.4f}")

    print("\n7. Testing extract_event_signals for T0, T1, T2...")
    t0_sig = extract_event_signals(raw, ch_name, df_events, "T0")
    t1_sig = extract_event_signals(raw, ch_name, df_events, "T1")
    t2_sig = extract_event_signals(raw, ch_name, df_events, "T2")
    print(f"T0 signal length: {len(t0_sig)}, T1 signal length: {len(t1_sig)}, T2 signal length: {len(t2_sig)}")
    assert len(t0_sig) > 0
    assert len(t1_sig) > 0
    assert len(t2_sig) > 0

    print("\n8. Testing compute_feature_vector...")
    vec = compute_feature_vector(t1_sig, info["sfreq"])
    print("13-D Feature Vector keys:", list(vec.keys()))
    assert len(vec) == 13

    print("\n9. Testing compute_event_comparison...")
    df_comp = compute_event_comparison(raw, ch_name, df_events)
    print("\nEvent Comparison Table:")
    print(df_comp.to_string())
    assert len(df_comp) == 13
    assert "Rest (T0)" in df_comp.columns
    assert "Left Fist (T1)" in df_comp.columns
    assert "Right Fist (T2)" in df_comp.columns

    print("\n[SUCCESS] ALL STEP 3 FEATURE EXTRACTION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_step3_feature_extraction()
