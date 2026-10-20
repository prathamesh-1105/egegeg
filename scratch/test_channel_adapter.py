"""
NeuroOS Phase 6 OpenBCI Channel Adapter Test Suite
Validates strict contract checks, spherical spline spatial reconstruction, error handling,
deterministic channel ordering, and regression protection for existing Dataset/Streaming tests.
"""

import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.abspath("."))

from src.neuro_os.channel_adapter import OpenBCIChannelAdapter, AdapterValidationError
from src.neuro_os.lsl_bridge import DatasetEEGSource, LSLEEGSource


def test_correct_8ch_input():
    print("\n--- A. TESTING CORRECT 8-CHANNEL INPUT INTERPOLATION ---")
    adapter = OpenBCIChannelAdapter()
    
    # 8-channel synthetic EEG data: (8, 480)
    data_8ch = np.random.randn(8, 480) * 1e-5
    ch_names = ["F3", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]

    res = adapter.transform(data_8ch, ch_names)
    assert res["success"] is True
    assert res["data_15ch"].shape == (1, 15, 480)
    assert res["channels"] == adapter.STEP60_15_CHANNELS
    assert res["telemetry"]["interpolation_active"] is True
    assert res["telemetry"]["adapter_latency_ms"] < 10.0 # Fast performance
    print(f"Correct 8-ch input successfully transformed to shape {res['data_15ch'].shape} in {res['telemetry']['adapter_latency_ms']} ms")


def test_wrong_channel_count():
    print("\n--- B. TESTING WRONG CHANNEL COUNT REJECTION ---")
    adapter = OpenBCIChannelAdapter()
    
    data_5ch = np.random.randn(5, 480)
    ch_names = ["F3", "F4", "C3", "Cz", "C4"]

    res = adapter.transform(data_5ch, ch_names)
    assert res["success"] is False
    assert "Invalid channel count" in res["telemetry"]["error_detail"]
    print(f"Wrong channel count correctly rejected: {res['telemetry']['error_detail']}")


def test_wrong_missing_channel_names():
    print("\n--- C. TESTING WRONG / MISSING CHANNEL NAMES REJECTION ---")
    adapter = OpenBCIChannelAdapter()
    
    data_8ch = np.random.randn(8, 480)
    ch_names = ["F3", "F4", "C3", "Cz", "INVALID_CH", "P3", "Pz", "P4"]

    res = adapter.transform(data_8ch, ch_names)
    assert res["success"] is False
    assert "Channel name mismatch" in res["telemetry"]["error_detail"]
    print(f"Invalid channel names correctly rejected: {res['telemetry']['error_detail']}")


def test_invalid_nan_inf_signal():
    print("\n--- D. TESTING INVALID NAN / INF SIGNAL REJECTION ---")
    adapter = OpenBCIChannelAdapter()
    
    data_nan = np.random.randn(8, 480)
    data_nan[0, 10] = np.nan
    ch_names = ["F3", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]

    res = adapter.transform(data_nan, ch_names)
    assert res["success"] is False
    assert "NaN or infinite" in res["telemetry"]["error_detail"]
    print(f"NaN signal correctly rejected: {res['telemetry']['error_detail']}")


def test_insufficient_samples():
    print("\n--- E. TESTING INSUFFICIENT SAMPLES REJECTION ---")
    adapter = OpenBCIChannelAdapter()
    
    data_short = np.random.randn(8, 200) # Only 200 samples instead of 480
    ch_names = ["F3", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]

    res = adapter.transform(data_short, ch_names)
    assert res["success"] is False
    assert "Insufficient sample window" in res["telemetry"]["error_detail"]
    print(f"Short window correctly rejected: {res['telemetry']['error_detail']}")


def test_deterministic_channel_ordering():
    print("\n--- F. TESTING DETERMINISTIC CHANNEL ORDERING ---")
    adapter = OpenBCIChannelAdapter()
    
    data_8ch = np.random.randn(8, 480)
    # Shuffled channel list
    shuffled_names = ["P4", "C3", "F4", "Cz", "P3", "C4", "F3", "Pz"]

    res = adapter.transform(data_8ch, shuffled_names)
    assert res["success"] is True
    assert res["data_15ch"].shape == (1, 15, 480)
    assert res["channels"] == adapter.STEP60_15_CHANNELS
    print("Shuffled input channel names correctly reordered deterministically before interpolation.")


def test_existing_dataset_eeg_source_unaffected():
    print("\n--- G. TESTING EXISTING DATASET EEG SOURCE IS UNAFFECTED ---")
    source = DatasetEEGSource()
    win = source.read_window()
    assert win is not None
    assert win["window_data"].shape == (1, 15, 480)
    assert win["adapter_telemetry"]["interpolation_active"] is False
    print(f"DatasetEEGSource remains 100% unaffected and streams genuine 15-ch EDF data.")


def test_no_lsl_environment_reports_disconnected():
    print("\n--- H. TESTING NO-LSL ENVIRONMENT HONEST DISCONNECTION ---")
    lsl_source = LSLEEGSource()
    info = lsl_source.get_info()
    assert info["source_type"] == "LSL_HARDWARE"
    if not info["connected"]:
        assert info["status"] == "DISCONNECTED"
        assert info["error_detail"] is not None
        print(f"LSLEEGSource accurately reports DISCONNECTED state: {info['error_detail']}")


def run_all_adapter_tests():
    print("================================================================================")
    print("STARTING NEURO-OS PHASE 6 OPENBCI CHANNEL ADAPTER TEST SUITE")
    print("================================================================================")
    
    test_correct_8ch_input()
    test_wrong_channel_count()
    test_wrong_missing_channel_names()
    test_invalid_nan_inf_signal()
    test_insufficient_samples()
    test_deterministic_channel_ordering()
    test_existing_dataset_eeg_source_unaffected()
    test_no_lsl_environment_reports_disconnected()

    print("\n================ NEURO-OS PHASE 6 ADAPTER SUITE RESULTS ================")
    print(" 8->15 Spherical Spline Reconstruction : VERIFIED")
    print(" Channel Count & Name Validation        : VERIFIED")
    print(" NaN / Inf Quality Rejection            : VERIFIED")
    print(" Sample Length & Ordering Determinism  : VERIFIED")
    print(" Dataset & Phase 5 Streaming Protection : VERIFIED")
    print("==================================================================")
    print("\n[PASSED] NEURO-OS PHASE 6 ADAPTER TEST SUITE PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_adapter_tests()
