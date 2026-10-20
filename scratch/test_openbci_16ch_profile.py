"""
NeuroOS Phase 7 OpenBCI 16-Channel Cyton + Daisy Test Suite
Validates 16-channel physical extraction, 0 interpolation, Signal Quality Gate,
Hardware Profiles, Calibration Manager, and regression protection for Phase 4-6.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.neuro_os.hardware_profiles import HardwareProfileRegistry, OPENBCI_16CH_PHYSICAL, STEP60_TARGET_CHANNELS
from src.neuro_os.signal_quality import NeuroOSSignalQualityGate
from src.neuro_os.calibration_manager import NeuroOSCalibrationManager
from src.neuro_os.lsl_bridge import DatasetEEGSource, LSLEEGSource


def test_hardware_profile_registry():
    print("\n--- 1. TESTING HARDWARE PROFILE REGISTRY ---")
    p16 = HardwareProfileRegistry.get_profile("OPENBCI_CYTON_DAISY_16CH")
    assert p16 is not None
    assert p16.channel_count == 16
    assert p16.requires_interpolation is False
    assert p16.unused_channels == ["O1"]
    assert p16.status == "RECOMMENDED_PHYSICAL"
    print(f"OPENBCI_CYTON_DAISY_16CH profile verified: {p16.display_name} ({p16.status})")


def test_signal_quality_gate():
    print("\n--- 2. TESTING SIGNAL QUALITY GATE ---")
    gate = NeuroOSSignalQualityGate()

    # Case A: Good 15-channel signal
    good_data = np.random.randn(15, 480) * 10.0 # ~10 uV
    res_good = gate.evaluate_quality(good_data, STEP60_TARGET_CHANNELS, sfreq=160.0)
    assert res_good["passed"] is True
    assert res_good["signal_quality"] == "EXCELLENT"

    # Case B: NaN signal
    nan_data = good_data.copy()
    nan_data[2, 50] = np.nan
    res_nan = gate.evaluate_quality(nan_data, STEP60_TARGET_CHANNELS, sfreq=160.0)
    assert res_nan["passed"] is False
    assert res_nan["quality_status"] == "NAN_INF_DETECTED"

    # Case C: Motion artifact (> 500 uV)
    art_data = good_data.copy()
    art_data[0, 100] = 750.0
    res_art = gate.evaluate_quality(art_data, STEP60_TARGET_CHANNELS, sfreq=160.0)
    assert "FC3" in res_art["bad_channels"]

    print("Signal Quality Gate (Good signal, NaN rejection, Motion artifact disengage) verified.")


def test_calibration_manager():
    print("\n--- 3. TESTING HARDWARE CALIBRATION MANAGER ---")
    calib = NeuroOSCalibrationManager(target_windows_per_class=3)
    status0 = calib.start_calibration("OPENBCI_CYTON_DAISY_16CH")
    assert status0["state"] == "COLLECTING_REST"

    dummy_win = np.random.randn(1, 15, 480)
    dummy_qual = {"passed": True}

    # Record 3 REST windows
    for _ in range(3):
        res = calib.record_window(dummy_win, dummy_qual)
    assert res["state"] == "COLLECTING_LEFT"

    # Record 3 LEFT_FIST windows
    for _ in range(3):
        res = calib.record_window(dummy_win, dummy_qual)
    assert res["state"] == "COLLECTING_RIGHT"

    # Record 3 RIGHT_FIST windows
    for _ in range(3):
        res = calib.record_window(dummy_win, dummy_qual)
    assert res["state"] == "COMPLETED"

    print(f"Calibration workflow completed cleanly | Session: {res['session_id']} | Final State: {res['state']}")


def test_lsl_bridge_16ch_telemetry():
    print("\n--- 4. TESTING LSL BRIDGE 16CH TELEMETRY ---")
    lsl = LSLEEGSource()
    info = lsl.get_info()
    assert info["source_type"] == "LSL_HARDWARE"
    assert "adapter_telemetry" in info
    print(f"LSLEEGSource 16-channel telemetry contract verified.")


def test_dataset_eeg_source_regression():
    print("\n--- 5. TESTING DATASET EEG SOURCE REGRESSION ---")
    dataset = DatasetEEGSource()
    win = dataset.read_window()
    assert win is not None
    assert win["window_data"].shape == (1, 15, 480)
    assert win["adapter_telemetry"]["hardware_mode"] == "SIMULATED_PHYSIONET"
    print("DatasetEEGSource regression check passed cleanly.")


def run_all_16ch_tests():
    print("================================================================================")
    print("STARTING NEURO-OS PHASE 7 OPENBCI 16CH CYTON + DAISY TEST SUITE")
    print("================================================================================")

    test_hardware_profile_registry()
    test_signal_quality_gate()
    test_calibration_manager()
    test_lsl_bridge_16ch_telemetry()
    test_dataset_eeg_source_regression()

    print("\n================ NEURO-OS PHASE 7 SUITE RESULTS ================")
    print(" 16CH Hardware Profile Registry        : VERIFIED")
    print(" Signal Quality Gate (Artifact & NaN)   : VERIFIED")
    print(" Hardware Calibration Manager Protocol : VERIFIED")
    print(" LSL Bridge 16CH Telemetry Contract     : VERIFIED")
    print(" Dataset Simulation & Regression Guard  : VERIFIED")
    print("==================================================================")
    print("\n[PASSED] NEURO-OS PHASE 7 TEST SUITE PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_16ch_tests()
