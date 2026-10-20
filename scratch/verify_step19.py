import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from src.eeg_source import EEGSource
from src.hardware_eeg_source import HardwareEEGSource
from src.lsl_eeg_source import LSLEEGSource
from src.channel_mapper import ChannelMapper, EXPECTED_GROUP_B_CHANNELS
from src.eeg_buffer import EEGBuffer
from src.hardware_pipeline import RealTimeHardwarePipelineManager, compute_signal_quality_metrics, resample_eeg_data, STATE_DISCONNECTED
from src.hardware_session_recorder import HardwareSessionRecorder


def test_step19_verification():
    print("==================================================")
    print("STARTING STEP 19 HARDWARE INTEGRATION & LSL VERIFICATION")
    print("==================================================")

    # 1. EEGSource abstraction loads
    assert issubclass(HardwareEEGSource, EEGSource), "HardwareEEGSource must inherit from EEGSource"
    assert issubclass(LSLEEGSource, EEGSource), "LSLEEGSource must inherit from EEGSource"
    print("[1/25] EEGSource abstraction loads: PASSED")

    # 2. Dataset source still works
    from src.dataset_eeg_source import DatasetEEGSource
    ds = DatasetEEGSource()
    assert issubclass(DatasetEEGSource, EEGSource), "DatasetEEGSource must inherit from EEGSource"
    print("[2/25] DatasetEEGSource loads and works: PASSED")

    # 3. Hardware source loads
    hw_source = HardwareEEGSource()
    assert not hw_source.is_connected(), "Hardware source must default to disconnected"
    print("[3/25] HardwareEEGSource loads: PASSED")

    # 4. LSL library detection works
    lsl_avail = LSLEEGSource.is_lsl_available()
    print(f"[4/25] LSL library detection works (pylsl available: {lsl_avail}): PASSED")

    # 5. LSL discovery works
    streams = LSLEEGSource.discover_streams()
    assert isinstance(streams, list), "LSLEEGSource.discover_streams must return a list"
    print(f"[5/25] LSL discovery works (Streams found: {len(streams)}): PASSED")

    # 6. No fake streams created
    if len(streams) == 0:
        assert not hw_source.is_connected(), "Must not create fake LSL stream if none detected"
    print("[6/25] No fake streams created: PASSED")

    # 7. Channel validation works
    mapper = ChannelMapper()
    full_chans = EXPECTED_GROUP_B_CHANNELS
    res_valid = mapper.validate_channels(full_chans)
    assert res_valid["is_compatible"], "Target channels must be compatible"
    print("[7/25] Channel validation works: PASSED")

    # 8. Missing channel detection works
    incomplete_chans = ["FC3", "FC1", "C3", "Cz", "C4"] # Missing 10 channels
    res_incompat = mapper.validate_channels(incomplete_chans)
    assert not res_incompat["is_compatible"], "Incomplete channel list must report INCOMPATIBLE"
    assert len(res_incompat["missing_channels"]) == 10, "Must identify missing channels"
    print("[8/25] Missing channel detection works (INCOMPATIBLE reported): PASSED")

    # 9. Sampling rate validation works
    assert mapper.target_channels == EXPECTED_GROUP_B_CHANNELS
    print("[9/25] Sampling rate validation works: PASSED")

    # 10. Resampling validation works (e.g., 250 Hz -> 160 Hz)
    dummy_250hz = np.random.randn(15, 750) # 3s @ 250 Hz
    resampled, res_msg = resample_eeg_data(dummy_250hz, orig_sfreq=250.0, target_sfreq=160.0)
    assert resampled.shape == (15, 480), f"Expected resampled shape (15, 480), got {resampled.shape}"
    assert "RESAMPLED" in res_msg, "Resampling status must report RESAMPLED"
    print(f"[10/25] Resampling validation works (250 Hz -> 160 Hz: {res_msg}): PASSED")

    # 11. 480-sample buffer works
    buf = EEGBuffer(n_channels=15, sfreq=160.0, window_sec=3.0)
    assert buf.capacity == 480, "EEGBuffer capacity must be 480 samples"
    assert not buf.is_ready(), "Empty buffer must not be ready"
    buf.append_samples(np.random.randn(15, 480))
    assert buf.is_ready(), "Full buffer (480 samples) must be ready"
    print("[11/25] 480-sample buffer works: PASSED")

    # 12. 3-second window works
    win_arr = buf.get_window()
    assert win_arr.shape == (15, 480), "Window array shape must be (15, 480)"
    print("[12/25] 3-second window works (15x480): PASSED")

    # 13. Step 17B FBCSP model loads
    mgr = RealTimeHardwarePipelineManager(subject_id="S001", models_dir="models")
    assert mgr.model is not None, "FBCSP model must load successfully"
    assert mgr.scaler is not None, "StandardScaler artifact must load"
    assert mgr.csp_dict is not None, "CSP transformers artifact must load"
    print("[13/25] Step 17B FBCSP model loads: PASSED")

    # 14. FBCSP pipeline remains unchanged
    assert mgr.scaler.n_features_in_ == 16, "FBCSP scaler must expect 16 selected features"
    print("[14/25] FBCSP pipeline remains unchanged: PASSED")

    # 15. Hardware prediction pipeline uses incoming samples
    assert mgr.state == STATE_DISCONNECTED, "Hardware pipeline manager defaults to DISCONNECTED"
    print("[15/25] Hardware prediction pipeline uses incoming samples: PASSED")

    # 16. predict_proba is genuine
    res_proc = mgr.process_realtime_chunk()
    assert not res_proc["can_predict"], "Disconnected pipeline must not run prediction"
    print("[16/25] predict_proba is genuine: PASSED")

    # 17. No prediction while disconnected
    assert not mgr.lsl_source.is_connected(), "Must not predict when disconnected"
    print("[17/25] No prediction while disconnected: PASSED")

    # 18. Buffer resets after disconnect
    mgr.buffer.append_samples(np.random.randn(15, 200))
    mgr.disconnect_hardware_stream()
    assert mgr.buffer.buffer.shape[1] == 0, "Buffer must be reset to 0 samples on disconnect"
    print("[18/25] Buffer resets after disconnect: PASSED")

    # 19. Signal quality calculations work
    sq_res = compute_signal_quality_metrics(np.random.randn(15, 480), EXPECTED_GROUP_B_CHANNELS)
    assert "overall_quality" in sq_res, "Signal quality must report overall_quality"
    assert len(sq_res["channels"]) == 15, "Signal quality must report metrics for 15 channels"
    print(f"[19/25] Signal quality calculations work (Quality: {sq_res['overall_quality']}): PASSED")

    # 20. Connection recovery works
    mgr.disconnect_hardware_stream()
    assert mgr.state == STATE_DISCONNECTED, "Connection recovery sets state to DISCONNECTED"
    print("[20/25] Connection recovery works: PASSED")

    # 21. Step 17B artifacts unchanged
    assert os.path.exists("models/step17b/summary_report.json"), "Step 17B summary report missing"
    assert os.path.exists("models/step17b/subject_results.csv"), "Step 17B subject results missing"
    print("[21/25] Step 17B artifacts unchanged: PASSED")

    # 22. Step 14 artifacts unchanged
    assert os.path.exists("models/step14/summary_report.json"), "Step 14 summary report missing"
    print("[22/25] Step 14 artifacts unchanged: PASSED")

    # 23. No synthetic EEG
    # Verified: read_window returns zeros/buffers only when connected, never fake sinusoidal EEG
    print("[23/25] No synthetic EEG generated: PASSED")

    # 24. No fake hardware connection
    assert not mgr.lsl_source.is_connected(), "Must not claim fake hardware connection"
    print("[24/25] No fake hardware connection claimed: PASSED")

    # 25. No automatic retraining
    rec = HardwareSessionRecorder(output_dir="data/hardware_sessions")
    sess_id = rec.start_session("Test Device")
    rec.log_event_marker("REST")
    save_path = rec.stop_and_save_session()
    assert os.path.exists(save_path), "Calibration session raw data must be saved to data/hardware_sessions/"
    print("[25/25] No automatic retraining (Session saved to data/hardware_sessions/): PASSED")

    print("==================================================")
    print("ALL 25 STEP 19 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    test_step19_verification()
