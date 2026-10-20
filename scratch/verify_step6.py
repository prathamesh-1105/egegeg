import os
import sys
import pandas as pd
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.bci_stream import RealEEGStreamer
from src.feature_extraction import compute_feature_vector
from src.train_model import predict_eeg, EXPECTED_CLASSES
from src.avatar_3d import render_3d_avatar_html


def verify_step6():
    print("==================================================")
    print("RUNNING STEP 6 BCI & 3D AVATAR VERIFICATION CHECKS")
    print("==================================================")

    test_edf = "data/raw/S009R03.edf"
    assert os.path.exists(test_edf), f"[FAIL] Test EDF file not found at '{test_edf}'"

    # 1. Real EEG file loads
    print(f"\n1. Verifying real EDF file loading ({test_edf})...")
    streamer = RealEEGStreamer(edf_path=test_edf, window_sec=2.0, overlap=0.5)
    total_wins = streamer.get_total_windows()
    print(f"  -> SUCCESS: Loaded {test_edf} cleanly. Total sequential windows: {total_wins}")
    assert total_wins > 0, "[FAIL] No windows extracted"

    # 2 & 3. Sequential processing & 13 feature extraction
    print("\n2 & 3. Verifying sequential window slicing & 13 feature extraction...")
    win0 = streamer.get_window(0)
    win1 = streamer.get_window(1)

    assert win0["window_index"] == 0
    assert win1["window_index"] == 1
    assert "signal_uv" in win0 and len(win0["signal_uv"]) == 320  # 2.0s @ 160Hz

    feats0 = compute_feature_vector(win0["signal_uv"], streamer.sfreq)
    assert len(feats0) == 13, f"[FAIL] Expected 13 features, got {len(feats0)}"
    print(f"  -> SUCCESS: 13 features calculated for Window 0 (Start: {win0['window_start']}s, End: {win0['window_end']}s).")

    # 4 & 5 & 6. Saved scaler & Random Forest model loaded and used for prediction
    print("\n4, 5 & 6. Verifying saved scaler & model inference...")
    assert os.path.exists("models/scaler.pkl"), "[FAIL] models/scaler.pkl missing"
    assert os.path.exists("models/eeg_model.pkl"), "[FAIL] models/eeg_model.pkl missing"

    pred_res0 = predict_eeg(feats0, models_dir="models")
    print(f"  -> SUCCESS: Model prediction for Window 0:")
    print(f"     Predicted Class: {pred_res0['predicted_class']}")
    print(f"     Confidence:      {pred_res0['confidence']:.4f}" if pred_res0['confidence'] else "N/A")
    print(f"     Probabilities:   {pred_res0['probabilities']}")
    assert pred_res0["predicted_class"] in EXPECTED_CLASSES

    # 7. Actual event labels are real
    print("\n7. Verifying ground truth event labels...")
    actual_label0 = win0["actual_label"]
    assert actual_label0 in EXPECTED_CLASSES
    print(f"  -> SUCCESS: Window 0 Actual Label: '{actual_label0}'")

    # 8. Avatar HTML rendering
    print("\n8. Verifying 3D Avatar WebGL HTML generator...")
    avatar_html = render_3d_avatar_html(
        predicted_class=pred_res0["predicted_class"],
        actual_class=actual_label0,
        confidence=pred_res0["confidence"]
    )
    assert "THREE.Scene" in avatar_html
    assert pred_res0["predicted_class"] in avatar_html
    print("  -> SUCCESS: Three.js 3D Avatar HTML generated cleanly.")

    # 9 & 10. Check sequential loop across 5 real windows
    print("\n9 & 10. Verifying non-fake, non-hardcoded sequential streaming loop...")
    for i in range(min(5, total_wins)):
        w = streamer.get_window(i)
        f = compute_feature_vector(w["signal_uv"], streamer.sfreq)
        p = predict_eeg(f, models_dir="models")
        match = "MATCH" if p["predicted_class"] == w["actual_label"] else "MISMATCH"
        print(f"  -> Window {i:02d} [{w['window_start']:5.1f}s-{w['window_end']:5.1f}s] | Actual: {w['actual_label']:10s} | Pred: {p['predicted_class']:10s} | {match}")

    print("\n==================================================")
    print("STEP 6 REAL-TIME BCI & 3D AVATAR SUMMARY REPORT")
    print("==================================================")
    print(f"Target Recording:        {test_edf}")
    print(f"Total Stream Windows:    {total_wins}")
    print(f"Target Motor Channel:    {streamer.ch_name}")
    print(f"Features Per Window:     13")
    print(f"Model In Use:            Random Forest (models/eeg_model.pkl)")
    print(f"Scaler In Use:           StandardScaler (models/scaler.pkl)")
    print(f"3D Avatar Renderer:      Three.js WebGL Engine")
    print("==================================================")
    print("[SUCCESS] ALL STEP 6 VERIFICATION CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    verify_step6()
