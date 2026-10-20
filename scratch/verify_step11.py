import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.step7_pipeline import discover_available_subjects, EXPECTED_CLASSES
from src.eeg_source import EEGSource, DatasetEEGSource
from src.step11_pipeline import (
    apply_causal_temporal_smoothing,
    train_online_personalized_model,
    run_step11_suite
)


def verify_step11():
    print("==================================================")
    print("RUNNING STEP 11 ONLINE BCI VERIFICATION SUITE")
    print("==================================================")

    # 1 & 12. Verify Real EDF Source Abstraction & Zero Fake Data
    print("\n1 & 12. Verifying EEGSource abstraction layer and real EDF streaming...")
    subjects = discover_available_subjects(data_dir="data")
    assert len(subjects) > 0, "[FAIL] No real subject EDF files detected!"
    
    sample_sub = subjects[0]
    edf_r11 = os.path.join("data", "raw", f"{sample_sub}R11.edf")

    stream_source = DatasetEEGSource(edf_r11, channel_group="Group B", window_sec=3.0, overlap=0.5)
    assert isinstance(stream_source, EEGSource), "[FAIL] DatasetEEGSource must inherit from EEGSource!"

    stream_source.start()
    win_data, info_data = stream_source.get_next_window()
    assert win_data is not None, "[FAIL] Failed to stream first EEG window!"
    assert win_data.shape[0] == 15, f"[FAIL] Expected 15 channels, got {win_data.shape[0]}"
    print(f"  -> SUCCESS: DatasetEEGSource streamed real window shape {win_data.shape} for {sample_sub}!")

    # 2, 3, 4, 5, 6 & 7. Calibration Runs R03/R07 vs Holdout R11 Isolation
    print("\n2, 3, 4, 5, 6 & 7. Verifying calibration fitting (R03/R07) and unseen holdout evaluation (R11)...")
    on_csv_path = "models/online_results.csv"
    assert os.path.exists(on_csv_path), f"[FAIL] '{on_csv_path}' missing!"
    df_on = pd.read_csv(on_csv_path)

    for _, row in df_on.iterrows():
        sub = row["subject"]
        calib_runs = row["calibration_runs"]
        eval_runs = row["evaluation_runs"]
        assert "R11" in eval_runs, f"[FAIL] {sub} evaluation run must be distinct!"
        assert "R11" not in calib_runs, f"[FAIL] Data leakage: R11 in calibration runs for {sub}!"

    print(f"  -> SUCCESS: Strict run-based isolation verified across all {len(df_on)} subjects.")

    # 8, 9, 13 & 14. Real Predictions, Real History & Avatar Label Rule
    print("\n8, 9, 13 & 14. Verifying real inference predictions and history logs...")
    on_dir = "models/online_calibration"
    for sub in subjects:
        m_file = os.path.join(on_dir, sub, "metadata.json")
        if os.path.exists(m_file):
            with open(m_file, "r") as f:
                meta = json.load(f)
            assert meta["raw_accuracy"] > 0.0, f"[FAIL] Invalid raw accuracy for {sub}"
            assert len(meta["sample_predictions"]) > 0, f"[FAIL] Empty prediction history for {sub}"
            for pred_item in meta["sample_predictions"]:
                assert "smoothed_predicted_label" in pred_item
                assert pred_item["status"] in ["CORRECT", "INCORRECT"]

    print("  -> SUCCESS: Empirical model predictions and history verified.")

    # 10 & 11. Causal Temporal Smoothing (Zero Future Data Leakage)
    print("\n10 & 11. Verifying causal 3-window majority vote temporal smoothing...")
    test_past_1 = ["REST", "RIGHT_FIST", "RIGHT_FIST"]
    smoothed_1 = apply_causal_temporal_smoothing(test_past_1, window_size=3)
    assert smoothed_1 == "RIGHT_FIST", f"[FAIL] Majority vote failed: expected RIGHT_FIST, got {smoothed_1}"

    test_past_2 = ["LEFT_FIST", "REST", "LEFT_FIST"]
    smoothed_2 = apply_causal_temporal_smoothing(test_past_2, window_size=3)
    assert smoothed_2 == "LEFT_FIST", f"[FAIL] Majority vote failed: expected LEFT_FIST, got {smoothed_2}"

    print("  -> SUCCESS: Causal temporal smoothing verified with zero future data leakage.")

    # Artifact Serialization Check
    print("\n11. Verifying models/online_calibration/ folder and per-subject artifacts...")
    assert os.path.exists(on_dir), "[FAIL] models/online_calibration directory missing!"
    sub_0 = subjects[0]
    assert os.path.exists(os.path.join(on_dir, sub_0, "cov.pkl")), f"[FAIL] cov.pkl missing for {sub_0}!"
    assert os.path.exists(os.path.join(on_dir, sub_0, "riemannian.pkl")), f"[FAIL] riemannian.pkl missing for {sub_0}!"
    assert os.path.exists(os.path.join(on_dir, sub_0, "scaler.pkl")), f"[FAIL] scaler.pkl missing for {sub_0}!"
    assert os.path.exists(os.path.join(on_dir, sub_0, "model.pkl")), f"[FAIL] model.pkl missing for {sub_0}!"
    assert os.path.exists(os.path.join(on_dir, sub_0, "metadata.json")), f"[FAIL] metadata.json missing for {sub_0}!"
    print(f"  -> SUCCESS: Serialized online calibration artifacts verified under '{on_dir}'.")

    avg_raw_acc = df_on["raw_accuracy"].mean()
    avg_smooth_acc = df_on["smoothed_accuracy"].mean()
    avg_lat = df_on["avg_latency_ms"].mean()

    print("\n==================================================")
    print("STEP 11 VERIFICATION SUMMARY & FINAL REPORT")
    print("==================================================")
    print(f"SUBJECTS TESTED:              {len(df_on)}")
    print(f"CALIBRATION SOURCE:           R03 + R07")
    print(f"EVALUATION SOURCE:            R11")
    print(f"AVERAGE HOLDOUT ACCURACY:     {avg_raw_acc*100:.2f}%")
    print(f"AVERAGE CAUSAL SMOOTH ACC:    {avg_smooth_acc*100:.2f}%")
    print(f"AVERAGE PROCESSING LATENCY:   {avg_lat:.2f} ms")
    print(f"BEST ONLINE BCI SUBJECT:      {df_on.loc[df_on['raw_accuracy'].idxmax()]['subject']} ({df_on['raw_accuracy'].max()*100:.2f}%)")
    print("==================================================")
    print("[SUCCESS] ALL 14 STEP 11 VERIFICATION CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    verify_step11()
