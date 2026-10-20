import os
import sys
import pandas as pd
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.build_dataset import build_eeg_feature_dataset


def verify_step4():
    print("==================================================")
    print("RUNNING STEP 4 DATASET VERIFICATION CHECKS")
    print("==================================================")

    output_csv = "data/processed/eeg_features.csv"
    
    # 1 & 2 & 3. Load EDF files, parse T0/T1/T2 annotations, build 2s sliding windows
    print("\n1. Building feature dataset from real S001-S010 EDF files...")
    df_features, stats = build_eeg_feature_dataset(
        subjects=[f"S{i:03d}" for i in range(1, 11)],
        output_csv=output_csv,
        window_sec=2.0,
        overlap=0.5
    )

    # Verification Item 8 & 9: CSV created and rows > 0
    print(f"\n8 & 9. Verifying CSV creation at '{output_csv}'...")
    assert os.path.exists(output_csv), f"[FAIL] CSV file not found at {output_csv}"
    assert len(df_features) > 0, f"[FAIL] Dataset row count is 0"
    print(f"  -> SUCCESS: CSV file exists with {len(df_features)} total rows.")

    # Verification Item 1: Subjects processed > 0
    print(f"\n1. Verifying real subject files loaded...")
    assert stats["subjects_processed"] > 0, "[FAIL] No subject files were processed"
    print(f"  -> SUCCESS: {stats['subjects_processed']} subjects processed across {stats['recordings_processed']} EDF recordings.")

    # Verification Item 2 & 6: Labels strictly REST, LEFT_FIST, RIGHT_FIST
    print(f"\n2 & 6. Verifying class labels...")
    unique_labels = set(df_features["label"].unique())
    expected_labels = {"REST", "LEFT_FIST", "RIGHT_FIST"}
    assert unique_labels.issubset(expected_labels), f"[FAIL] Unexpected labels found: {unique_labels}"
    print(f"  -> SUCCESS: Labels found: {unique_labels}")
    print(f"     - REST windows:       {stats['rest_windows']}")
    print(f"     - LEFT_FIST windows:  {stats['left_fist_windows']}")
    print(f"     - RIGHT_FIST windows: {stats['right_fist_windows']}")

    # Verification Item 4: Exactly 13 features per window
    print(f"\n4. Verifying feature column count...")
    feature_cols = [
        "mean", "variance", "std", "rms", "min", "max", "peak_to_peak", "energy",
        "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"
    ]
    for col in feature_cols:
        assert col in df_features.columns, f"[FAIL] Feature column missing: {col}"
    print(f"  -> SUCCESS: All 13 mathematical feature columns present.")

    # Verification Item 5: No NaN or Infinity values
    print(f"\n5. Verifying data quality (NaN / Infinity check)...")
    nan_count = df_features[feature_cols].isna().sum().sum()
    inf_count = np.isinf(df_features[feature_cols].values).sum()
    assert nan_count == 0, f"[FAIL] Found {nan_count} NaN values in feature columns"
    assert inf_count == 0, f"[FAIL] Found {inf_count} Infinity values in feature columns"
    print(f"  -> SUCCESS: 0 NaN and 0 Infinity values found across all windows.")

    # Verification Item 7: Subject IDs preserved
    print(f"\n7. Verifying subject separation (subject column)...")
    assert "subject" in df_features.columns, "[FAIL] 'subject' column missing"
    subjects_in_df = df_features["subject"].unique()
    print(f"  -> SUCCESS: Subject IDs preserved: {list(subjects_in_df)}")

    print("\n==================================================")
    print("STEP 4 DATASET SUMMARY & FINAL REPORT")
    print("==================================================")
    print(f"Output File:             {output_csv}")
    print(f"Subjects Processed:      {stats['subjects_processed']}")
    print(f"Recordings Processed:    {stats['recordings_processed']}")
    print(f"Total Real EEG Windows:  {stats['total_windows']}")
    print(f"REST Windows:            {stats['rest_windows']}")
    print(f"LEFT FIST Windows:       {stats['left_fist_windows']}")
    print(f"RIGHT FIST Windows:      {stats['right_fist_windows']}")
    print(f"Rejected Windows:        {stats['rejected_windows']}")
    print(f"Features Per Window:     13")
    print("==================================================")
    print("[SUCCESS] ALL STEP 4 VERIFICATION CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    verify_step4()
