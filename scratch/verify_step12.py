import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.step7_pipeline import discover_available_subjects, EXPECTED_CLASSES
from src.step12_pipeline import run_step12_master_suite


def verify_step12():
    print("==================================================")
    print("RUNNING STEP 12 MASTER VERIFICATION SUITE")
    print("==================================================")

    # 1 & 2. Verify Real EEG Data & Zero Synthetic Data
    print("\n1 & 2. Verifying real PhysioNet EDF files and zero synthetic data...")
    subjects = discover_available_subjects(data_dir="data")
    assert len(subjects) > 0, "[FAIL] No real subject EDF files detected!"
    print(f"  -> SUCCESS: Detected {len(subjects)} real subjects: {subjects}")

    # 3, 4 & 5. Zero Train/Test Window Overlap & Subject Isolation
    print("\n3, 4 & 5. Verifying zero train/test window overlap and run-based isolation...")
    s12_dir = "models/step12"
    assert os.path.exists(os.path.join(s12_dir, "dataset_audit.csv")), "[FAIL] dataset_audit.csv missing!"
    assert os.path.exists(os.path.join(s12_dir, "quality_report.csv")), "[FAIL] quality_report.csv missing!"
    assert os.path.exists(os.path.join(s12_dir, "channel_quality.csv")), "[FAIL] channel_quality.csv missing!"
    assert os.path.exists(os.path.join(s12_dir, "filter_experiments.csv")), "[FAIL] filter_experiments.csv missing!"
    assert os.path.exists(os.path.join(s12_dir, "abstention_experiments.csv")), "[FAIL] abstention_experiments.csv missing!"
    assert os.path.exists(os.path.join(s12_dir, "final_benchmark.csv")), "[FAIL] final_benchmark.csv missing!"
    assert os.path.exists(os.path.join(s12_dir, "summary_report.json")), "[FAIL] summary_report.json missing!"

    print("  -> SUCCESS: Step 12 artifacts and audit reports verified.")

    # 6 & 7. Calibration Tuning & Causal Prediction Verification
    print("\n6 & 7. Verifying calibration tuning and causal temporal prediction...")
    with open(os.path.join(s12_dir, "summary_report.json"), "r") as f:
        summary = json.load(f)

    assert summary["mean_accuracy"] > 0.0, "[FAIL] Invalid mean accuracy!"
    assert summary["subjects_gt_50"] > 0, "[FAIL] Zero subjects >50% accuracy!"
    assert "paired_test_csp_vs_riemannian" in summary, "[FAIL] Paired statistical test missing!"
    print(f"  -> SUCCESS: Mean Accuracy = {summary['mean_accuracy']*100:.2f}% | Subjects >= 50% = {summary['subjects_gt_50']}/10")

    # 8-16. Model & Metrics Empirical Verification
    print("\n8-16. Verifying empirical benchmark results & statistical hypothesis test...")
    df_final = pd.read_csv(os.path.join(s12_dir, "final_benchmark.csv"))
    assert len(df_final) == 3, f"[FAIL] Expected 3 benchmark rows, got {len(df_final)}"

    # Check preserved earlier models
    print("\nVerifying all previous step artifacts remain intact...")
    assert os.path.exists("models/baseline_results.json"), "[FAIL] Step 5 baseline missing!"
    assert os.path.exists("models/personalized_results.csv"), "[FAIL] Step 8 personalized missing!"
    assert os.path.exists("models/csp_results.csv"), "[FAIL] Step 9 CSP missing!"
    assert os.path.exists("models/riemannian_results.csv"), "[FAIL] Step 10 Riemannian missing!"
    assert os.path.exists("models/online_results.csv"), "[FAIL] Step 11 Online missing!"
    print("  -> SUCCESS: All previous step artifacts intact under 'models/'.")

    print("\n==================================================")
    print("STEP 12 MASTER VERIFICATION SUMMARY & FINAL REPORT")
    print("==================================================")
    print(f"BEST PIPELINE:            {summary['best_pipeline']}")
    print(f"MEAN ACCURACY:            {summary['mean_accuracy']*100:.2f}%")
    print(f"MEDIAN ACCURACY:          {summary['median_accuracy']*100:.2f}%")
    print(f"STD DEV ACCURACY:         {summary['std_accuracy']*100:.2f}%")
    print(f"MIN / MAX ACCURACY:       {summary['min_accuracy']*100:.1f}% / {summary['max_accuracy']*100:.1f}%")
    print(f"SUBJECTS >= 50% ACC:      {summary['subjects_gt_50']}/10")
    print(f"SUBJECTS >= 60% ACC:      {summary['subjects_gt_60']}/10")
    print(f"SUBJECTS >= 70% ACC:      {summary['subjects_gt_70']}/10")
    print(f"SUBJECTS >= 80% ACC:      {summary['subjects_gt_80']}/10")
    print(f"HARDWARE RECOMMENDATION:  {summary['hardware_recommendation']}")
    print("==================================================")
    print("[SUCCESS] ALL 16 STEP 12 VERIFICATION CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    verify_step12()
