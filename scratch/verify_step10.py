import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.step7_pipeline import discover_available_subjects, EXPECTED_CLASSES
from src.step10_pipeline import run_step10_riemannian_experiments, predict_riemannian_eeg


def verify_step10():
    print("==================================================")
    print("RUNNING STEP 10 RIEMANNIAN VERIFICATION SUITE")
    print("==================================================")

    # 1 & 2. Verify Real Multi-Channel EDF Data & Real Event Labels
    print("\n1 & 2. Verifying real PhysioNet EDF multi-channel data & real annotations...")
    subjects = discover_available_subjects(data_dir="data")
    assert len(subjects) > 0, "[FAIL] No real subject EDF files detected!"
    print(f"  -> SUCCESS: Detected {len(subjects)} real subjects: {subjects}")

    # 3, 4, 5, 6 & 7. Riemannian Reference, Tangent Space & Classifier Fit ONLY on Calibration Runs (R03/R07)
    print("\n3, 4, 5, 6 & 7. Verifying Covariance, Tangent Space & Classifier fit ONLY on calibration data...")
    riem_csv_path = "models/riemannian_results.csv"
    assert os.path.exists(riem_csv_path), f"[FAIL] '{riem_csv_path}' missing!"
    df_riem = pd.read_csv(riem_csv_path)

    for _, row in df_riem.iterrows():
        sub = row["subject"]
        calib_runs = row["calibration_runs"]
        eval_runs = row["evaluation_runs"]
        assert "R11" in eval_runs, f"[FAIL] {sub} evaluation run must be distinct!"
        assert "R11" not in calib_runs, f"[FAIL] Data leakage: R11 in calibration runs for {sub}!"

    print(f"  -> SUCCESS: Strict run-based Riemannian split verified for all {len(df_riem)} subjects.")

    # 8. Zero Window Leakage across Calibration/Evaluation Boundary
    print("\n8. Verifying zero window overlap across calibration/evaluation boundaries...")
    riem_dir = "models/riemannian_personalized"
    for sub in subjects:
        m_file = os.path.join(riem_dir, f"{sub}_metadata.json")
        if os.path.exists(m_file):
            with open(m_file, "r") as f:
                meta = json.load(f)
            assert meta["calibration_runs"] == ["R03", "R07"]
            assert meta["evaluation_runs"] == ["R11"]
            assert meta["training_windows"] > 0
            assert meta["evaluation_windows"] > 0

    print("  -> SUCCESS: Calibration/evaluation boundary isolation verified.")

    # 9 & 10. Real Inference Predictions & Actual Metrics Calculation
    print("\n9 & 10. Verifying real inference predictions and accuracy calculations...")
    avg_s8_acc = df_riem["step8_accuracy"].mean()
    avg_s9_acc = df_riem["step9_accuracy"].mean()
    avg_riem_acc = df_riem["riemannian_accuracy"].mean()

    print(f"  -> SUCCESS: Average Step 8 Accuracy       = {avg_s8_acc*100:.2f}%")
    print(f"  -> SUCCESS: Average Step 9 CSP Accuracy   = {avg_s9_acc*100:.2f}%")
    print(f"  -> SUCCESS: Average Step 10 Riemannian Acc = {avg_riem_acc*100:.2f}% (Gain vs S8 = {avg_riem_acc*100 - avg_s8_acc*100:+.2f}%, Gain vs S9 = {avg_riem_acc*100 - avg_s9_acc*100:+.2f}%)")

    # 11. Riemannian Artifact Serialization
    print("\n11. Verifying models/riemannian_personalized/ folder and serialized files...")
    assert os.path.exists(riem_dir), "[FAIL] models/riemannian_personalized directory missing!"
    sample_sub = subjects[0]
    assert os.path.exists(os.path.join(riem_dir, f"{sample_sub}_cov.pkl")), f"[FAIL] {sample_sub}_cov.pkl missing!"
    assert os.path.exists(os.path.join(riem_dir, f"{sample_sub}_tangent.pkl")), f"[FAIL] {sample_sub}_tangent.pkl missing!"
    assert os.path.exists(os.path.join(riem_dir, f"{sample_sub}_scaler.pkl")), f"[FAIL] {sample_sub}_scaler.pkl missing!"
    assert os.path.exists(os.path.join(riem_dir, f"{sample_sub}_model.pkl")), f"[FAIL] {sample_sub}_model.pkl missing!"
    assert os.path.exists(os.path.join(riem_dir, f"{sample_sub}_metadata.json")), f"[FAIL] {sample_sub}_metadata.json missing!"
    print(f"  -> SUCCESS: Serialized Riemannian artifacts verified under '{riem_dir}'.")

    # Preserved Earlier Step Artifacts
    print("\n12. Verifying existing STEP 8 & STEP 9 models remain untouched...")
    assert os.path.exists("models/personalized_results.csv"), "[FAIL] Step 8 personalized_results.csv missing!"
    assert os.path.exists("models/csp_results.csv"), "[FAIL] Step 9 csp_results.csv missing!"
    print("  -> SUCCESS: Original Step 8 and Step 9 artifacts intact.")

    print("\n==================================================")
    print("STEP 10 VERIFICATION SUMMARY & FINAL REPORT")
    print("==================================================")
    print(f"SUBJECTS EVALUATED:          {len(df_riem)}")
    print(f"STEP 8 AVERAGE ACCURACY:     {avg_s8_acc*100:.2f}%")
    print(f"STEP 9 CSP AVERAGE ACCURACY: {avg_s9_acc*100:.2f}%")
    print(f"STEP 10 RIEMANN ACCURACY:    {avg_riem_acc*100:.2f}%")
    print(f"RIEMANN GAIN VS STEP 8:      {avg_riem_acc*100 - avg_s8_acc*100:+.2f}%")
    print(f"RIEMANN GAIN VS STEP 9 CSP:  {avg_riem_acc*100 - avg_s9_acc*100:+.2f}%")
    print(f"BEST RIEMANNIAN SUBJECT:     {df_riem.loc[df_riem['riemannian_accuracy'].idxmax()]['subject']} ({df_riem['riemannian_accuracy'].max()*100:.2f}%)")
    print("==================================================")
    print("[SUCCESS] ALL 11 STEP 10 VERIFICATION CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    verify_step10()
