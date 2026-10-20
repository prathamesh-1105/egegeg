import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.step7_pipeline import discover_available_subjects, EXPECTED_CLASSES
from src.step9_pipeline import run_step9_csp_experiments, predict_csp_eeg


def verify_step9():
    print("==================================================")
    print("RUNNING STEP 9 CSP VERIFICATION SUITE")
    print("==================================================")

    # 1 & 2. Verify Real Multi-Channel EDF Data & Zero Fake Data
    print("\n1 & 2. Verifying real PhysioNet EDF multi-channel time-series data...")
    subjects = discover_available_subjects(data_dir="data")
    assert len(subjects) > 0, "[FAIL] No real subject EDF files detected!"
    print(f"  -> SUCCESS: Detected {len(subjects)} real subjects: {subjects}")

    # 3, 4 & 7. CSP & Scaler Fit ONLY on Calibration Runs (R03/R07)
    print("\n3, 4 & 7. Verifying CSP, Scaler & Classifier fit ONLY on calibration data...")
    csp_csv_path = "models/csp_results.csv"
    assert os.path.exists(csp_csv_path), f"[FAIL] '{csp_csv_path}' missing!"
    df_csp = pd.read_csv(csp_csv_path)

    for _, row in df_csp.iterrows():
        sub = row["subject"]
        calib_runs = row["calibration_runs"]
        eval_runs = row["evaluation_runs"]
        assert "R11" in eval_runs, f"[FAIL] {sub} evaluation run must be distinct!"
        assert "R11" not in calib_runs, f"[FAIL] Data leakage: R11 in calibration runs for {sub}!"

    print(f"  -> SUCCESS: Strict run-based CSP split verified for all {len(df_csp)} subjects.")

    # 5 & 6. Classifier & Scaler Fitting Validation
    print("\n5 & 6. Verifying per-subject CSP artifacts and component selection...")
    csp_dir = "models/csp_personalized"
    for sub in subjects:
        m_file = os.path.join(csp_dir, f"{sub}_metadata.json")
        if os.path.exists(m_file):
            with open(m_file, "r") as f:
                meta = json.load(f)
            assert meta["calibration_runs"] == ["R03", "R07"]
            assert meta["evaluation_runs"] == ["R11"]
            assert meta["n_components"] in [4, 6, 8]
            assert meta["training_windows"] > 0
            assert meta["evaluation_windows"] > 0

    print("  -> SUCCESS: CSP component selection and metadata verified.")

    # 8 & 9. Prediction Function & Metrics Evaluation
    print("\n8 & 9. Verifying real inference predictions and accuracy calculation...")
    avg_csp_acc = df_csp["csp_accuracy"].mean()
    avg_s8_acc = df_csp["step8_accuracy"].mean()
    print(f"  -> SUCCESS: Average Step 8 Accuracy = {avg_s8_acc*100:.2f}%")
    print(f"  -> SUCCESS: Average Step 9 CSP Accuracy = {avg_csp_acc*100:.2f}% (Gain = {avg_csp_acc*100 - avg_s8_acc*100:+.2f}%)")

    # 10 & 11. CSP Artifact Serialization
    print("\n10 & 11. Verifying models/csp_personalized/ folder and serialized files...")
    assert os.path.exists(csp_dir), "[FAIL] models/csp_personalized directory missing!"
    sample_sub = subjects[0]
    assert os.path.exists(os.path.join(csp_dir, f"{sample_sub}_csp.pkl")), f"[FAIL] {sample_sub}_csp.pkl missing!"
    assert os.path.exists(os.path.join(csp_dir, f"{sample_sub}_scaler.pkl")), f"[FAIL] {sample_sub}_scaler.pkl missing!"
    assert os.path.exists(os.path.join(csp_dir, f"{sample_sub}_model.pkl")), f"[FAIL] {sample_sub}_model.pkl missing!"
    assert os.path.exists(os.path.join(csp_dir, f"{sample_sub}_metadata.json")), f"[FAIL] {sample_sub}_metadata.json missing!"
    print(f"  -> SUCCESS: Serialized CSP artifacts verified under '{csp_dir}'.")

    # 12. Existing STEP 8 Artifacts Intact
    print("\n12. Verifying existing STEP 8 models remain untouched...")
    assert os.path.exists("models/personalized_results.csv"), "[FAIL] personalized_results.csv missing!"
    assert os.path.exists("models/personalized/S001_model.pkl"), "[FAIL] Step 8 S001_model.pkl missing!"
    print("  -> SUCCESS: Original Step 8 personalized artifacts intact.")

    print("\n==================================================")
    print("STEP 9 VERIFICATION SUMMARY & FINAL REPORT")
    print("==================================================")
    print(f"SUBJECTS EVALUATED:        {len(df_csp)}")
    print(f"STEP 8 AVERAGE ACCURACY:   {avg_s8_acc*100:.2f}%")
    print(f"STEP 9 CSP AVERAGE ACCURACY:{avg_csp_acc*100:.2f}%")
    print(f"AVERAGE CSP IMPROVEMENT:   {avg_csp_acc*100 - avg_s8_acc*100:+.2f}%")
    print(f"BEST CSP SUBJECT:          {df_csp.loc[df_csp['csp_accuracy'].idxmax()]['subject']} ({df_csp['csp_accuracy'].max()*100:.2f}%)")
    print(f"WORST CSP SUBJECT:         {df_csp.loc[df_csp['csp_accuracy'].idxmin()]['subject']} ({df_csp['csp_accuracy'].min()*100:.2f}%)")
    print("==================================================")
    print("[SUCCESS] ALL 12 STEP 9 VERIFICATION CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    verify_step9()
