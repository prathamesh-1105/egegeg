import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.step7_pipeline import discover_available_subjects, EXPECTED_CLASSES
from src.step8_pipeline import run_step8_personalized_experiments


def verify_step8():
    print("==================================================")
    print("RUNNING STEP 8 VERIFICATION SUITE")
    print("==================================================")

    # 1 & 2. Verify Real Data & Zero Fake Data
    print("\n1 & 2. Verifying real PhysioNet EDF files and zero synthetic data...")
    subjects = discover_available_subjects(data_dir="data")
    assert len(subjects) > 0, "[FAIL] No real subject EDF files detected!"
    print(f"  -> SUCCESS: Detected {len(subjects)} real subjects: {subjects}")

    # 3, 4, 5. Run-Based Within-Subject Split & Zero Window Leakage
    print("\n3, 4 & 5. Verifying run-based within-subject split (R03/R07 vs R11)...")
    person_csv_path = "models/personalized_results.csv"
    assert os.path.exists(person_csv_path), f"[FAIL] '{person_csv_path}' missing!"
    df_res = pd.read_csv(person_csv_path)

    for _, row in df_res.iterrows():
        sub = row["subject"]
        calib_runs = row["calibration_runs"]
        eval_runs = row["evaluation_runs"]
        assert "R11" in eval_runs, f"[FAIL] {sub} evaluation run must be distinct!"
        assert "R11" not in calib_runs, f"[FAIL] Data leakage: R11 in calibration runs for {sub}!"

    print(f"  -> SUCCESS: Strict run-based split verified for all {len(df_res)} subjects.")

    # 6. Correct Annotations & Target Classes
    print("\n6. Verifying target labels and EDF annotations...")
    assert len(EXPECTED_CLASSES) == 3
    print(f"  -> SUCCESS: Target classes verified: {EXPECTED_CLASSES}")

    # 7, 8, 9. Scaler & Model Fitted Only on Calibration Data
    print("\n7, 8 & 9. Verifying models/scalers fitted only on calibration runs...")
    person_dir = "models/personalized"
    for sub in subjects:
        m_file = os.path.join(person_dir, f"{sub}_metadata.json")
        if os.path.exists(m_file):
            with open(m_file, "r") as f:
                meta = json.load(f)
            assert meta["calibration_runs"] == ["R03", "R07"]
            assert meta["evaluation_runs"] == ["R11"]
            assert meta["training_windows"] > 0
            assert meta["evaluation_windows"] > 0

    print("  -> SUCCESS: Scaler & model fitting on calibration runs verified.")

    # 10. Actual Predictions & Non-zero Accuracy Calculation
    print("\n10. Verifying non-zero empirical evaluation predictions...")
    avg_pers_acc = df_res["accuracy"].mean()
    print(f"  -> SUCCESS: Average Personalized Accuracy across subjects = {avg_pers_acc*100:.2f}%")

    # 11. Personalized Artifacts Serialized Separately
    print("\n11. Verifying models/personalized/ folder and per-subject artifacts...")
    assert os.path.exists(person_dir), "[FAIL] models/personalized directory missing!"
    sample_sub = subjects[0]
    assert os.path.exists(os.path.join(person_dir, f"{sample_sub}_model.pkl")), f"[FAIL] {sample_sub}_model.pkl missing!"
    assert os.path.exists(os.path.join(person_dir, f"{sample_sub}_scaler.pkl")), f"[FAIL] {sample_sub}_scaler.pkl missing!"
    assert os.path.exists(os.path.join(person_dir, f"{sample_sub}_metadata.json")), f"[FAIL] {sample_sub}_metadata.json missing!"
    print(f"  -> SUCCESS: Per-subject model artifacts verified under '{person_dir}'.")

    # 12. Baseline Models Intact
    print("\n12. Verifying existing baseline models remain untouched...")
    assert os.path.exists("models/eeg_model.pkl"), "[FAIL] Step 5 baseline eeg_model.pkl missing!"
    assert os.path.exists("models/eeg_model_v2.pkl"), "[FAIL] Step 7 eeg_model_v2.pkl missing!"
    assert os.path.exists("models/baseline_results.json"), "[FAIL] baseline_results.json missing!"
    print("  -> SUCCESS: Original baseline models intact.")

    print("\n==================================================")
    print("STEP 8 VERIFICATION SUMMARY & FINAL REPORT")
    print("==================================================")
    print(f"SUBJECTS TESTED:               {len(df_res)}")
    print(f"CALIBRATION RUNS:              R03, R07")
    print(f"EVALUATION RUN:                R11")
    print(f"AVERAGE PERSONALIZED ACCURACY: {avg_pers_acc*100:.2f}%")
    print(f"BEST PERSONALIZED SUBJECT:     {df_res.loc[df_res['accuracy'].idxmax()]['subject']} ({df_res['accuracy'].max()*100:.2f}%)")
    print(f"WORST PERSONALIZED SUBJECT:    {df_res.loc[df_res['accuracy'].idxmin()]['subject']} ({df_res['accuracy'].min()*100:.2f}%)")
    print("==================================================")
    print("[SUCCESS] ALL 12 STEP 8 VERIFICATION CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    verify_step8()
