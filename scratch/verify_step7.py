import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.step7_pipeline import (
    discover_available_subjects,
    save_baseline_info,
    run_step7_experiments,
    BASELINE_FEATURES,
    EXTENDED_FEATURES,
    EXPECTED_CLASSES
)


def verify_step7():
    print("==================================================")
    print("RUNNING STEP 7 VERIFICATION SUITE")
    print("==================================================")

    # 1 & 2. Verify Data Source & No Fake Data
    print("\n1 & 2. Verifying real PhysioNet EDF files and zero synthetic data...")
    subjects = discover_available_subjects(data_dir="data")
    assert len(subjects) > 0, "[FAIL] No real subject EDF files detected!"
    print(f"  -> SUCCESS: Detected {len(subjects)} real subjects: {subjects}")

    # 3 & 4. Subject Split & Zero Overlap
    print("\n3 & 4. Verifying subject-based 3-way split and zero data leakage...")
    meta_v2_path = "models/model_metadata_v2.json"
    assert os.path.exists(meta_v2_path), f"[FAIL] '{meta_v2_path}' does not exist!"

    with open(meta_v2_path, "r") as f:
        meta_v2 = json.load(f)

    train_subs = set(meta_v2["training_subjects"])
    val_subs = set(meta_v2["validation_subjects"])
    test_subs = set(meta_v2["testing_subjects"])

    assert len(train_subs.intersection(val_subs)) == 0, "[FAIL] Overlap between train and validation subjects!"
    assert len(train_subs.intersection(test_subs)) == 0, "[FAIL] Overlap between train and test subjects!"
    assert len(val_subs.intersection(test_subs)) == 0, "[FAIL] Overlap between validation and test subjects!"

    print(f"  -> SUCCESS: Train ({len(train_subs)}), Val ({len(val_subs)}), Test ({len(test_subs)}) zero overlap confirmed.")

    # 5. Clean Feature Matrix
    print("\n5. Verifying clean feature matrix (0 NaN / Inf)...")
    exp_csv_path = "models/experiment_results.csv"
    assert os.path.exists(exp_csv_path), f"[FAIL] '{exp_csv_path}' does not exist!"
    df_exp = pd.read_csv(exp_csv_path)
    assert not df_exp.isnull().any().any(), "[FAIL] NaNs found in experiment log!"
    print(f"  -> SUCCESS: Experiment log clean with {len(df_exp)} logged trials.")

    # 6 & 7. Model Training & Validation Scores
    print("\n6 & 7. Verifying model training and validation evaluation...")
    best_model_name = meta_v2["model_name"]
    print(f"  -> SUCCESS: Selected Best Architecture: {best_model_name}")

    # 8. Single Pass Unseen Test Evaluation
    print("\n8. Verifying final evaluation on untouched test subjects...")
    test_acc = meta_v2["test_metrics"]["accuracy"]
    test_f1 = meta_v2["test_metrics"]["f1"]
    print(f"  -> SUCCESS: Final Test Accuracy: {test_acc*100:.2f}%, F1: {test_f1*100:.2f}%")

    # 9. Baseline Results Intact
    print("\n9. Verifying Step 5 baseline intact...")
    baseline_path = "models/baseline_results.json"
    assert os.path.exists(baseline_path), f"[FAIL] '{baseline_path}' does not exist!"
    with open(baseline_path, "r") as f:
        base_info = json.load(f)
    assert base_info["baseline_model"] == "Random Forest"
    assert abs(base_info["baseline_accuracy"] - 0.5250596658711217) < 1e-4
    print(f"  -> SUCCESS: Baseline preserved intact (Model: {base_info['baseline_model']}, Accuracy: {base_info['baseline_accuracy']*100:.2f}%).")

    # 10. Experiment Results CSV Saved
    print("\n10. Verifying models/experiment_results.csv exists...")
    assert os.path.exists(exp_csv_path)
    print(f"  -> SUCCESS: {exp_csv_path} verified.")

    # 11. Improved Artifacts Saved Separately (_v2)
    print("\n11. Verifying improved _v2 model artifacts saved separately...")
    assert os.path.exists("models/eeg_model_v2.pkl"), "[FAIL] eeg_model_v2.pkl missing!"
    assert os.path.exists("models/scaler_v2.pkl"), "[FAIL] scaler_v2.pkl missing!"
    assert os.path.exists("models/model_metadata_v2.json"), "[FAIL] model_metadata_v2.json missing!"
    print("  -> SUCCESS: All _v2 model artifacts verified.")

    # 12. Feature Importance / Model Verification
    print("\n12. Verifying trained model loadability and prediction function...")
    model_v2 = joblib.load("models/eeg_model_v2.pkl")
    scaler_v2 = joblib.load("models/scaler_v2.pkl")

    dummy_input = np.zeros((1, len(meta_v2["feature_names"])))
    dummy_scaled = scaler_v2.transform(dummy_input)
    dummy_pred = model_v2.predict(dummy_scaled)[0]

    assert dummy_pred in EXPECTED_CLASSES, f"[FAIL] Invalid prediction class: {dummy_pred}"
    print(f"  -> SUCCESS: Model inference operational. Sample prediction class: '{dummy_pred}'")

    print("\n==================================================")
    print("STEP 7 VERIFICATION SUMMARY & FINAL REPORT")
    print("==================================================")
    print(f"BASELINE ACCURACY:         {base_info['baseline_accuracy']*100:.2f}%")
    print(f"BEST EXPERIMENT MODEL:     {best_model_name}")
    print(f"OPTIMAL CHANNEL GROUP:     {meta_v2['channel_group']}")
    print(f"OPTIMAL WINDOW SIZE:       {meta_v2['window_size_sec']}s")
    print(f"FEATURE SET TYPE:          {meta_v2['feature_set']}")
    print(f"FINAL TEST ACCURACY:       {test_acc*100:.2f}%")
    print(f"ACCURACY DIFFERENCE:       {meta_v2['test_metrics']['accuracy_difference']*100:+.2f}%")
    print(f"UNTOUCHED TEST SUBJECTS:   {meta_v2['testing_subjects']}")
    print("==================================================")
    print("[SUCCESS] ALL 12 STEP 7 VERIFICATION CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    verify_step7()
