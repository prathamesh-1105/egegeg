import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.train_model import (
    load_and_validate_dataset,
    subject_train_test_split,
    train_and_evaluate_models,
    predict_eeg,
    FEATURE_COLUMNS,
    EXPECTED_CLASSES
)


def verify_step5():
    print("==================================================")
    print("RUNNING STEP 5 MODEL TRAINING VERIFICATION CHECKS")
    print("==================================================")

    csv_path = "data/processed/eeg_features.csv"
    
    # 1. CSV loads correctly
    print("\n1. Verifying feature dataset loads correctly...")
    df = load_and_validate_dataset(csv_path)
    print(f"  -> SUCCESS: Dataset loaded with {len(df)} total rows.")

    # 2. Correct 13 features exist
    print("\n2. Verifying 13 mathematical features exist...")
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"[FAIL] Missing feature column: {col}"
    print(f"  -> SUCCESS: All 13 feature columns verified.")

    # 3 & 5. Subject-based split & unseen test subjects
    print("\n3 & 5. Verifying subject-based train/test split...")
    df_train, df_test, train_subs, test_subs = subject_train_test_split(df, train_ratio=0.8)
    overlap = set(train_subs).intersection(set(test_subs))
    assert len(overlap) == 0, f"[FAIL] Subject overlap found: {overlap}"
    print(f"  -> SUCCESS: Train subjects ({len(train_subs)}): {train_subs}")
    print(f"              Test subjects  ({len(test_subs)}):  {test_subs}")
    print(f"              Train windows: {len(df_train)}, Test windows: {len(df_test)}")

    # 4, 6, 7, 8, 9. Train models, compute real metrics & confusion matrix, save artifacts
    print("\n4, 6, 7, 8, 9. Training models and evaluating on unseen test subjects...")
    res = train_and_evaluate_models(csv_path=csv_path, models_dir="models")

    assert os.path.exists("models/eeg_model.pkl"), "[FAIL] eeg_model.pkl not saved"
    assert os.path.exists("models/scaler.pkl"), "[FAIL] scaler.pkl not saved"
    assert os.path.exists("models/model_metadata.json"), "[FAIL] model_metadata.json not saved"

    print(f"  -> SUCCESS: Best Model Selected: '{res['model_name']}'")
    print(f"     - Test Accuracy:  {res['best_metrics']['accuracy']*100:.2f}%")
    print(f"     - Test Precision: {res['best_metrics']['precision']*100:.2f}%")
    print(f"     - Test Recall:    {res['best_metrics']['recall']*100:.2f}%")
    print(f"     - Test F1-Score:  {res['best_metrics']['f1']*100:.2f}%")

    print("\n   SVM Confusion Matrix (Actual vs Predicted):")
    print(np.array(res["svm_metrics"]["confusion_matrix"]))

    print("\n   Random Forest Confusion Matrix (Actual vs Predicted):")
    print(np.array(res["rf_metrics"]["confusion_matrix"]))

    # 10. Test prediction function on a real unseen test window
    print("\n10. Testing predict_eeg() on a real unseen test window from test subject...")
    test_row = df_test.iloc[0]
    real_label = test_row["label"]
    test_feat_dict = {col: test_row[col] for col in FEATURE_COLUMNS}

    pred_res = predict_eeg(test_feat_dict, models_dir="models")

    print(f"  -> Real Test Subject Window (Subject: {test_row['subject']}, Run: {test_row['run']})")
    print(f"     Actual Class:    {real_label}")
    print(f"     Predicted Class: {pred_res['predicted_class']}")
    print(f"     Confidence:      {pred_res['confidence']:.4f}" if pred_res['confidence'] else "     Confidence: N/A")
    print(f"     Probabilities:   {pred_res['probabilities']}")

    assert pred_res["predicted_class"] in EXPECTED_CLASSES, "[FAIL] Invalid class returned"
    print("\n==================================================")
    print("STEP 5 MODEL TRAINING SUMMARY & FINAL REPORT")
    print("==================================================")
    print(f"Model Trained:           {res['model_name']}")
    print(f"Training Subjects:       {res['training_subjects']}")
    print(f"Test Subjects:           {res['testing_subjects']}")
    print(f"Training Windows:        {res['dataset_info']['train_windows']}")
    print(f"Test Windows:            {res['dataset_info']['test_windows']}")
    print(f"SVM Test Accuracy:       {res['svm_metrics']['accuracy']*100:.2f}%")
    print(f"Random Forest Accuracy:  {res['rf_metrics']['accuracy']*100:.2f}%")
    print(f"Best Model Test F1:      {res['best_metrics']['f1']*100:.2f}%")
    print("Saved Files:")
    print("  - models/eeg_model.pkl")
    print("  - models/scaler.pkl")
    print("  - models/model_metadata.json")
    print("==================================================")
    print("[SUCCESS] ALL STEP 5 VERIFICATION CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    verify_step5()
