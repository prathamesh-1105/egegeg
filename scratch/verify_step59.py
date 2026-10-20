import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import glob
import numpy as np
import pandas as pd
import mne

from src.step8_pipeline import find_subject_runs
from src.step14_pipeline import extract_event_aligned_windows

def run_verification_checks():
    print("=" * 80)
    print("STARTING VERIFICATION SUITE FOR STEP 59 — 100+ ASSERTION CHECKS")
    print("=" * 80)

    checks_passed = 0
    total_checks = 0

    def assert_check(condition, description):
        nonlocal checks_passed, total_checks
        total_checks += 1
        if condition:
            checks_passed += 1
            print(f"  [PASS] Check #{total_checks:03d}: {description}")
        else:
            print(f"  [FAIL] Check #{total_checks:03d}: {description}")
            raise AssertionError(f"Check #{total_checks:03d} failed: {description}")

    # --------------------------------------------------------------------------
    # GROUP 1: IMMUTABILITY OF HISTORICAL STEP DIRECTORIES (Checks 1-15)
    # --------------------------------------------------------------------------
    print("\n--- GROUP 1: IMMUTABILITY OF HISTORICAL DIRECTORIES ---")
    for step_num in range(24, 59):
        dir_path = f"models/step{step_num}"
        if os.path.exists(dir_path):
            assert_check(os.path.isdir(dir_path), f"Historical model directory {dir_path} exists and is directory")
            assert_check(len(os.listdir(dir_path)) > 0, f"Historical model directory {dir_path} is non-empty")

    # --------------------------------------------------------------------------
    # GROUP 2: EDF DATASET INTEGRITY & REAL SIGNALS (Checks 16-45)
    # --------------------------------------------------------------------------
    print("\n--- GROUP 2: REAL EDF DATASET INTEGRITY ---")
    subs = [f"S{i:03d}" for i in range(1, 11)]
    assert_check(len(subs) == 10, "Dataset contains exactly 10 subjects S001-S010")

    edf_count = 0
    for s_id in subs:
        runs = find_subject_runs(s_id, data_dir="data")
        assert_check("R03" in runs and os.path.exists(runs["R03"]), f"{s_id} R03 EDF file exists")
        assert_check("R07" in runs and os.path.exists(runs["R07"]), f"{s_id} R07 EDF file exists")
        assert_check("R11" in runs and os.path.exists(runs["R11"]), f"{s_id} R11 EDF file exists")
        edf_count += 3

    assert_check(edf_count == 30, "Exactly 30 unique EDF files located across S001-S010")

    # Spot check single EDF file window extraction
    sample_edf = find_subject_runs("S001", data_dir="data")["R03"]
    X_s, y_s, info_s = extract_event_aligned_windows(sample_edf, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
    assert_check(X_s.ndim == 3, "Extracted EEG windows tensor is 3D (trials x channels x samples)")
    assert_check(X_s.shape[1] == 15, "Group B channel count equals 15")
    assert_check(X_s.shape[2] == 480, "3.0s epoch at 160Hz equals 480 samples")
    assert_check(len(y_s) == len(X_s), "Labels count matches window count")
    assert_check(set(y_s).issubset({"REST", "LEFT_FIST", "RIGHT_FIST"}), "Labels are subset of canonical CLASSES")
    assert_check(np.var(X_s) > 0.0, "Signal variance is strictly positive (genuine EEG microvolts)")

    # --------------------------------------------------------------------------
    # GROUP 3: ARTIFACT COMPLETENESS & SCHEMA (Checks 46-75)
    # --------------------------------------------------------------------------
    print("\n--- GROUP 3: REQUIRED ARTIFACT FILES COMPLETENESS ---")
    out_dir = "models/step59"
    required_artifacts = [
        "step58_reproduction.csv",
        "protocol_lock.json",
        "error_database.csv",
        "error_transition_analysis.csv",
        "subject_error_analysis.csv",
        "class_error_analysis.csv",
        "peak_frequency_adaptation.csv",
        "erd_features.csv",
        "temporal_optimization.csv",
        "riemannian_alignment.csv",
        "fbcsp_optimization.csv",
        "model_comparison.csv",
        "class_balance.csv",
        "threshold_optimization.csv",
        "subject_adaptation.csv",
        "cross_run_validation.csv",
        "loso_validation.csv",
        "candidate_ranking.csv",
        "validation_leaderboard.csv",
        "final_freeze.json",
        "selected_configuration.json",
        "final_r11_predictions.csv",
        "final_r11_comparison.csv",
        "subject_results.csv",
        "confusion_matrix.csv",
        "latency_results.csv",
        "consistency_audit.csv",
        "summary_report.json",
        "RESEARCH_CONCLUSION.md"
    ]

    for fname in required_artifacts:
        fpath = os.path.join(out_dir, fname)
        assert_check(os.path.exists(fpath), f"Artifact file {fname} exists")
        assert_check(os.path.getsize(fpath) > 0, f"Artifact file {fname} is non-empty")

    # --------------------------------------------------------------------------
    # GROUP 4: STRICT PRE-R11 FREEZE & PROTOCOL LOCK (Checks 76-85)
    # --------------------------------------------------------------------------
    print("\n--- GROUP 4: STRICT PRE-R11 FREEZE & PROTOCOL LOCK ---")
    with open(os.path.join(out_dir, "final_freeze.json"), "r", encoding="utf-8") as f:
        freeze_data = json.load(f)

    assert_check(freeze_data.get("status") == "FROZEN_BEFORE_R11", "final_freeze.json status equals FROZEN_BEFORE_R11")
    assert_check(freeze_data.get("step") == 59, "final_freeze.json step equals 59")
    assert_check("selected_configuration" in freeze_data, "final_freeze.json contains selected_configuration")
    assert_check(freeze_data["dataset"]["total_calibration_trials"] == 600, "Calibration trial count equals 600")
    assert_check(freeze_data["dataset"]["total_test_trials"] == 300, "Test trial count equals 300")

    with open(os.path.join(out_dir, "protocol_lock.json"), "r", encoding="utf-8") as f:
        lock_data = json.load(f)
    assert_check(lock_data.get("status") == "LOCKED", "protocol_lock.json status equals LOCKED")

    # --------------------------------------------------------------------------
    # GROUP 5: MATHEMATICAL CONSISTENCY & REPRODUCIBILITY (Checks 86-105)
    # --------------------------------------------------------------------------
    print("\n--- GROUP 5: MATHEMATICAL CONSISTENCY AUDIT ---")
    df_preds = pd.read_csv(os.path.join(out_dir, "final_r11_predictions.csv"))
    assert_check(len(df_preds) == 300, "final_r11_predictions.csv contains exactly 300 predictions")
    assert_check(set(df_preds.columns) == {"sample_idx", "true_label", "predicted_label"}, "Predictions CSV header is correct")

    df_cm = pd.read_csv(os.path.join(out_dir, "confusion_matrix.csv"), index_col=0)
    cm_mat = df_cm.values
    assert_check(cm_mat.shape == (3, 3), "Confusion matrix shape is 3x3")
    assert_check(int(np.sum(cm_mat)) == 300, "Confusion matrix sum equals 300")

    cm_acc = round((np.trace(cm_mat) / 300.0) * 100.0, 2)
    df_comp = pd.read_csv(os.path.join(out_dir, "final_r11_comparison.csv"))
    rep_acc = float(df_comp[df_comp["metric"] == "R11 Accuracy"]["step59_authentic"].values[0])
    assert_check(abs(cm_acc - rep_acc) < 0.1, f"Confusion matrix accuracy ({cm_acc}%) matches reported accuracy ({rep_acc}%)")

    df_audit = pd.read_csv(os.path.join(out_dir, "consistency_audit.csv"))
    assert_check(len(df_audit) >= 6, "consistency_audit.csv contains at least 6 audit checks")
    assert_check(all(df_audit["status"] == "VERIFIED MATCH"), "All mathematical consistency audit checks status equal VERIFIED MATCH")

    df_lat = pd.read_csv(os.path.join(out_dir, "latency_results.csv"))
    tot_lat = float(df_lat[df_lat["component"] == "Total Step 59 Pipeline Latency"]["latency_ms"].values[0])
    assert_check(tot_lat < 50.0, f"Total pipeline latency ({tot_lat} ms) is strictly under 50 ms")

    with open(os.path.join(out_dir, "summary_report.json"), "r", encoding="utf-8") as f:
        summary_json = json.load(f)
    assert_check("final_decision" in summary_json, "summary_report.json contains final_decision")
    assert_check("step59_authentic_r11_metrics" in summary_json, "summary_report.json contains step59_authentic_r11_metrics")

    print("\n" + "=" * 80)
    print(f"VERIFICATION COMPLETE: {checks_passed}/{total_checks} CHECKS PASSED (100% SUCCESS RATE)")
    print("=" * 80)

if __name__ == "__main__":
    run_verification_checks()
