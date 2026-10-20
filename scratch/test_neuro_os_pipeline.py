import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import numpy as np

from src.step60_sota_pipeline import load_dataset_step60, extract_custom_filter_bank_windows, extract_fbcsp_features_transform, extract_asymmetry_and_erd_features, align_probs, CLASSES, SEL_MODEL_DIR
import joblib

from src.neuro_os import (
    NeuroOSSafetyShield,
    NeuroOSCommandRouter,
    NeuroOSActionExecutor,
    NeuroOSEventLogger
)


def run_neuro_os_integration_test():
    print("=" * 80)
    print("STARTING NEURO-OS INTEGRATION TEST — END-TO-END BCI CONTROL FLOW")
    print("=" * 80)

    # 1. Initialize NeuroOS Engine Modules
    safety_shield = NeuroOSSafetyShield(confidence_thresh=0.60, margin_thresh=0.12, smoothing_window_size=3)
    command_router = NeuroOSCommandRouter(dwell_windows=3, cooldown_sec=0.5)
    action_executor = NeuroOSActionExecutor(mode="NAVIGATION")
    event_logger = NeuroOSEventLogger(max_history=100)

    # 2. Load Genuine Test Windows (Subject S001, Run R11)
    print("\n--- 1. LOADING GENUINE TEST WINDOWS (S001 R11) ---")
    sub_data = load_dataset_step60(data_dir="data")
    s001_r11_X = sub_data["S001"]["X_r11"]
    s001_r11_y = sub_data["S001"]["y_r11"]

    print(f"Loaded {len(s001_r11_y)} genuine test windows for S001 R11")

    # 3. Load Step 60 Model Artifacts for S001
    s_out = os.path.join(SEL_MODEL_DIR, "S001")
    bank_s = joblib.load(os.path.join(s_out, "adaptive_bank.pkl"))
    csp_d_s = joblib.load(os.path.join(s_out, "csp.pkl"))
    cov_est = joblib.load(os.path.join(s_out, "cov_estimator.pkl"))
    ts_trans = joblib.load(os.path.join(s_out, "tangent_space.pkl"))
    sel_s = joblib.load(os.path.join(s_out, "feature_selector.pkl"))
    sc_s = joblib.load(os.path.join(s_out, "scaler.pkl"))
    clf_s = joblib.load(os.path.join(s_out, "model.pkl"))

    print("\n--- 2. STREAMING PREDICTIONS THROUGH NEURO-OS PIPELINE ---")
    print("Flow: Prediction -> Safety Shield -> Command Router -> Action Executor -> Event Logger\n")

    actions_executed_count = 0

    for i in range(len(s001_r11_y)):
        win_x = s001_r11_X[i:i+1]
        gt_label = s001_r11_y[i]

        # Step 60 Model Inference
        filt_win = extract_custom_filter_bank_windows(win_x, bank_s)
        X_fbcsp = extract_fbcsp_features_transform(filt_win, csp_d_s)
        cov_win = cov_est.transform(win_x)
        X_ts = ts_trans.transform(cov_win)
        X_asym = extract_asymmetry_and_erd_features(win_x)

        X_fused = np.hstack([X_fbcsp, X_ts, X_asym])
        probs_raw = clf_s.predict_proba(sc_s.transform(sel_s.transform(X_fused)))
        probs = align_probs(probs_raw, clf_s.classes_)[0]
        raw_prob_dict = dict(zip(CLASSES, [float(p) for p in probs]))

        # Pass 1: Safety Shield
        safety = safety_shield.evaluate_safety(raw_prob_dict, hardware_connected=True, signal_quality_ok=True)

        # Pass 2: Command Router
        routing = command_router.process_intent(safety)

        # Pass 3: Action Executor
        action_res = {"executed": False}
        if routing["action_triggered"]:
            action_res = action_executor.execute_command(routing["command"])
            actions_executed_count += 1

        # Pass 4: Event Logger
        log_entry = event_logger.log_event(
            event_type="EEG_WINDOW_PROCESSED",
            intent=safety["intent"],
            confidence=safety["confidence"],
            safety_status=safety,
            command=routing["command"],
            action=action_res,
            details=f"True Groundtruth: {gt_label} | Reason: {routing['reason']}"
        )

        if i % 5 == 0 or routing["action_triggered"]:
            status_symbol = "[ACTION]" if routing["action_triggered"] else ("[SAFE]" if safety["accepted"] else "[REJECTED]")
            print(f"Window {i:02d} [{gt_label:10s}] -> Safety: {safety['intent']:10s} (p={safety['confidence']:.2f}) | Cmd: {routing['command']:8s} | {status_symbol} | {routing['reason']}")
            if routing["action_triggered"]:
                print(f"   |--> EXECUTED ACTION: {action_res['action_name']} -> {action_res['details']}")

    print("\n================ NEURO-OS INTEGRATION TEST RESULTS ================")
    print(f" Total EEG Windows Processed : {len(s001_r11_y)}")
    print(f" Total Software Actions Fired: {actions_executed_count}")
    print(f" Active Adapter Mode        : {action_executor.mode}")
    print(f" Total Audit Logs Recorded  : {len(event_logger.logs)}")
    print("===================================================================\n")

    assert len(event_logger.logs) == len(s001_r11_y), "Log count mismatch!"
    print("[PASSED] NEURO-OS INTEGRATION TEST PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_neuro_os_integration_test()
