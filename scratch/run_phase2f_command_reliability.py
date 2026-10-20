"""
NeuroOS Phase 2F — Command Reliability & Replay Analytics Benchmark
Evaluates Step 62 champion model across all 10 PhysioNet subjects on Run 11 (300 holdout trials).

Separates:
1. Raw classifier accuracy (ML decoder) from command execution reliability (Safety Shield + Router).
2. Compares Immediate Execution (dwell=1) vs 2-Window Confirmation (dwell=2) vs 3-Window Confirmation (dwell=3).
3. Quantifies exact trade-offs: command error rate, suppressed actions, REST neutrality, and P95 latencies.

All metrics are computed with explicit, documented denominators.
"""

import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.step33_pipeline import CLASSES, GROUP_B_CHANNELS
from src.neuro_os.session_auditor import NeuroOSSessionAuditor
from src.neuro_os.safety_shield import NeuroOSSafetyShield
from src.neuro_os.command_router import NeuroOSCommandRouter
from src.neuro_os.action_executor import NeuroOSActionExecutor
from src.neuro_os.signal_quality import NeuroOSSignalQualityGate
from api_server import get_cached_subject_data, predict_eeg_window


def run_benchmark_configuration(
    config_name: str,
    dwell_windows: int,
    cooldown_sec: float,
    confidence_thresh: float,
    margin_thresh: float,
    smoothing_window_size: int,
    subjects: List[str] = [f"S{i:03d}" for i in range(1, 11)],
    run_id: str = "R11"
) -> Dict[str, Any]:
    """
    Evaluates one specific safety & routing configuration across all subjects on Run 11.
    """
    auditor = NeuroOSSessionAuditor(max_records=1000)
    auditor.reset(session_id=f"bench_{config_name}")

    safety_shield = NeuroOSSafetyShield(
        confidence_thresh=confidence_thresh,
        margin_thresh=margin_thresh,
        smoothing_window_size=smoothing_window_size,
        rest_emergency_windows=25
    )
    command_router = NeuroOSCommandRouter(
        dwell_windows=dwell_windows,
        temporal_confirmation_windows=dwell_windows,
        cooldown_sec=cooldown_sec,
        profile_id="scrolling"
    )
    action_executor = NeuroOSActionExecutor(mode="SCROLL_DEMO")
    quality_gate = NeuroOSSignalQualityGate()

    sim_time = 1000.0
    dt_step = 3.0  # Simulated 3.0s trial interval

    for sub_id in subjects:
        # Load genuine 30 trials from R11 holdout
        X, y, info = get_cached_subject_data(sub_id, run_id)
        n_trials = len(y)

        # Reset per-subject router and safety smoothing state for zero subject leakage
        safety_shield.reset()
        command_router.reset()

        for t_idx in range(n_trials):
            sim_time += dt_step

            # 1. Predict trial window using frozen Step 62 champion model
            t_ml_start = time.perf_counter()
            pred_res = predict_eeg_window(subject_id=sub_id, run_id=run_id, trial_idx=t_idx)
            t_ml_end = time.perf_counter()
            ml_latency = round((t_ml_end - t_ml_start) * 1000.0, 2)

            raw_probs = pred_res["probabilities"]
            true_label = pred_res["true_label"]

            # 2. Quality Gate
            win_data = X[t_idx:t_idx+1]
            quality_res = quality_gate.evaluate_quality(
                win_data,
                GROUP_B_CHANNELS,
                expected_channels=GROUP_B_CHANNELS
            )

            # 3. Safety Shield Evaluation
            t_pipe_start = time.perf_counter()
            safety_res = safety_shield.evaluate_safety(
                raw_probs=raw_probs,
                hardware_connected=True,
                signal_quality_ok=quality_res["passed"]
            )
            prediction_id = f"pred_{sub_id}_{run_id}_{t_idx}_{int(sim_time)}"
            safety_res["prediction_id"] = prediction_id
            safety_res["timestamp"] = sim_time

            # 4. Command Routing (passing sim_time so router uses simulation clock)
            routing_res = command_router.process_intent(safety_res, current_time=sim_time)

            # 5. Action Execution
            action_res = {
                "executed": False,
                "command": routing_res.get("command", "NEUTRAL"),
                "action": "NONE",
                "action_name": "NONE",
                "details": "No action executed",
                "state_change": {"parameter": "none", "delta": 0}
            }
            if routing_res.get("action_triggered", False):
                action_res = action_executor.execute_command(routing_res["command"])

            t_pipe_end = time.perf_counter()
            pipeline_latency = round(ml_latency + (t_pipe_end - t_pipe_start) * 1000.0, 2)

            # 6. Decoupled Session Audit Record (ground_truth passed only for scoring)
            auditor.record_trial(
                session_id=f"bench_{config_name}",
                predicted_label=pred_res["predicted_label"],
                confidence=pred_res["confidence"],
                probabilities=raw_probs,
                safety_status=safety_res,
                command_routing=routing_res,
                action_execution=action_res,
                signal_quality=quality_res,
                latencies_ms={
                    "ml_latency": ml_latency,
                    "pipeline_latency": pipeline_latency,
                    "ui_broadcast_latency": 0.0
                },
                subject_id=sub_id,
                run_id=run_id,
                trial_idx=t_idx,
                data_source="PHYSIO_NET_R11_HOLDOUT",
                model_version="Step62-SurfaceLaplacian-Ensemble",
                ground_truth_label=true_label,
                timestamp=sim_time
            )

    metrics = auditor.get_summary_metrics()
    metrics["config_name"] = config_name
    metrics["dwell_windows"] = dwell_windows
    metrics["cooldown_sec"] = cooldown_sec
    metrics["confidence_thresh"] = confidence_thresh
    metrics["margin_thresh"] = margin_thresh
    metrics["smoothing_window_size"] = smoothing_window_size
    return metrics


def main():
    print("=" * 80)
    print("NeuroOS Phase 2F — Command Reliability & Analytics Evaluation (Run 11)")
    print("Subjects: S001-S010 | Total Holdout Trials: 300 | Model: Step 62 Frozen Base")
    print("=" * 80)

    configs = [
        {
            "config_name": "Immediate Execution (Dwell=1, w=1)",
            "dwell_windows": 1,
            "cooldown_sec": 0.0,
            "confidence_thresh": 0.50,
            "margin_thresh": 0.05,
            "smoothing_window_size": 1
        },
        {
            "config_name": "Two-Window Confirmation (Dwell=2, w=2)",
            "dwell_windows": 2,
            "cooldown_sec": 0.5,
            "confidence_thresh": 0.60,
            "margin_thresh": 0.10,
            "smoothing_window_size": 2
        },
        {
            "config_name": "Standard Confirmation (Dwell=3, w=3)",
            "dwell_windows": 3,
            "cooldown_sec": 0.8,
            "confidence_thresh": 0.60,
            "margin_thresh": 0.12,
            "smoothing_window_size": 3
        }
    ]

    results = []
    for cfg in configs:
        print(f"\nEvaluating: {cfg['config_name']} ...")
        t0 = time.time()
        res = run_benchmark_configuration(**cfg)
        t_elapsed = time.time() - t0
        print(f"Completed in {t_elapsed:.2f}s.")
        results.append(res)

    # Print formatted comparative table
    print("\n" + "=" * 90)
    print(f"{'Metric':<36} | {'Immediate (D=1)':<16} | {'2-Window (D=2)':<16} | {'Standard (D=3)':<16}")
    print("-" * 90)

    rows = [
        ("Total Processed Trials", "total_trials_processed", "{}"),
        ("Raw Classification Accuracy", "classification_accuracy_pct", "{:.2f}%"),
        ("Macro F1 Score", "macro_f1_pct", "{:.2f}%"),
        ("Safety Rejection Rate", "safety_rejection_rate_pct", "{:.2f}%"),
        ("Accepted-Trial Accuracy", "accepted_trial_accuracy_pct", "{:.2f}%"),
        ("Command Execution Rate", "command_execution_rate_pct", "{:.2f}%"),
        ("Actions Executed (Count)", "actions_executed_count", "{}"),
        ("Incorrect Actions (Count)", "actions_incorrect_count", "{}"),
        ("Incorrect Rate (Total Denom)", "incorrect_command_rate_total_pct", "{:.2f}%"),
        ("Incorrect Rate (Executed Denom)", "incorrect_command_rate_executed_pct", "{:.2f}%"),
        ("Suppressed Misclassifications", "shield_suppressed_errors_count", "{}"),
        ("Suppressed Valid Intentions", "shield_suppressed_correct_count", "{}"),
        ("REST Neutrality Rate", "rest_neutrality_rate_pct", "{:.2f}%"),
        ("Mean Latency (ms)", lambda r: r["latencies_ms"]["mean"], "{:.2f} ms"),
        ("P95 Latency (ms)", lambda r: r["latencies_ms"]["p95"], "{:.2f} ms")
    ]

    for label, key_fn, fmt in rows:
        vals = []
        for r in results:
            val = key_fn(r) if callable(key_fn) else r.get(key_fn)
            vals.append(fmt.format(val) if val is not None else "N/A")
        print(f"{label:<36} | {vals[0]:<16} | {vals[1]:<16} | {vals[2]:<16}")

    print("=" * 90)

    # Save to JSON
    docs_dir = os.path.join(REPO_ROOT, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    out_json = os.path.join(docs_dir, "phase2f_reliability_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed metrics saved to: {out_json}")
    return results


if __name__ == "__main__":
    main()
