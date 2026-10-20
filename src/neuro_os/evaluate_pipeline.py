"""
NeuroOS Phase 2A — Comprehensive Holdout Evaluation Protocol (R11).
Evaluates the Step 60/61 Riemannian-FBCSP Engine across all 10 PhysioNet subjects on Run 11 (300 trials).
Computes:
1. Raw Classifier Metrics: Accuracy, Macro F1, Per-class Precision/Recall/F1, Confusion Matrix.
2. Safety Guardrail Metrics: Rejections, Coverage, Selective Accuracy.
3. Command Execution Metrics: Dispatched Commands, Correct vs. Incorrect Commands, REST Neutrality.
4. Measured Latency: Average Inference and Pipeline Latency (ms).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

sys.path.insert(0, os.path.abspath("."))

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.step33_pipeline import CLASSES
from src.neuro_os.safety_shield import NeuroOSSafetyShield
from src.neuro_os.command_router import NeuroOSCommandRouter
from src.neuro_os.action_executor import NeuroOSActionExecutor


def run_comprehensive_evaluation(data_dir: str = "data") -> Dict[str, Any]:
    """
    Executes zero-leakage evaluation across subjects S001-S010 on Run 11.
    Separates raw ML classification performance from safety-shielded command execution.
    """
    from api_server import predict_eeg_window, get_cached_subject_data

    subjects = [f"S{i:03d}" for i in range(1, 11)]
    run_id = "R11"

    all_true = []
    all_pred = []
    all_conf = []
    all_latencies = []

    # Safety and routing tracking
    safety_shield = NeuroOSSafetyShield(confidence_thresh=0.60, margin_thresh=0.10, smoothing_window_size=1)
    command_router = NeuroOSCommandRouter(dwell_windows=3, cooldown_sec=0.0, command_mode="SCROLL")
    action_executor = NeuroOSActionExecutor(mode="SCROLL_DEMO")

    accepted_flags = []
    rejection_reasons = []
    dispatched_commands = []
    executed_flags = []
    executed_correct = []

    total_trials = 0
    t_start = time.perf_counter()

    for sub_id in subjects:
        try:
            X, y, info = get_cached_subject_data(sub_id, run_id)
        except Exception as e:
            print(f"Warning: Could not load data for {sub_id} {run_id}: {e}")
            continue

        n_trials = len(y)
        for t_idx in range(n_trials):
            total_trials += 1
            t_eval_0 = time.perf_counter()
            pred_res = predict_eeg_window(subject_id=sub_id, run_id=run_id, trial_idx=t_idx)
            t_eval_1 = time.perf_counter()

            inf_lat = (t_eval_1 - t_eval_0) * 1000.0
            all_latencies.append(inf_lat)

            true_cls = str(y[t_idx])
            pred_cls = str(pred_res["predicted_label"])
            conf = float(pred_res["confidence"])

            all_true.append(true_cls)
            all_pred.append(pred_cls)
            all_conf.append(conf)

            # Pass through Safety Shield
            safety_res = safety_shield.evaluate_safety(
                raw_probs=pred_res["probabilities"],
                hardware_connected=True,
                signal_quality_ok=True
            )

            is_accepted = bool(safety_res["accepted"])
            accepted_flags.append(is_accepted)
            rejection_reasons.append(safety_res.get("reason", "NONE"))

            # Pass through Command Router
            routing_res = command_router.process_intent(safety_res)
            cmd = routing_res.get("command", "NEUTRAL")
            dispatched_commands.append(cmd)

            # Pass through Action Executor
            act_res = action_executor.execute_command(cmd)
            was_executed = bool(act_res.get("executed", False))
            executed_flags.append(was_executed)

            # Check execution correctness
            # SCROLL_UP is correct if true class is LEFT_FIST
            # SCROLL_DOWN is correct if true class is RIGHT_FIST
            # NEUTRAL is correct if true class is REST
            if was_executed:
                if cmd == "SCROLL_UP" and true_cls == "LEFT_FIST":
                    executed_correct.append(True)
                elif cmd == "SCROLL_DOWN" and true_cls == "RIGHT_FIST":
                    executed_correct.append(True)
                else:
                    executed_correct.append(False)
            else:
                # No action executed
                if true_cls == "REST":
                    # Correctly stationary during rest
                    executed_correct.append(True)
                else:
                    # Missed motor opportunity due to rejection or neutral
                    executed_correct.append(False)

    t_total = time.perf_counter() - t_start

    # 1. Raw Classifier Evaluation
    acc = float(accuracy_score(all_true, all_pred) * 100.0)
    prec, rec, f1, sup = precision_recall_fscore_support(all_true, all_pred, labels=CLASSES, zero_division=0)
    macro_f1 = float(np.mean(f1) * 100.0)

    cm = confusion_matrix(all_true, all_pred, labels=CLASSES).tolist()

    class_metrics = {}
    for i, c in enumerate(CLASSES):
        class_metrics[c] = {
            "precision": round(float(prec[i]) * 100.0, 2),
            "recall": round(float(rec[i]) * 100.0, 2),
            "f1_score": round(float(f1[i]) * 100.0, 2),
            "support": int(sup[i])
        }

    # 2. Safety & Coverage Evaluation
    total_rejections = int(sum(1 for a in accepted_flags if not a))
    total_accepted = int(sum(1 for a in accepted_flags if a))
    coverage_pct = round((total_accepted / max(1, total_trials)) * 100.0, 2)

    # Selective Accuracy on accepted trials only
    accepted_indices = [i for i, a in enumerate(accepted_flags) if a]
    if accepted_indices:
        sel_true = [all_true[i] for i in accepted_indices]
        sel_pred = [all_pred[i] for i in accepted_indices]
        selective_acc = round(float(accuracy_score(sel_true, sel_pred) * 100.0), 2)
    else:
        selective_acc = 0.0

    # 3. Command Execution Statistics
    cmd_counts = {
        "SCROLL_UP": int(sum(1 for c in dispatched_commands if c == "SCROLL_UP")),
        "SCROLL_DOWN": int(sum(1 for c in dispatched_commands if c == "SCROLL_DOWN")),
        "NEUTRAL": int(sum(1 for c in dispatched_commands if c == "NEUTRAL"))
    }
    total_executed_scrolls = cmd_counts["SCROLL_UP"] + cmd_counts["SCROLL_DOWN"]
    
    # Correct scroll commands (true motor matched dispatched direction)
    correct_scrolls = 0
    incorrect_scrolls = 0
    for i in range(total_trials):
        cmd = dispatched_commands[i]
        tr = all_true[i]
        if cmd == "SCROLL_UP":
            if tr == "LEFT_FIST":
                correct_scrolls += 1
            else:
                incorrect_scrolls += 1
        elif cmd == "SCROLL_DOWN":
            if tr == "RIGHT_FIST":
                correct_scrolls += 1
            else:
                incorrect_scrolls += 1

    scroll_accuracy = round((correct_scrolls / max(1, total_executed_scrolls)) * 100.0, 2) if total_executed_scrolls > 0 else 0.0

    # REST Neutrality: percentage of REST trials that produced NO scrolling
    rest_indices = [i for i, tr in enumerate(all_true) if tr == "REST"]
    rest_neutral_count = sum(1 for i in rest_indices if dispatched_commands[i] == "NEUTRAL")
    rest_neutrality_pct = round((rest_neutral_count / max(1, len(rest_indices))) * 100.0, 2) if rest_indices else 100.0

    # 4. Latency
    avg_latency = round(float(np.mean(all_latencies)), 2)
    p95_latency = round(float(np.percentile(all_latencies, 95)), 2)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "PhysioNet EEGMMIDB R11 Untouched Holdout (Zero Leakage)",
        "subjects_evaluated": len(subjects),
        "total_trials_processed": total_trials,
        "raw_classifier_metrics": {
            "accuracy_pct": round(acc, 2),
            "macro_f1_pct": round(macro_f1, 2),
            "class_metrics": class_metrics,
            "confusion_matrix": cm,
            "classes": CLASSES
        },
        "safety_guardrail_metrics": {
            "total_accepted": total_accepted,
            "total_rejections": total_rejections,
            "coverage_pct": coverage_pct,
            "selective_accuracy_pct": selective_acc
        },
        "command_execution_metrics": {
            "dispatched_commands": cmd_counts,
            "total_executed_scrolls": total_executed_scrolls,
            "correct_scrolls": correct_scrolls,
            "incorrect_scrolls": incorrect_scrolls,
            "scroll_command_accuracy_pct": scroll_accuracy,
            "rest_trials_count": len(rest_indices),
            "rest_neutral_count": rest_neutral_count,
            "rest_neutrality_pct": rest_neutrality_pct
        },
        "latency_metrics": {
            "average_inference_ms": avg_latency,
            "p95_inference_ms": p95_latency,
            "total_benchmark_time_sec": round(t_total, 2)
        }
    }

    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", "evaluation_report_r11.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = run_comprehensive_evaluation()
    print("\n" + "=" * 60)
    print("NEURO-OS PHASE 2A EMPIRICAL EVALUATION REPORT (R11)")
    print("=" * 60)
    print(f"Total Trials Processed     : {rep['total_trials_processed']}")
    print(f"Raw Holdout Accuracy       : {rep['raw_classifier_metrics']['accuracy_pct']}%")
    print(f"Macro F1 Score             : {rep['raw_classifier_metrics']['macro_f1_pct']}%")
    print(f"REST Recall                : {rep['raw_classifier_metrics']['class_metrics']['REST']['recall']}%")
    print(f"LEFT_FIST Recall           : {rep['raw_classifier_metrics']['class_metrics']['LEFT_FIST']['recall']}%")
    print(f"RIGHT_FIST Recall          : {rep['raw_classifier_metrics']['class_metrics']['RIGHT_FIST']['recall']}%")
    print(f"Confusion Matrix (R, L, R) : {rep['raw_classifier_metrics']['confusion_matrix']}")
    print("-" * 60)
    print(f"Safety Rejections          : {rep['safety_guardrail_metrics']['total_rejections']} / {rep['total_trials_processed']}")
    print(f"Safety Coverage            : {rep['safety_guardrail_metrics']['coverage_pct']}%")
    print(f"Selective Accuracy         : {rep['safety_guardrail_metrics']['selective_accuracy_pct']}%")
    print("-" * 60)
    print(f"Total Scrolling Actions    : {rep['command_execution_metrics']['total_executed_scrolls']}")
    print(f"Correct Scrolls            : {rep['command_execution_metrics']['correct_scrolls']}")
    print(f"Incorrect Scrolls          : {rep['command_execution_metrics']['incorrect_scrolls']}")
    print(f"Scroll Command Accuracy    : {rep['command_execution_metrics']['scroll_command_accuracy_pct']}%")
    print(f"REST Neutrality Guarantee  : {rep['command_execution_metrics']['rest_neutrality_pct']}%")
    print(f"Average Inference Latency  : {rep['latency_metrics']['average_inference_ms']} ms")
    print("=" * 60)
