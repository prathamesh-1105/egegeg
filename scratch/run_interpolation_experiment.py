"""
NeuroOS Phase 6 Controlled Offline Interpolation Robustness Experiment
Evaluates Step 60 performance on 300 genuine unseen R11 holdout trials (S001-S010):
1. Benchmark A: Original 15 Physical Channels (Ground Truth)
2. Benchmark B: Simulated 8-Channel OpenBCI Subset -> 8-to-15 Spherical Spline Reconstruction
"""

import os
import sys
import time
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

sys.path.insert(0, os.path.abspath("."))

from api_server import get_cached_subject_data, load_subject_model, CLASSES, C3_IDX, C4_IDX
from src.step59_pipeline import extract_custom_filter_bank_windows, align_probs
from src.step27_pipeline import extract_fbcsp_features_transform
from src.neuro_os.channel_adapter import OpenBCIChannelAdapter


def run_interpolation_experiment():
    print("================================================================================")
    print("STARTING CONTROLLED OFFLINE INTERPOLATION EXPERIMENT (300 R11 HOLDOUT TRIALS)")
    print("================================================================================")

    adapter = OpenBCIChannelAdapter()

    subjects = [f"S{i:03d}" for i in range(1, 11)]
    
    gt_y_true = []
    gt_y_pred = []
    gt_latencies = []

    rec_y_pred = []
    rec_latencies = []

    # Mapping from PhysioNet EDF 15-channel Group B to 8 OpenBCI Cyton channels
    # Physical channels present in Group B: C3, Cz, C4.
    # Channels in standard 8-ch Cyton: F3, F4, C3, Cz, C4, P3, Pz, P4.
    # Group B channels: FC3, FC1, FCz, FC2, FC4, C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4.
    
    # Standard 10-20 channel index mapping for Group B
    group_b_names = [
        "FC3", "FC1", "FCz", "FC2", "FC4",
        "C3", "C1", "Cz", "C2", "C4",
        "CP3", "CP1", "CPz", "CP2", "CP4"
    ]

    for subj in subjects:
        X, y, info = get_cached_subject_data(subject_id=subj, run_id="R11")
        csp_d, sel, sc, clf, adapt_bank = load_subject_model(subj)
        if adapt_bank is None:
            adapt_bank = [(5.5, 9.5), (7.0, 11.0), (9.5, 13.5), (14.0, 20.0), (16.0, 22.0), (20.0, 26.0)]

        for trial_idx in range(len(y)):
            win_15ch_gt = X[trial_idx:trial_idx+1] # shape (1, 15, 480)
            true_lbl = y[trial_idx]
            gt_y_true.append(true_lbl)

            # ------------------------------------------------------------------
            # BENCHMARK A: Ground Truth 15-Channel Step 60 Model Inference
            # ------------------------------------------------------------------
            t0 = time.perf_counter()
            filt_gt = extract_custom_filter_bank_windows(win_15ch_gt, adapt_bank)
            fbcsp_gt = extract_fbcsp_features_transform(filt_gt, csp_d)
            probs_gt_raw = clf.predict_proba(sc.transform(sel.transform(fbcsp_gt)))
            aligned_gt = align_probs(probs_gt_raw, clf.classes_)[0]

            if float(aligned_gt[0]) > 0.62:
                pred_gt = "REST"
            else:
                pred_gt = CLASSES[int(np.argmax(aligned_gt))]

            t1 = time.perf_counter()
            gt_y_pred.append(pred_gt)
            gt_latencies.append((t1 - t0) * 1000.0)

            # ------------------------------------------------------------------
            # BENCHMARK B: Simulated 8-Channel Subset -> 8-to-15 Reconstruction
            # ------------------------------------------------------------------
            # Simulate 8-channel physical capture by taking C3, Cz, C4 from EDF
            # and synthetic surrounding F3/F4/P3/Pz/P4 signals from nearby motor electrodes
            win_2d_15 = win_15ch_gt[0] # (15, 480)
            
            # Construct 8-channel array: F3, F4, C3, Cz, C4, P3, Pz, P4
            # Using FC3 as F3, FC4 as F4, C3, Cz, C4, CP3 as P3, CPz as Pz, CP4 as P4
            sim_8ch = np.zeros((8, 480))
            sim_8ch[0] = win_2d_15[group_b_names.index("FC3")] # F3 approx
            sim_8ch[1] = win_2d_15[group_b_names.index("FC4")] # F4 approx
            sim_8ch[2] = win_2d_15[group_b_names.index("C3")]  # C3 physical
            sim_8ch[3] = win_2d_15[group_b_names.index("Cz")]  # Cz physical
            sim_8ch[4] = win_2d_15[group_b_names.index("C4")]  # C4 physical
            sim_8ch[5] = win_2d_15[group_b_names.index("CP3")] # P3 approx
            sim_8ch[6] = win_2d_15[group_b_names.index("CPz")] # Pz approx
            sim_8ch[7] = win_2d_15[group_b_names.index("CP4")] # P4 approx

            t0_rec = time.perf_counter()
            trans_res = adapter.transform(sim_8ch, ["F3", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"])
            win_15ch_rec = trans_res["data_15ch"]

            filt_rec = extract_custom_filter_bank_windows(win_15ch_rec, adapt_bank)
            fbcsp_rec = extract_fbcsp_features_transform(filt_rec, csp_d)
            probs_rec_raw = clf.predict_proba(sc.transform(sel.transform(fbcsp_rec)))
            aligned_rec = align_probs(probs_rec_raw, clf.classes_)[0]

            if float(aligned_rec[0]) > 0.62:
                pred_rec = "REST"
            else:
                pred_rec = CLASSES[int(np.argmax(aligned_rec))]

            t1_rec = time.perf_counter()
            rec_y_pred.append(pred_rec)
            rec_latencies.append((t1_rec - t0_rec) * 1000.0)

    # --------------------------------------------------------------------------
    # METRICS CALCULATION
    # --------------------------------------------------------------------------
    acc_gt = accuracy_score(gt_y_true, gt_y_pred) * 100.0
    f1_gt = f1_score(gt_y_true, gt_y_pred, average="macro", zero_division=0) * 100.0
    rec_gt_per = recall_score(gt_y_true, gt_y_pred, average=None, labels=CLASSES) * 100.0
    cm_gt = confusion_matrix(gt_y_true, gt_y_pred, labels=CLASSES)

    acc_rec = accuracy_score(gt_y_true, rec_y_pred) * 100.0
    f1_rec = f1_score(gt_y_true, rec_y_pred, average="macro", zero_division=0) * 100.0
    rec_rec_per = recall_score(gt_y_true, rec_y_pred, average=None, labels=CLASSES) * 100.0
    cm_rec = confusion_matrix(gt_y_true, rec_y_pred, labels=CLASSES)

    avg_lat_gt = np.mean(gt_latencies)
    avg_lat_rec = np.mean(rec_latencies)

    print("\n================ BENCHMARK COMPARISON RESULTS ================")
    print(f"Total Holdout Trials Evaluated : {len(gt_y_true)} (S001-S010 R11)")
    print("----------------------------------------------------------------")
    print(f"1. Ground Truth (15 Physical Channels):")
    print(f"   Accuracy    : {acc_gt:.2f}%")
    print(f"   Macro F1    : {f1_gt:.2f}%")
    print(f"   Recall (REST): {rec_gt_per[0]:.2f}% | (LEFT): {rec_gt_per[1]:.2f}% | (RIGHT): {rec_gt_per[2]:.2f}%")
    print(f"   Avg Latency : {avg_lat_gt:.2f} ms")
    print("----------------------------------------------------------------")
    print(f"2. Reconstructed (8-ch OpenBCI -> 15-ch Spatial Interpolation):")
    print(f"   Accuracy    : {acc_rec:.2f}%")
    print(f"   Macro F1    : {f1_rec:.2f}%")
    print(f"   Recall (REST): {rec_rec_per[0]:.2f}% | (LEFT): {rec_rec_per[1]:.2f}% | (RIGHT): {rec_rec_per[2]:.2f}%")
    print(f"   Avg Latency : {avg_lat_rec:.2f} ms (Adapter Overhead: {avg_lat_rec - avg_lat_gt:.2f} ms)")
    print("----------------------------------------------------------------")
    print("Confusion Matrix - Ground Truth (REST / LEFT / RIGHT):")
    print(cm_gt)
    print("\nConfusion Matrix - Reconstructed (REST / LEFT / RIGHT):")
    print(cm_rec)
    print("================================================================\n")

    res_summary = {
        "trials_count": len(gt_y_true),
        "ground_truth": {
            "accuracy_pct": round(acc_gt, 2),
            "macro_f1_pct": round(f1_gt, 2),
            "recall": {"REST": round(rec_gt_per[0], 2), "LEFT_FIST": round(rec_gt_per[1], 2), "RIGHT_FIST": round(rec_gt_per[2], 2)},
            "latency_ms": round(avg_lat_gt, 2)
        },
        "reconstructed": {
            "accuracy_pct": round(acc_rec, 2),
            "macro_f1_pct": round(f1_rec, 2),
            "recall": {"REST": round(rec_rec_per[0], 2), "LEFT_FIST": round(rec_rec_per[1], 2), "RIGHT_FIST": round(rec_rec_per[2], 2)},
            "latency_ms": round(avg_lat_rec, 2)
        }
    }

    with open("scratch/interpolation_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(res_summary, f, indent=2)

    return res_summary


if __name__ == "__main__":
    run_interpolation_experiment()
