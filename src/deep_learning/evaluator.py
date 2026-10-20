import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Tuple, List, Optional

from src.step7_pipeline import EXPECTED_CLASSES
from src.deep_learning.eegnet import EEGNet


class EEGNetEvaluator:
    """
    Evaluation & Statistical Benchmark Suite for EEGNet vs FBCSP.
    """

    def __init__(self, model: EEGNet, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    def evaluate_dataloader(self, dataloader: DataLoader) -> Dict:
        """
        Calculates full classification metrics on DataLoader.
        """
        all_preds = []
        all_probs = []
        all_targets = []

        start_t = time.perf_counter()
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self.device)
                logits = self.model(X)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_probs.append(probs)
                all_preds.extend(preds)
                all_targets.extend(y.numpy())
        elapsed_sec = time.perf_counter() - start_t

        if not all_targets:
            return {}

        all_probs = np.vstack(all_probs)
        acc = float(accuracy_score(all_targets, all_preds) * 100.0)
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
        per_cls_prec, per_cls_rec, per_cls_f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0
        )

        cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])

        total_samples = len(all_targets)
        latency_ms_per_window = (elapsed_sec / max(1, total_samples)) * 1000.0

        return {
            "accuracy": round(acc, 2),
            "macro_precision": round(float(prec * 100.0), 2),
            "macro_recall": round(float(rec * 100.0), 2),
            "macro_f1": round(float(f1 * 100.0), 2),
            "rest_recall": round(float(per_cls_rec[0] * 100.0), 2),
            "left_fist_recall": round(float(per_cls_rec[1] * 100.0), 2),
            "right_fist_recall": round(float(per_cls_rec[2] * 100.0), 2),
            "confusion_matrix": cm.tolist(),
            "n_samples": total_samples,
            "inference_latency_ms_per_window": round(latency_ms_per_window, 3),
            "predictions": all_preds,
            "probabilities": all_probs,
            "targets": all_targets
        }

    @staticmethod
    def compute_statistical_tests(fbcsp_accs: List[float], eegnet_accs: List[float]) -> Dict:
        """
        Computes Paired Student t-test and Wilcoxon Signed-Rank Test between FBCSP and EEGNet per-subject accuracies.
        """
        fbcsp_arr = np.array(fbcsp_accs)
        eegnet_arr = np.array(eegnet_accs)
        diffs = eegnet_arr - fbcsp_arr

        # Paired t-test
        t_stat, p_val_t = stats.ttest_rel(eegnet_arr, fbcsp_arr)

        # Wilcoxon signed-rank test
        try:
            w_stat, p_val_w = stats.wilcoxon(eegnet_arr, fbcsp_arr)
        except Exception:
            w_stat, p_val_w = 0.0, 1.0

        n_improved = int(np.sum(diffs > 0))
        n_degraded = int(np.sum(diffs < 0))
        n_unchanged = int(np.sum(diffs == 0))

        return {
            "n_subjects": len(fbcsp_accs),
            "mean_fbcsp": round(float(np.mean(fbcsp_arr)), 2),
            "mean_eegnet": round(float(np.mean(eegnet_arr)), 2),
            "mean_difference": round(float(np.mean(diffs)), 2),
            "median_difference": round(float(np.median(diffs)), 2),
            "std_difference": round(float(np.std(diffs)), 2),
            "subjects_improved": n_improved,
            "subjects_degraded": n_degraded,
            "subjects_unchanged": n_unchanged,
            "paired_t_test": {
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_val_t), 4),
                "is_statistically_significant": bool(p_val_t < 0.05)
            },
            "wilcoxon_test": {
                "statistic": round(float(w_stat), 4),
                "p_value": round(float(p_val_w), 4),
                "is_statistically_significant": bool(p_val_w < 0.05)
            }
        }
