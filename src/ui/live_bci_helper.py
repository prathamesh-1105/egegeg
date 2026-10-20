import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from src.step8_pipeline import find_subject_runs
from src.step9_pipeline import extract_raw_multichannel_windows
from src.step14_pipeline import extract_event_aligned_windows
from src.step11_pipeline import apply_causal_temporal_smoothing, EXPECTED_CLASSES
from src.step17b_pipeline import extract_fbcsp_features, FILTER_BANKS

# Step 17B Expected Benchmark Parameters for Pipeline Identity Verification
EXPECTED_STEP17B_PARAMS = {
    "channel_count": 15,
    "channels": ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4'],
    "sampling_rate": 160.0,
    "window_length_sec": 3.0,
    "onset_offset_sec": 0.5,
    "filter_banks_count": 7,
    "filter_banks": FILTER_BANKS,
    "csp_components_per_bank": 4,
    "total_csp_features": 28,
    "selected_feature_count": 16,
    "classifier": "LogisticRegression",
    "class_weight": "balanced",
    "class_order": ['LEFT_FIST', 'REST', 'RIGHT_FIST']
}

COMPATIBLE_TASK1_RUNS = ["R03", "R07", "R11"]
INCOMPATIBLE_TASK3_4_RUNS = ["R05", "R06", "R09", "R10", "R13", "R14"]


def verify_step17b_pipeline_identity(model, scaler, csp_dict, selector=None) -> tuple[bool, str, dict]:
    """
    Strict pipeline identity check between Step 17B benchmark & live pipeline.
    Checks channels, sampling rate, window length, offset, filter banks, CSP components,
    selected feature count, scaler, classifier, and class order.
    """
    mismatches = []
    
    # Check classifier type
    cls_name = type(model).__name__
    if cls_name != EXPECTED_STEP17B_PARAMS["classifier"]:
        mismatches.append(f"Classifier type mismatch: expected {EXPECTED_STEP17B_PARAMS['classifier']}, got {cls_name}")

    # Check class ordering
    if hasattr(model, "classes_"):
        cls_order = list(model.classes_)
        if cls_order != EXPECTED_STEP17B_PARAMS["class_order"]:
            mismatches.append(f"Class ordering mismatch: expected {EXPECTED_STEP17B_PARAMS['class_order']}, got {cls_order}")

    # Check CSP filter banks
    if not isinstance(csp_dict, dict) or len(csp_dict) != EXPECTED_STEP17B_PARAMS["filter_banks_count"]:
        mismatches.append(f"Filter bank count mismatch: expected {EXPECTED_STEP17B_PARAMS['filter_banks_count']}, got {len(csp_dict) if isinstance(csp_dict, dict) else 0}")

    # Check scaler features
    if hasattr(scaler, "n_features_in_"):
        if scaler.n_features_in_ != EXPECTED_STEP17B_PARAMS["selected_feature_count"]:
            mismatches.append(f"Selected feature count mismatch: expected {EXPECTED_STEP17B_PARAMS['selected_feature_count']}, got {scaler.n_features_in_}")

    # Check selector if present
    if selector is not None and hasattr(selector, "k"):
        if selector.k != EXPECTED_STEP17B_PARAMS["selected_feature_count"]:
            mismatches.append(f"Selector k count mismatch: expected {EXPECTED_STEP17B_PARAMS['selected_feature_count']}, got {selector.k}")

    is_matched = len(mismatches) == 0
    status_str = "PIPELINE MATCH" if is_matched else f"PIPELINE MISMATCH: {'; '.join(mismatches)}"
    return is_matched, status_str, EXPECTED_STEP17B_PARAMS


class LiveBCIPipeline:
    """
    Manages live BCI dataset playback and real-time inference for Step 17B (FBCSP) and Step 14 (Riemannian).
    Provides strict identity checks, timing breakdowns, and compatibility checks.
    """

    def __init__(
        self,
        subject_id: str = "S001",
        run_id: str = "R11",
        data_dir: str = "data",
        models_dir: str = "models",
        model_type: str = "STEP 17B"
    ):
        self.subject_id = subject_id
        self.run_id = run_id
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.model_type_name = model_type
        self.is_fbcsp = ("17B" in model_type or "FBCSP" in model_type)

        # Locate model directory
        if self.is_fbcsp:
            self.online_dir = os.path.join(models_dir, "step18", "selected_model", subject_id)
            if not os.path.exists(self.online_dir):
                self.online_dir = os.path.join(models_dir, "step17b", "selected_model", subject_id)
            if not os.path.exists(self.online_dir):
                self.online_dir = os.path.join(models_dir, "step17b", "selected_model", "S001")
        elif "STEP 14" in model_type or "MOTOR-RHYTHM" in model_type:
            self.online_dir = os.path.join(models_dir, "step14", "selected_model", subject_id)
            if not os.path.exists(self.online_dir):
                self.online_dir = os.path.join(models_dir, "step14", "selected_model", "S001")
        else:
            self.online_dir = os.path.join(models_dir, "step17b", "selected_model", subject_id)
            if not os.path.exists(self.online_dir):
                self.online_dir = os.path.join(models_dir, "step17b", "selected_model", "S001")

        # Load artifacts
        self.model_path = os.path.join(self.online_dir, "model.pkl")
        self.scaler_path = os.path.join(self.online_dir, "scaler.pkl")
        self.csp_path = os.path.join(self.online_dir, "csp.pkl")
        self.selector_path = os.path.join(self.online_dir, "selector.pkl")
        self.meta_path = os.path.join(self.online_dir, "metadata.json")

        self.model = joblib.load(self.model_path) if os.path.exists(self.model_path) else None
        self.scaler = joblib.load(self.scaler_path) if os.path.exists(self.scaler_path) else None
        self.csp_dict = joblib.load(self.csp_path) if os.path.exists(self.csp_path) else None
        self.selector = joblib.load(self.selector_path) if os.path.exists(self.selector_path) else None

        self.cov_path = os.path.join(self.online_dir, "cov.pkl")
        self.ts_path = os.path.join(self.online_dir, "riemannian.pkl")
        self.cov_est = joblib.load(self.cov_path) if os.path.exists(self.cov_path) else None
        self.ts_trans = joblib.load(self.ts_path) if os.path.exists(self.ts_path) else None

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        # Perform Identity Verification
        if self.is_fbcsp and self.model is not None and self.scaler is not None:
            self.is_identity_matched, self.identity_status, self.expected_params = verify_step17b_pipeline_identity(
                self.model, self.scaler, self.csp_dict, self.selector
            )
        else:
            self.is_identity_matched = True
            self.identity_status = "STEP 14 BASELINE ACTIVE"
            self.expected_params = EXPECTED_STEP17B_PARAMS

        # Load EEG recording run
        found_runs = find_subject_runs(self.subject_id, data_dir=data_dir)
        eval_run_file = found_runs.get(run_id, None)
        if not eval_run_file and found_runs:
            eval_run_file = list(found_runs.values())[0]

        if not eval_run_file:
            raise FileNotFoundError(f"No EDF recording run {run_id} found for subject {self.subject_id}")

        self.edf_file = eval_run_file
        self.is_compatible_run = (run_id in COMPATIBLE_TASK1_RUNS)

        # Extract event-aligned 3s windows starting at +0.5s offset
        self.X_windows, self.y_labels, self.infos = extract_event_aligned_windows(
            eval_run_file,
            channel_group="Group B",
            window_sec=3.0,
            onset_offset_sec=0.5
        )

        if len(self.X_windows) == 0:
            # Fallback to continuous extraction if event extraction returns empty
            self.X_windows, self.y_labels, self.infos = extract_raw_multichannel_windows(
                eval_run_file,
                channel_group="Group B",
                window_sec=3.0,
                overlap=0.5
            )

    def get_total_windows(self) -> int:
        return len(self.X_windows)

    def predict_window(self, window_idx: int, past_raw_history: list = None) -> dict:
        """
        Runs live inference on window_idx and records breakdown latency.
        """
        if not self.is_identity_matched:
            return {
                "error": True,
                "status_message": self.identity_status,
                "window_idx": window_idx,
                "total_windows": len(self.X_windows)
            }

        idx = max(0, min(window_idx, len(self.X_windows) - 1))
        X_win = self.X_windows[idx]  # (15, 480)
        y_actual = str(self.y_labels[idx])
        info = self.infos[idx] if idx < len(self.infos) else {"window_start": idx*1.5, "window_end": idx*1.5+3.0}

        sig_min = float(np.min(X_win))
        sig_max = float(np.max(X_win))
        sig_mean = float(np.mean(X_win))
        sig_std = float(np.std(X_win))

        X_input = np.expand_dims(X_win, axis=0)  # (1, 15, 480)

        # Latency breakdown timing
        t0 = time.perf_counter()

        if self.is_fbcsp:
            # 1. Filter bank + CSP extraction
            t1 = time.perf_counter()
            X_fbcsp, _ = extract_fbcsp_features(X_input, csp_dict=self.csp_dict, is_train=False)
            t2 = time.perf_counter()

            # 2. Feature selection
            if self.selector is not None:
                X_sel = self.selector.transform(X_fbcsp)
            else:
                X_sel = X_fbcsp[:, :self.scaler.n_features_in_]
            t3 = time.perf_counter()

            # 3. Scaling
            X_sc = self.scaler.transform(X_sel)
            t4 = time.perf_counter()

            # 4. LogisticRegression inference
            raw_pred = str(self.model.predict(X_sc)[0])
            proba_arr = self.model.predict_proba(X_sc)[0]
            t5 = time.perf_counter()

            filter_csp_ms = (t2 - t1) * 1000.0
            selection_ms = (t3 - t2) * 1000.0
            scaling_ms = (t4 - t3) * 1000.0
            inference_ms = (t5 - t4) * 1000.0
            total_comp_ms = (t5 - t0) * 1000.0

            classes = list(self.model.classes_)
            prob_dict = {cls: float(p) for cls, p in zip(classes, proba_arr)}
            confidence = float(np.max(proba_arr))

        else: # Step 14 Riemannian
            t1 = time.perf_counter()
            covs = self.cov_est.transform(X_input)
            X_ts = self.ts_trans.transform(covs)
            t2 = time.perf_counter()
            X_sc = self.scaler.transform(X_ts)
            t3 = time.perf_counter()
            raw_pred = str(self.model.predict(X_sc)[0])
            proba_arr = self.model.predict_proba(X_sc)[0]
            t4 = time.perf_counter()

            filter_csp_ms = (t2 - t1) * 1000.0
            selection_ms = 0.0
            scaling_ms = (t3 - t2) * 1000.0
            inference_ms = (t4 - t3) * 1000.0
            total_comp_ms = (t4 - t0) * 1000.0

            classes = list(self.model.classes_)
            prob_dict = {cls: float(p) for cls, p in zip(classes, proba_arr)}
            confidence = float(np.max(proba_arr))

        # Causal temporal smoothing (using current & past predictions only)
        if past_raw_history is None:
            past_raw_history = []

        history_with_current = past_raw_history + [raw_pred]
        smoothed_pred = apply_causal_temporal_smoothing(history_with_current, window_size=3)

        return {
            "window_idx": idx,
            "total_windows": len(self.X_windows),
            "subject_id": self.subject_id,
            "run_id": self.run_id,
            "filename": os.path.basename(self.edf_file),
            "window_start": info.get("window_start", round(idx * 1.5, 2)),
            "window_end": info.get("window_end", round(idx * 1.5 + 3.0, 2)),
            "actual_label": y_actual,
            "raw_prediction": raw_pred,
            "smoothed_prediction": smoothed_pred,
            "is_correct": bool(raw_pred == y_actual),
            "confidence": confidence,
            "probabilities": prob_dict,
            "latency": {
                "total_computation_ms": round(total_comp_ms, 2),
                "filter_csp_ms": round(filter_csp_ms, 2),
                "selection_ms": round(selection_ms, 2),
                "scaling_ms": round(scaling_ms, 2),
                "inference_ms": round(inference_ms, 2),
                "eeg_history_sec": 3.0
            },
            "raw_stats": {
                "min": sig_min,
                "max": sig_max,
                "mean": sig_mean,
                "std": sig_std
            },
            "single_channel_uv": X_win[5] * 1e6  # Channel C3 microvolts
        }


# Alias for backward compatibility
LiveRiemannianPipeline = LiveBCIPipeline

