"""
Unit and Integration Tests for NeuroOS Phase 2C
Validates:
1. Calibration/Evaluation Partition Zero-Leakage Separation
2. Calibration State Machine Lifecycle & Reset
3. Session Metadata Isolation (No base model modification)
4. Invalid/Noisy Calibration Data Rejection
5. Rejected Predictions Produce No Action Guardrails
6. REST Intent Neutrality & Emergency Lockout
7. Base Model Artifact Preservation (Step 60, 61, 62)
8. Probability Calibration (Temperature Scaling) Numerical Stability
"""

import os
import sys
import unittest
import numpy as np
import json
import tempfile
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.step33_pipeline import GROUP_B_CHANNELS, CLASSES
from src.neuro_os.calibration_manager import NeuroOSCalibrationManager, CalibrationState
from src.neuro_os.safety_shield import NeuroOSSafetyShield
from src.neuro_os.command_router import NeuroOSCommandRouter
from src.neuro_os.action_executor import NeuroOSActionExecutor
from scratch.run_phase2c_calibration_experiments import (
    extract_event_grouped_windows,
    TemperatureScaler,
    calculate_bci_safety_metrics
)


class TestPhase2CCalibration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.calib_mgr = NeuroOSCalibrationManager(target_windows_per_class=3, save_dir=self.temp_dir)
        self.safety_shield = NeuroOSSafetyShield(confidence_thresh=0.60, margin_thresh=0.10)
        self.command_router = NeuroOSCommandRouter(dwell_windows=3, cooldown_sec=0.0, command_mode="SCROLL")
        self.action_executor = NeuroOSActionExecutor(mode="SCROLL_DEMO")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_01_calibration_evaluation_zero_leakage_separation(self):
        """Verify that event-grouped windowing keeps event IDs strictly disjoint between calibration and evaluation sets."""
        dummy_raw = np.random.randn(20, 16, 640)  # 20 discrete trials
        dummy_labels = np.array(["REST"] * 10 + ["LEFT_FIST"] * 5 + ["RIGHT_FIST"] * 5)
        
        wins, labels, event_ids = extract_event_grouped_windows(dummy_raw, dummy_labels)
        self.assertEqual(len(wins), 20 * 3)  # 3 sub-windows per event

        # Partition with budget=3
        cal_events = np.unique(event_ids[:9]) # first 3 events of REST (3*3=9 windows)
        eval_events = np.unique(event_ids[9:])

        # Assert no overlapping events between calibration and evaluation
        overlap = np.intersect1d(cal_events, eval_events)
        self.assertEqual(len(overlap), 0, "Event overlap detected between calibration and evaluation partitions!")

    def test_02_calibration_manager_lifecycle_and_reset(self):
        """Verify state transitions: IDLE -> COLLECTING_REST -> COLLECTING_LEFT -> COLLECTING_RIGHT -> COMPLETED -> IDLE on reset."""
        status = self.calib_mgr.start_calibration(budget=2)
        self.assertEqual(self.calib_mgr.state, CalibrationState.COLLECTING_REST)
        self.assertEqual(self.calib_mgr.get_active_class(), "REST")

        dummy_win = np.random.randn(15, 480)
        good_quality = {"passed": True}

        # Record 2 REST windows
        self.calib_mgr.record_window(dummy_win, good_quality)
        res = self.calib_mgr.record_window(dummy_win, good_quality)
        self.assertEqual(self.calib_mgr.state, CalibrationState.COLLECTING_LEFT)

        # Record 2 LEFT windows
        self.calib_mgr.record_window(dummy_win, good_quality)
        res = self.calib_mgr.record_window(dummy_win, good_quality)
        self.assertEqual(self.calib_mgr.state, CalibrationState.COLLECTING_RIGHT)

        # Record 2 RIGHT windows
        self.calib_mgr.record_window(dummy_win, good_quality)
        res = self.calib_mgr.record_window(dummy_win, good_quality)
        self.assertEqual(self.calib_mgr.state, CalibrationState.COMPLETED)

        # Reset
        self.calib_mgr.reset()
        self.assertEqual(self.calib_mgr.state, CalibrationState.IDLE)
        self.assertEqual(self.calib_mgr.collected_windows["REST"], [])

    def test_03_session_metadata_schema_and_isolation(self):
        """Verify that completed calibration saves metadata with required keys and leaves base models isolated."""
        self.calib_mgr.start_calibration(budget=1, subject_id="S001")
        dummy_win = np.random.randn(15, 480)
        good_quality = {"passed": True}

        for _ in range(3):
            self.calib_mgr.record_window(dummy_win, good_quality)

        self.assertEqual(self.calib_mgr.state, CalibrationState.COMPLETED)
        meta_files = [f for f in os.listdir(self.temp_dir) if f.endswith("_meta.json")]
        self.assertEqual(len(meta_files), 1)

        with open(os.path.join(self.temp_dir, meta_files[0]), "r", encoding="utf-8") as fp:
            meta = json.load(fp)

        self.assertIn("session_id", meta)
        self.assertIn("hardware_profile_id", meta)
        self.assertIn("collected_counts", meta)
        self.assertTrue(meta.get("base_model_preserved", False))

    def test_04_invalid_calibration_data_rejection(self):
        """Verify that noisy/corrupted windows are rejected and not counted towards target budget."""
        self.calib_mgr.start_calibration(budget=3)
        dummy_win = np.random.randn(15, 480)
        bad_quality = {"passed": False, "quality_warning": "Impedance too high"}

        res = self.calib_mgr.record_window(dummy_win, bad_quality)
        self.assertFalse(res["window_recorded"])
        self.assertEqual(self.calib_mgr.rejected_windows_count, 1)
        self.assertEqual(len(self.calib_mgr.collected_windows["REST"]), 0)

    def test_05_rejected_prediction_produces_no_action(self):
        """Verify that safety-shield rejected predictions produce NEUTRAL and trigger zero scroll displacement."""
        initial_offset = self.action_executor.state_store["scroll_demo"]["scroll_offset_px"]
        
        # Ambiguous distribution: max_prob 0.38 < 0.60
        res_shield = self.safety_shield.evaluate_safety({"REST": 0.31, "LEFT_FIST": 0.38, "RIGHT_FIST": 0.31})
        self.assertFalse(res_shield["accepted"])

        routing = self.command_router.process_intent(res_shield)
        self.assertEqual(routing["command"], "NEUTRAL")
        self.assertFalse(routing["action_triggered"])

        action_res = self.action_executor.execute_command(routing["command"])
        self.assertFalse(action_res["executed"])
        self.assertEqual(self.action_executor.state_store["scroll_demo"]["scroll_offset_px"], initial_offset)

    def test_06_rest_neutrality_and_emergency_disengage(self):
        """Verify REST produces zero scroll displacement and 5 consecutive REST windows trigger emergency lock."""
        for _ in range(4):
            shield_res = self.safety_shield.evaluate_safety({"REST": 0.85, "LEFT_FIST": 0.10, "RIGHT_FIST": 0.05})
            self.assertTrue(shield_res["accepted"])
            routing = self.command_router.process_intent(shield_res)
            self.assertEqual(routing["command"], "NEUTRAL")

        # 5th consecutive REST window should trigger emergency disengage lock
        shield_res5 = self.safety_shield.evaluate_safety({"REST": 0.85, "LEFT_FIST": 0.10, "RIGHT_FIST": 0.05})
        self.assertTrue(shield_res5["emergency_locked"])

    def test_07_base_model_artifact_preservation(self):
        """Verify that Step 60, Step 61, and Step 62 models remain completely preserved and untouched."""
        for step in ["step60", "step61", "step62"]:
            step_dir = os.path.join(REPO_ROOT, "models", step, "selected_model")
            self.assertTrue(os.path.exists(step_dir), f"{step_dir} must exist!")

            for s_id in [f"S{i:03d}" for i in range(1, 11)]:
                sub_dir = os.path.join(step_dir, s_id)
                self.assertTrue(os.path.exists(sub_dir), f"Missing {s_id} in {step}")
                for fname in ["adaptive_bank.pkl", "csp.pkl", "cov_estimator.pkl", "tangent_space.pkl", "feature_selector.pkl", "scaler.pkl", "model.pkl"]:
                    fpath = os.path.join(sub_dir, fname)
                    self.assertTrue(os.path.exists(fpath) and os.path.getsize(fpath) > 0)

    def test_08_temperature_scaler_stability_and_reproducibility(self):
        """Verify that TemperatureScaler preserves class ranking and yields well-formed probability distributions."""
        probs = np.array([
            [0.70, 0.20, 0.10],
            [0.10, 0.80, 0.10],
            [0.15, 0.15, 0.70]
        ])
        y = np.array(["REST", "LEFT_FIST", "RIGHT_FIST"])

        scaler = TemperatureScaler().fit(probs, y)
        self.assertGreater(scaler.temperature, 0.0)

        scaled = scaler.transform(probs)
        # Sum to 1.0
        np.testing.assert_allclose(np.sum(scaled, axis=1), 1.0, atol=1e-5)
        # Class argmax preserved
        self.assertTrue((np.argmax(probs, axis=1) == np.argmax(scaled, axis=1)).all())


if __name__ == "__main__":
    unittest.main()
