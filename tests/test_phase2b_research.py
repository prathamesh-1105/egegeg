"""
Unit Tests for Phase 2B Motor Imagery Accuracy Research
Validates:
1. Zero Data Leakage: R11 holdout split is strictly disjoint from R03 and R07.
2. Class Label Mapping & Alphabet Consistency across NeuroOS components.
3. Surface Laplacian and CAR spatial preprocessing invariants.
4. Regularized CSP numerical stability under rank deficiency.
5. Safety Shield rejection & REST command neutrality.
6. Step 60 and Step 61 Model Artifact Versioning and Preservation.
"""

import os
import sys
import unittest
import numpy as np
import joblib

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.step33_pipeline import GROUP_B_CHANNELS, CLASSES, C3_IDX, C4_IDX, CZ_IDX
from src.step61_pipeline import load_dataset_step61
from scratch.run_phase2b_accuracy_experiments import (
    apply_common_average_reference,
    apply_surface_laplacian_approx,
    safe_extract_fbcsp_features_train
)
from src.neuro_os.safety_shield import NeuroOSSafetyShield
from src.neuro_os.command_router import NeuroOSCommandRouter


class TestPhase2BResearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(REPO_ROOT, "data")
        cls.sub_data = load_dataset_step61(data_dir=cls.data_dir)
        cls.subjects = sorted(list(cls.sub_data.keys()))

    def test_01_zero_data_leakage_between_dev_and_test(self):
        """Verify that R11 holdout windows are strictly disjoint from R03/R07 development windows."""
        for s_id in self.subjects:
            X_dev = self.sub_data[s_id]["X_calib"]
            X_r11 = self.sub_data[s_id]["X_r11"]
            y_dev = self.sub_data[s_id]["y_calib"]
            y_r11 = self.sub_data[s_id]["y_r11"]

            # Development: R03 (30 trials) + R07 (30 trials) = 60 trials
            self.assertEqual(len(X_dev), 60, f"{s_id} development set should contain exactly 60 trials.")
            self.assertEqual(len(y_dev), 60)

            # Holdout: R11 = 30 trials
            self.assertEqual(len(X_r11), 30, f"{s_id} holdout test set should contain exactly 30 trials.")
            self.assertEqual(len(y_r11), 30)

            # Check that no window in R11 is numerically identical to any window in X_dev
            dev_first_sample = X_dev[:, 0, 0]
            r11_first_sample = X_r11[:, 0, 0]
            common_samples = np.intersect1d(dev_first_sample, r11_first_sample)
            self.assertEqual(len(common_samples), 0, f"Found overlapping window starting values in {s_id}!")

    def test_02_class_mapping_consistency(self):
        """Verify that CLASSES matches ['REST', 'LEFT_FIST', 'RIGHT_FIST'] across the pipeline."""
        expected_classes = ["REST", "LEFT_FIST", "RIGHT_FIST"]
        self.assertEqual(CLASSES, expected_classes)

        # Check that all loaded y_dev and y_r11 labels belong strictly to CLASSES
        for s_id in self.subjects:
            y_dev = self.sub_data[s_id]["y_calib"]
            y_r11 = self.sub_data[s_id]["y_r11"]
            for label in np.concatenate([y_dev, y_r11]):
                self.assertIn(label, CLASSES)

    def test_03_common_average_reference_invariants(self):
        """Verify that CAR subtracts the spatial mean across channels for each time point."""
        np.random.seed(42)
        dummy_eeg = np.random.randn(5, 16, 640)
        car_eeg = apply_common_average_reference(dummy_eeg)

        self.assertEqual(car_eeg.shape, dummy_eeg.shape)
        # Spatial mean across the 16 channels should be virtually zero (< 1e-6)
        spatial_means = np.mean(car_eeg, axis=1)
        np.testing.assert_allclose(spatial_means, 0.0, atol=1e-6)

    def test_04_surface_laplacian_operator_invariants(self):
        """Verify that surface Laplacian creates valid local derivations without NaNs or infs."""
        np.random.seed(42)
        dummy_eeg = np.random.randn(5, 16, 640)
        lap_eeg = apply_surface_laplacian_approx(dummy_eeg)

        self.assertEqual(lap_eeg.shape, dummy_eeg.shape)
        self.assertFalse(np.isnan(lap_eeg).any())
        self.assertFalse(np.isinf(lap_eeg).any())

        # For a uniform spatial field, Laplacian should be zero
        uniform_eeg = np.ones((2, 16, 100)) * 5.0
        lap_uniform = apply_surface_laplacian_approx(uniform_eeg)
        np.testing.assert_allclose(lap_uniform[:, C3_IDX, :], 0.0, atol=1e-6)

    def test_05_regularized_csp_stability(self):
        """Verify that regularized CSP handles rank-deficient CAR signals without SVD crashing."""
        s001_dev = self.sub_data["S001"]["X_calib"]
        s001_y = self.sub_data["S001"]["y_calib"]
        car_dev = apply_common_average_reference(s001_dev)

        # Dummy single-band dictionary
        filt_dict = {"mu": car_dev}
        X_fbcsp, csp_dict = safe_extract_fbcsp_features_train(filt_dict, s001_y, n_components=4)
        self.assertIn("mu", csp_dict)
        self.assertEqual(X_fbcsp.shape[0], len(s001_y))
        self.assertFalse(np.isnan(X_fbcsp).any())

    def test_06_safety_shield_rejection_and_rest_neutrality(self):
        """Verify safety shield rejects below confidence 0.60 or margin 0.10, and REST yields NEUTRAL."""
        shield = NeuroOSSafetyShield(confidence_thresh=0.60, margin_thresh=0.10)
        router = NeuroOSCommandRouter(command_mode="SCROLL")

        # 1. Low confidence -> rejected
        res_low = shield.evaluate_safety({"REST": 0.35, "LEFT_FIST": 0.35, "RIGHT_FIST": 0.30})
        self.assertFalse(res_low["accepted"])
        self.assertEqual(router.process_intent(res_low)["command"], "NEUTRAL")

        # 2. Low margin -> rejected
        res_margin = shield.evaluate_safety({"REST": 0.10, "LEFT_FIST": 0.46, "RIGHT_FIST": 0.44})
        self.assertFalse(res_margin["accepted"])

        # 3. High confidence REST -> accepted as REST, mapped to NEUTRAL
        shield.reset()
        res_rest = shield.evaluate_safety({"REST": 0.85, "LEFT_FIST": 0.10, "RIGHT_FIST": 0.05})
        self.assertTrue(res_rest["accepted"])
        self.assertEqual(res_rest["intent"], "REST")
        self.assertEqual(router.process_intent(res_rest)["command"], "NEUTRAL")

    def test_07_model_artifact_versioning_and_integrity(self):
        """Verify Step 60, Step 61, and Step 62 model directories are preserved with all required pipeline components."""
        for step in ["step60", "step61", "step62"]:
            step_dir = os.path.join(REPO_ROOT, "models", step, "selected_model")
            self.assertTrue(os.path.exists(step_dir), f"{step_dir} must exist!")

            for s_id in ["S001", "S002", "S003", "S004", "S005", "S006", "S007", "S008", "S009", "S010"]:
                sub_dir = os.path.join(step_dir, s_id)
                self.assertTrue(os.path.exists(sub_dir), f"Missing {s_id} directory in {step}")
                expected_files = [
                    "adaptive_bank.pkl", "csp.pkl", "cov_estimator.pkl",
                    "tangent_space.pkl", "feature_selector.pkl", "scaler.pkl", "model.pkl"
                ]
                for fname in expected_files:
                    fpath = os.path.join(sub_dir, fname)
                    self.assertTrue(os.path.exists(fpath), f"Missing {fname} in {sub_dir}")
                    self.assertGreater(os.path.getsize(fpath), 0, f"Empty artifact {fpath}")


if __name__ == "__main__":
    unittest.main()
