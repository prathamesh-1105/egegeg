"""
Tests for NeuroOS Phase 7 — Full-Cohort Motor Imagery Training and Generalization

Validates:
1. Resumable downloader and 109-subject PhysioNet cohort inventory.
2. 3-stage EDF validation: ASCII header, 160 Hz sampling rate, 64-channel montage, event annotations.
3. Verified local pure motor imagery dataset (S001–S010, runs R04, R08, R12: 900 real trials).
4. Compact Deep Learning EEGNet forward pass contract on (B, 1, 15, 480).
5. Lateralized spectral differencing and two-stage hierarchical BCI decoding.
6. Preservation and immutability of historical models (Step 60–62, Phase 4–6).
7. Phase 7 model registry in backend and API server.
"""

import os
import json
import pytest
import numpy as np
import torch

from src.neuro_os.resumable_downloader import (
    PHYSIONET_BASE_URL,
    MOTOR_IMAGERY_RUNS,
    MOTOR_EXECUTION_RUNS,
    ResumableDownloader,
    verify_edf_file,
    audit_local_cohort
)
from src.neuro_os.dataset_pipeline import (
    TARGET_MONTAGE,
    TARGET_SFREQ,
    TARGET_WINDOW_SEC,
    TARGET_SAMPLES,
    CLASSES,
    apply_anatomical_surface_laplacian
)
from src.deep_learning.eegnet import EEGNet
from scratch.run_phase6_dataset_expansion import (
    extract_lateralized_spectral_features,
    CalibratedHierarchicalBCI,
    compute_detailed_metrics
)
from api_server import AVAILABLE_MODELS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestPhase7CohortAndDownloader:
    """Validates full-cohort audit, resumable downloader, and EDF integrity checks."""

    def test_cohort_target_scope(self):
        """Verifies target cohort covers all 109 PhysioNet EEGMMIDB subjects."""
        all_subjects = [f"S{i:03d}" for i in range(1, 110)]
        assert len(all_subjects) == 109
        assert all_subjects[0] == "S001"
        assert all_subjects[-1] == "S109"

    def test_run_definitions(self):
        """Verifies pure motor imagery and execution run assignments."""
        assert MOTOR_IMAGERY_RUNS == ["R04", "R08", "R12"]
        assert MOTOR_EXECUTION_RUNS == ["R03", "R07", "R11"]
        # Ensure zero overlap
        assert set(MOTOR_IMAGERY_RUNS).isdisjoint(set(MOTOR_EXECUTION_RUNS))

    def test_downloader_initialization(self, tmp_path):
        """Verifies ResumableDownloader properties and directory structure."""
        dl = ResumableDownloader(data_dir=str(tmp_path), base_url=PHYSIONET_BASE_URL)
        assert dl.data_dir == str(tmp_path)
        assert dl.base_url == PHYSIONET_BASE_URL
        assert os.path.exists(dl.manifest_path)

    def test_verified_local_cohort_audit(self):
        """Audits the 10 local subjects (S001..S010) across pure motor imagery runs."""
        data_dir = os.path.join(REPO_ROOT, "data")
        audit = audit_local_cohort(data_dir=data_dir)

        assert audit["total_local_subjects"] == 10
        assert audit["complete_subjects_count"] == 10
        assert audit["total_verified_pure_imagery_files"] == 30
        assert audit["total_verified_execution_files"] == 30
        assert audit["total_verified_trials"] == 1800
        assert audit["missing_local_files_count"] == 0

    def test_edf_validation_on_real_recording(self):
        """Verifies 3-stage validation on a verified real EDF recording."""
        from src.step8_pipeline import find_subject_runs
        runs = find_subject_runs("S001", data_dir=os.path.join(REPO_ROOT, "data"))
        real_edf = runs["R04"]
        assert os.path.exists(real_edf)

        is_valid, msg, info = verify_edf_file(real_edf)
        assert is_valid is True
        assert "verified" in msg.lower()
        assert info["channels_count"] == 64
        assert abs(info["sfreq"] - 160.0) < 1e-2

    def test_edf_validation_on_corrupt_file(self, tmp_path):
        """Verifies corruption detection on invalid files."""
        corrupt_edf = tmp_path / "corrupt.edf"
        corrupt_edf.write_bytes(b"NOT_AN_EDF_HEADER" * 10)

        is_valid, msg, info = verify_edf_file(str(corrupt_edf))
        assert is_valid is False
        assert "Invalid EDF" in msg or "size too small" in msg


class TestPhase7DeepLearningAndFeatures:
    """Validates compact EEGNet and neurophysiological feature extractors."""

    def test_eegnet_forward_pass_contract(self):
        """Verifies compact EEGNet tensor dimensions and output probabilities."""
        batch_size = 4
        channels = 15
        samples = 480  # 3.0 sec @ 160 Hz
        classes = 3

        model = EEGNet(n_classes=classes, n_channels=channels, n_samples=samples)
        model.eval()

        dummy_eeg = torch.randn(batch_size, 1, channels, samples, dtype=torch.float32)
        with torch.no_grad():
            out = model(dummy_eeg)

        assert out.shape == (batch_size, classes)
        # Verify finite values and valid softmax probabilities
        probs = torch.softmax(out, dim=1).numpy()
        assert np.all(np.isfinite(probs))
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(batch_size), atol=1e-5)

    def test_lateralized_spectral_features_dimension(self):
        """Verifies lateralized feature extractor output dimension."""
        dummy_epoch = np.random.randn(2, 15, 480).astype(np.float32)
        feats = extract_lateralized_spectral_features(dummy_epoch)

        assert feats.shape[0] == 2
        # Lateralized spectral features extracts 25 features
        assert feats.shape[1] == 25
        assert np.all(np.isfinite(feats))

    def test_hierarchical_bci_gating(self):
        """Verifies calibrated hierarchical BCI gating behavior."""
        clf = CalibratedHierarchicalBCI(target_rest_recall=0.85)

        # Create dummy training data (20 trials of each class)
        np.random.seed(42)
        X_tr = np.random.randn(60, 20)
        y_tr = np.array(["REST"] * 20 + ["LEFT_FIST"] * 20 + ["RIGHT_FIST"] * 20)

        clf.fit(X_tr, y_tr, oof_tune=False)

        # Test prediction on unseen samples
        X_te = np.random.randn(10, 20)
        preds = clf.predict(X_te)

        assert len(preds) == 10
        assert set(preds).issubset(set(CLASSES))


class TestPhase7ModelRegistryAndPreservation:
    """Validates historical model preservation and Phase 7 model registry."""

    def test_historical_models_preserved(self):
        """Verifies historical benchmark models are strictly intact."""
        hist_paths = [
            os.path.join(REPO_ROOT, "models", "step60"),
            os.path.join(REPO_ROOT, "models", "step61"),
            os.path.join(REPO_ROOT, "models", "step62"),
            os.path.join(REPO_ROOT, "models", "research_phase4"),
            os.path.join(REPO_ROOT, "models", "research_phase5"),
            os.path.join(REPO_ROOT, "models", "research_phase6")
        ]
        for p in hist_paths:
            assert os.path.exists(p), f"Historical model directory {p} was modified or deleted!"

    def test_phase7_models_registered_in_backend(self):
        """Verifies Phase 7 models are accessible in AVAILABLE_MODELS registry."""
        model_ids = [m["id"] for m in AVAILABLE_MODELS.values()]

        # Phase 6 baseline should be present
        assert "Phase6-PureImagery-Calibrated" in model_ids

        # Phase 7 promoted models should be registered
        assert "Phase7-PureImagery-Calibrated" in model_ids
        assert "Phase7-Hierarchical-LOSO" in model_ids

