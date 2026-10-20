"""
Tests for NeuroOS Phase 6 — Multi-Dataset Preprocessing, Harmonization, & Model Artifacts
"""

import os
import json
import pytest
import numpy as np

from src.neuro_os.dataset_pipeline import (
    TARGET_MONTAGE,
    TARGET_SFREQ,
    TARGET_WINDOW_SEC,
    TARGET_SAMPLES,
    CLASSES,
    clean_channel_name,
    harmonize_montage,
    resample_epoch,
    apply_anatomical_surface_laplacian,
    extract_advanced_motor_features,
    load_bci_competition_iv_2a
)
from scratch.run_phase6_dataset_expansion import (
    extract_lateralized_spectral_features,
    CalibratedHierarchicalBCI,
    compute_detailed_metrics
)
from api_server import AVAILABLE_MODELS, load_subject_model


class TestPhase6DatasetPipeline:
    """Validates multi-dataset ingestion, channel harmonization, and resampling."""

    def test_target_montage_contract(self):
        """Verifies 15-channel motor montage specification."""
        assert len(TARGET_MONTAGE) == 15
        expected = [
            'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C3',  'C1',  'Cz',  'C2',  'C4',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4'
        ]
        assert TARGET_MONTAGE == expected
        assert TARGET_SFREQ == 160.0
        assert TARGET_SAMPLES == 480

    def test_clean_channel_name(self):
        """Verifies channel standardization removes prefixes, dots, spaces, casing."""
        assert clean_channel_name("EEG-C3") == "C3"
        assert clean_channel_name("EEG FCz") == "FCZ"
        assert clean_channel_name("C4..") == "C4"
        assert clean_channel_name(" cp3 ") == "CP3"

    def test_harmonize_montage_success(self):
        """Verifies correct sub-selection and reordering from larger montage."""
        raw_channels = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
            'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'Oz'
        ]
        dummy_data = np.random.randn(len(raw_channels), 480).astype(np.float32)

        harmonized, montage = harmonize_montage(dummy_data, raw_channels, TARGET_MONTAGE)
        assert harmonized.shape == (15, 480)
        assert montage == TARGET_MONTAGE
        # Check that C3 in harmonized matches C3 from raw_channels
        c3_raw_idx = raw_channels.index('C3')
        c3_harm_idx = TARGET_MONTAGE.index('C3')
        np.testing.assert_allclose(harmonized[c3_harm_idx], dummy_data[c3_raw_idx])

    def test_harmonize_montage_missing_channels_raises(self):
        """Verifies ValueError when required motor channel is missing."""
        incomplete_channels = ['FC3', 'FC1', 'FCz', 'C3', 'Cz', 'C4']
        dummy_data = np.random.randn(len(incomplete_channels), 480)
        with pytest.raises(ValueError, match="Cannot harmonize montage"):
            harmonize_montage(dummy_data, incomplete_channels, TARGET_MONTAGE)

    def test_resample_epoch_from_250_to_160(self):
        """Verifies polyphase resampling from 250 Hz (Graz 2a) to 160 Hz (NeuroOS standard)."""
        # 3.0 seconds at 250 Hz = 750 samples
        epoch_250 = np.random.randn(15, 750).astype(np.float32)
        resampled = resample_epoch(epoch_250, orig_sfreq=250.0, target_sfreq=160.0)

        assert resampled.shape == (15, 480)
        assert resampled.dtype == np.float32

    def test_resample_epoch_identical_sfreq(self):
        """Verifies identity pass-through when sampling rate matches."""
        epoch_160 = np.random.randn(15, 480).astype(np.float32)
        resampled = resample_epoch(epoch_160, orig_sfreq=160.0, target_sfreq=160.0)
        assert resampled.shape == (15, 480)
        np.testing.assert_allclose(resampled, epoch_160)


class TestPhase6FeatureExtraction:
    """Validates spatial filters and neurophysiological feature extractors."""

    def test_anatomical_surface_laplacian(self):
        """Verifies local orthogonal Laplacian derivation."""
        X = np.ones((2, 15, 480), dtype=np.float32)
        # In uniform signal, Laplacian diff to neighbors is zero
        X_lap = apply_anatomical_surface_laplacian(X)
        assert X_lap.shape == (2, 15, 480)
        # For C3 (idx 5), neighbors FC3(0), C1(6), CP3(10) are all 1.0, so 1.0 - 1.0 = 0.0
        np.testing.assert_allclose(X_lap[:, 5, :], 0.0, atol=1e-5)

    def test_lateralized_spectral_features(self):
        """Verifies extraction of contralateral ERD and asymmetry features."""
        X = np.random.randn(4, 15, 480).astype(np.float32)
        feats = extract_lateralized_spectral_features(X, sfreq=160.0)
        assert feats.shape[0] == 4
        assert feats.shape[1] == 25  # 15 channel variances + 10 asymmetry/differentials
        assert not np.isnan(feats).any()
        assert not np.isinf(feats).any()

    def test_calibrated_hierarchical_bci_contract(self):
        """Verifies 2-stage hierarchical classifier prediction and probability calibration."""
        clf = CalibratedHierarchicalBCI(target_rest_recall=0.90)
        X_toy = np.random.randn(30, 20).astype(np.float32)
        y_toy = np.array(["REST"] * 15 + ["LEFT_FIST"] * 8 + ["RIGHT_FIST"] * 7)

        clf.fit(X_toy, y_toy, oof_tune=True)
        probs = clf.predict_proba(X_toy)
        preds = clf.predict(X_toy)

        assert probs.shape == (30, 3)
        assert preds.shape == (30,)
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)
        assert set(preds).issubset(set(CLASSES))


class TestPhase6ModelArtifactsAndRegistry:
    """Verifies Phase 6 promoted artifacts and API integration."""

    def test_phase6_artifacts_exist(self):
        """Verifies models/research_phase6/ contains metadata and trained subject models."""
        p6_dir = "models/research_phase6"
        assert os.path.exists(p6_dir)
        assert os.path.exists(os.path.join(p6_dir, "metadata.json"))
        assert os.path.exists(os.path.join(p6_dir, "p6_pure_imagery_calibrated", "selected_model", "S001", "model.pkl"))
        assert os.path.exists(os.path.join(p6_dir, "p6_hierarchical_imagery", "selected_model", "S001", "model.pkl"))

    def test_phase6_metadata_integrity(self):
        """Verifies metadata trial counts, tasks, and metrics."""
        with open("models/research_phase6/metadata.json", "r") as f:
            meta = json.load(f)
        assert meta["phase"] == "Phase 6"
        assert meta["total_corpus_trials"] == 1800
        assert "p6_pure_imagery_calibrated" in meta["models"]
        assert "p6_hierarchical_imagery" in meta["models"]
        assert meta["models"]["p6_pure_imagery_calibrated"]["mean_holdout_accuracy"] == 57.0

    def test_api_model_registry_resolution(self):
        """Verifies API server resolves Phase 6 models and loads subject bundles."""
        assert "Phase6-PureImagery-Calibrated" in AVAILABLE_MODELS
        assert "Phase6-Hierarchical-Imagery" in AVAILABLE_MODELS

        bundle_p6 = load_subject_model("S001", model_version="Phase6-PureImagery-Calibrated")
        assert bundle_p6["clf"] is not None
        assert bundle_p6["scaler"] is not None if "scaler" in bundle_p6 else bundle_p6["sc"] is not None
        assert bundle_p6["model_cfg"]["is_phase6"] is True

    def test_historical_models_remain_frozen(self):
        """Verifies immutable preservation of Steps 60-62 and Phases 4-5."""
        assert os.path.exists("models/step60/selected_model/S001")
        assert os.path.exists("models/step61/selected_model/S001")
        assert os.path.exists("models/step62/selected_model/S001")
        assert os.path.exists("models/research_phase4/exp_07_anatomical_laplacian")
        assert os.path.exists("models/research_phase4/exp_08_k28_ensemble")
        assert os.path.exists("models/research_phase5/p5_advanced_motor_champion")
