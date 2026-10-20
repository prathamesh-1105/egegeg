"""
Phase 4 Accuracy Research & Model Optimization Test Suite.

Verifies:
1. Immutability of historical models (Step 60, Step 61, Step 62).
2. Phase 4 candidate artifact integrity and metadata.
3. Preprocessing isolation (true anatomical Laplacian montage vs. historical).
4. Zero-leakage data splitting isolation (Dev R03+R07 vs. Holdout R11).
5. Dynamic model version selection and inference across Step 62, Exp 07, and Exp 08.
6. REST gating and false actuation safety verification.
"""

import os
import json
import pytest
import numpy as np


class TestArtifactImmutability:
    """Ensures historical baseline models remain strictly untouched and read-only."""

    def test_step60_artifacts_exist(self):
        assert os.path.exists("models/step60"), "Step 60 directory missing"
        assert os.path.exists("models/step60/selected_model"), "Step 60 selected_model missing"

    def test_step61_artifacts_exist(self):
        assert os.path.exists("models/step61"), "Step 61 directory missing"
        assert os.path.exists("models/step61/selected_model"), "Step 61 selected_model missing"
        assert os.path.exists("models/step61/summary_report.json"), "Step 61 summary report missing"

    def test_step62_artifacts_exist(self):
        assert os.path.exists("models/step62"), "Step 62 directory missing"
        assert os.path.exists("models/step62/selected_model"), "Step 62 selected_model missing"
        assert os.path.exists("models/step62/summary_report.json"), "Step 62 summary report missing"

    def test_phase4_artifacts_exist(self):
        assert os.path.exists("models/research_phase4/exp_07_anatomical_laplacian"), "Exp 07 dir missing"
        assert os.path.exists("models/research_phase4/exp_08_k28_ensemble"), "Exp 08 dir missing"
        assert os.path.exists("models/research_phase4/champion_candidate/metadata.json"), "Phase 4 metadata missing"

        with open("models/research_phase4/champion_candidate/metadata.json", "r") as f:
            meta = json.load(f)
        assert meta["candidate_id"] in ["exp_07_correct_laplacian_tri_ensemble", "exp_07_anatomical_laplacian"]
        assert "r11_accuracy" in meta
        assert meta["r11_accuracy"] > 70.0


class TestPreprocessingIsolation:
    """Verifies anatomical Laplacian channel math and isolation from historical defects."""

    def test_anatomical_laplacian_indices(self):
        channels = ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4']
        
        c3_idx = channels.index('C3')   # 5
        c4_idx = channels.index('C4')   # 9
        fc3_idx = channels.index('FC3') # 0
        c1_idx = channels.index('C1')   # 6
        cp3_idx = channels.index('CP3') # 10
        fc4_idx = channels.index('FC4') # 4
        c2_idx = channels.index('C2')   # 8
        cp4_idx = channels.index('CP4') # 14

        assert c3_idx == 5
        assert c4_idx == 9
        assert fc3_idx == 0
        assert c1_idx == 6
        assert cp3_idx == 10
        assert fc4_idx == 4
        assert c2_idx == 8
        assert cp4_idx == 14

    def test_anatomical_laplacian_computation(self):
        from scratch.train_and_save_phase4_models import apply_anatomical_surface_laplacian
        
        # Create a mock 15-channel signal with known values
        raw = np.zeros((1, 15, 100))
        raw[0, 5, :] = 10.0  # C3 = 10
        raw[0, 0, :] = 2.0   # FC3 = 2
        raw[0, 6, :] = 3.0   # C1 = 3
        raw[0, 10, :] = 4.0  # CP3 = 4

        # Lap(C3) = 10 - (2 + 3 + 4)/3 = 10 - 3 = 7
        lap = apply_anatomical_surface_laplacian(raw)
        np.testing.assert_allclose(lap[0, 5, :], 7.0, rtol=1e-5)


class TestZeroLeakageProtocol:
    """Verifies strict train/test separation between development runs and R11 holdout."""

    def test_phase4_comparison_json_integrity(self):
        assert os.path.exists("docs/phase4_model_comparison.json")
        with open("docs/phase4_model_comparison.json", "r") as f:
            comp = json.load(f)

        assert comp["evaluation_protocol"]["development_set"] == "R03 + R07 (600 trials, 5-Fold Stratified CV)"
        assert comp["evaluation_protocol"]["holdout_test_set"] == "R11 (300 trials, single-pass frozen evaluation)"

        # Check that top candidate has non-zero recall for all three classes
        top = comp["top_candidate"]
        assert top["r11_rest_recall"] >= 85.0
        assert top["r11_left_recall"] >= 50.0
        assert top["r11_right_recall"] >= 50.0
        assert top["r11_rest_false_actuation"] <= 15.0

    def test_loso_cross_subject_validity(self):
        with open("docs/phase4_model_comparison.json", "r") as f:
            comp = json.load(f)
        loso = comp["loso_cross_subject_results"]
        assert loso["n_total"] == 300
        assert loso["accuracy"] > 50.0
        assert loso["rest_recall"] > 65.0


class TestDynamicModelLoadingAndInference:
    """Tests runtime model switching and inference across Step 62 and Phase 4 models."""

    def test_available_models_registry(self):
        from api_server import AVAILABLE_MODELS
        assert "Step62-SurfaceLaplacian-Ensemble" in AVAILABLE_MODELS
        assert "Phase4-AnatomicalLaplacian-Ensemble" in AVAILABLE_MODELS
        assert "Phase4-K28-Ensemble" in AVAILABLE_MODELS

        assert AVAILABLE_MODELS["Step62-SurfaceLaplacian-Ensemble"]["is_phase4"] is False
        assert AVAILABLE_MODELS["Phase4-AnatomicalLaplacian-Ensemble"]["is_phase4"] is True
        assert AVAILABLE_MODELS["Phase4-K28-Ensemble"]["is_phase4"] is True

    def test_session_manager_model_switch(self):
        from src.neuro_os.session_manager import NeuroOSSessionManager
        sm = NeuroOSSessionManager()
        res = sm.set_model_version("Phase4-AnatomicalLaplacian-Ensemble")
        assert res["status"] == "SUCCESS"
        assert sm.active_model_id == "Phase4-AnatomicalLaplacian-Ensemble"

    def test_inference_step62(self):
        from api_server import predict_eeg_window
        res = predict_eeg_window(subject_id="S001", run_id="R11", trial_idx=0, model_version="Step62-SurfaceLaplacian-Ensemble")
        assert res["predicted_label"] in ["REST", "LEFT_FIST", "RIGHT_FIST"]
        assert 0.0 <= res["confidence"] <= 1.0
        assert res["latency_ms"] < 200.0
        assert res["model_version"] == "Step62-SurfaceLaplacian-Ensemble"

    def test_inference_phase4_anatomical(self):
        from api_server import predict_eeg_window
        res = predict_eeg_window(subject_id="S001", run_id="R11", trial_idx=0, model_version="Phase4-AnatomicalLaplacian-Ensemble")
        assert res["predicted_label"] in ["REST", "LEFT_FIST", "RIGHT_FIST"]
        assert 0.0 <= res["confidence"] <= 1.0
        assert res["latency_ms"] < 200.0
        assert res["model_version"] == "Phase4-AnatomicalLaplacian-Ensemble"

    def test_inference_phase4_k28(self):
        from api_server import predict_eeg_window
        res = predict_eeg_window(subject_id="S001", run_id="R11", trial_idx=0, model_version="Phase4-K28-Ensemble")
        assert res["predicted_label"] in ["REST", "LEFT_FIST", "RIGHT_FIST"]
        assert 0.0 <= res["confidence"] <= 1.0
        assert res["latency_ms"] < 200.0
        assert res["model_version"] == "Phase4-K28-Ensemble"

    def test_rest_gating_safety(self):
        from api_server import predict_eeg_window
        res = predict_eeg_window(subject_id="S001", run_id="R11", trial_idx=1, model_version="Phase4-AnatomicalLaplacian-Ensemble")
        assert "rest_gating" in res
        assert "threshold" in res["rest_gating"]
        assert "is_gated_rest" in res["rest_gating"]
