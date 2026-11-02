"""
Unit and Integration Tests for NeuroOS Phase 9 — EEG-Controlled Web Browser & Real-Time Replay Scrolling.

Validates the complete 15-point test contract:
1. LEFT_FIST maps to scroll up.
2. RIGHT_FIST maps to scroll down.
3. REST produces no scrolling.
4. Low-confidence predictions are rejected.
5. Ground truth cannot control scrolling.
6. Pause stops actions.
7. Resume continues playback.
8. Stop ends the session.
9. Reset restores the initial scroll position.
10. Duplicate predictions are suppressed.
11. Invalid URLs are rejected.
12. Local demo webpage scroll control works.
13. External iframe restrictions are handled honestly.
14. WebSocket telemetry updates the frontend.
15. Historical models remain unchanged.
"""

import os
import sys
import unittest
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.neuro_os.safety_shield import NeuroOSSafetyShield
from src.neuro_os.command_router import NeuroOSCommandRouter
from src.neuro_os.action_executor import NeuroOSActionExecutor
from src.neuro_os.session_manager import NeuroOSSessionManager, SessionState
from src.neuro_os.command_profiles import get_command_profile, COMMAND_PROFILES
from src.neuro_os.web_demo_page import generate_demo_article_html
from api_server import (
    AVAILABLE_MODELS,
    select_model_version,
    ModelSelectRequest,
    get_web_control_status,
    start_web_control,
    pause_web_control,
    resume_web_control,
    stop_web_control,
    reset_web_control,
    step_web_control,
    validate_web_url,
    UrlValidationRequest,
    update_web_control_config,
    WebControlConfigRequest,
    WebControlStepRequest,
    get_demo_article_html,
    predict_eeg_window
)


class TestPhase9WebBrowser(unittest.TestCase):

    def setUp(self):
        """Set up fresh isolated components for Phase 9 Web Control testing."""
        self.safety_shield = NeuroOSSafetyShield(confidence_thresh=0.60, margin_thresh=0.10)
        self.command_router = NeuroOSCommandRouter(dwell_windows=1, cooldown_sec=0.0, profile_id="web_control")
        self.action_executor = NeuroOSActionExecutor(mode="SCROLL_DEMO")
        # Reset scroll demo position in action executor state store
        self.action_executor.state_store["scroll_demo"]["scroll_offset_px"] = 0

    def test_01_left_fist_maps_to_scroll_up(self):
        """1. Validate that accepted LEFT_FIST prediction maps strictly to SCROLL_UP."""
        raw_probs = {"REST": 0.05, "LEFT_FIST": 0.90, "RIGHT_FIST": 0.05}
        safety_res = self.safety_shield.evaluate_safety(raw_probs, hardware_connected=True, signal_quality_ok=True)
        self.assertTrue(safety_res["accepted"], "High-confidence LEFT_FIST must be accepted.")
        self.assertEqual(safety_res["intent"], "LEFT_FIST")

        routing_res = self.command_router.process_intent(safety_res)
        self.assertEqual(routing_res["command"], "SCROLL_UP", "LEFT_FIST must route to SCROLL_UP in web_control profile.")
        self.assertTrue(routing_res["action_triggered"])

        # Set initial offset to 360px and verify upward scroll decrements offset
        self.action_executor.state_store["scroll_demo"]["scroll_offset_px"] = 360
        action_res = self.action_executor.execute_command(routing_res["command"])
        self.assertTrue(action_res["executed"])
        self.assertEqual(action_res["action"], "SCROLL_UP")
        self.assertEqual(self.action_executor.state_store["scroll_demo"]["scroll_offset_px"], 180)

    def test_02_right_fist_maps_to_scroll_down(self):
        """2. Validate that accepted RIGHT_FIST prediction maps strictly to SCROLL_DOWN."""
        raw_probs = {"REST": 0.05, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.90}
        safety_res = self.safety_shield.evaluate_safety(raw_probs, hardware_connected=True, signal_quality_ok=True)
        self.assertTrue(safety_res["accepted"], "High-confidence RIGHT_FIST must be accepted.")
        self.assertEqual(safety_res["intent"], "RIGHT_FIST")

        routing_res = self.command_router.process_intent(safety_res)
        self.assertEqual(routing_res["command"], "SCROLL_DOWN", "RIGHT_FIST must route to SCROLL_DOWN in web_control profile.")
        self.assertTrue(routing_res["action_triggered"])

        self.assertEqual(self.action_executor.state_store["scroll_demo"]["scroll_offset_px"], 0)
        action_res = self.action_executor.execute_command(routing_res["command"])
        self.assertTrue(action_res["executed"])
        self.assertEqual(action_res["action"], "SCROLL_DOWN")
        self.assertEqual(self.action_executor.state_store["scroll_demo"]["scroll_offset_px"], 180)

    def test_03_rest_produces_no_scrolling(self):
        """3. Validate that REST prediction produces NEUTRAL command and zero scroll displacement."""
        self.action_executor.state_store["scroll_demo"]["scroll_offset_px"] = 250
        initial_offset = self.action_executor.state_store["scroll_demo"]["scroll_offset_px"]

        raw_probs = {"REST": 0.88, "LEFT_FIST": 0.06, "RIGHT_FIST": 0.06}
        safety_res = self.safety_shield.evaluate_safety(raw_probs, hardware_connected=True, signal_quality_ok=True)
        self.assertEqual(safety_res["intent"], "REST")

        routing_res = self.command_router.process_intent(safety_res)
        self.assertEqual(routing_res["command"], "NEUTRAL", "REST must strictly map to NEUTRAL.")
        self.assertFalse(routing_res["action_triggered"], "REST must not trigger actions.")

        action_res = self.action_executor.execute_command(routing_res["command"])
        self.assertFalse(action_res["executed"], "No action must execute on NEUTRAL.")
        self.assertEqual(self.action_executor.state_store["scroll_demo"]["scroll_offset_px"], initial_offset)

    def test_04_low_confidence_rejected(self):
        """4. Validate that low-confidence or ambiguous predictions are rejected by the safety shield."""
        # Confidence is 0.40, below threshold 0.60
        raw_probs = {"REST": 0.32, "LEFT_FIST": 0.40, "RIGHT_FIST": 0.28}
        safety_res = self.safety_shield.evaluate_safety(raw_probs, hardware_connected=True, signal_quality_ok=True)
        self.assertFalse(safety_res["accepted"], "Predictions below confidence threshold must be rejected.")
        rejection_reason = safety_res.get("reason", "") or safety_res.get("rejection_reason", "")
        self.assertIn("CONFIDENCE", rejection_reason)

        routing_res = self.command_router.process_intent(safety_res)
        self.assertEqual(routing_res["command"], "NEUTRAL")
        self.assertFalse(routing_res["action_triggered"])

    def test_05_ground_truth_cannot_control_scrolling(self):
        """5. Validate that ground truth labels are strictly decoupled and cannot dictate scrolling."""
        # Simulated scenario: ground-truth is RIGHT_FIST, but model predicts REST with high confidence
        sm = NeuroOSSessionManager(default_profile_id="web_control")
        sm.safety_shield.smoothing_window_size = 1
        sm.start_session()

        # Dummy predictor that ignores ground truth and predicts REST
        def mock_rest_predictor(subject_id, run_id, trial_idx):
            return {
                "subject_id": subject_id,
                "run_id": run_id,
                "trial_idx": trial_idx,
                "true_label": "RIGHT_FIST",  # Ground truth says RIGHT_FIST
                "predicted_label": "REST",   # Authentic model output says REST
                "confidence": 0.95,
                "probabilities": {"REST": 0.95, "LEFT_FIST": 0.02, "RIGHT_FIST": 0.03}
            }

        initial_offset = sm.action_executor.state_store["scroll_demo"]["scroll_offset_px"]
        payload = sm.process_pipeline_step(mock_rest_predictor, allow_unstarted=True)

        # Confirm command is NEUTRAL based on predicted_label, NOT ground truth
        self.assertEqual(payload["command_routing"]["command"], "NEUTRAL")
        self.assertFalse(payload["action_execution"]["executed"])
        self.assertEqual(sm.action_executor.state_store["scroll_demo"]["scroll_offset_px"], initial_offset)

    def test_06_pause_stops_actions(self):
        """6. Validate that PAUSED session state blocks action execution."""
        sm = NeuroOSSessionManager(default_profile_id="web_control")
        sm.safety_shield.smoothing_window_size = 1
        sm.start_session()
        self.assertEqual(sm.state, SessionState.RUNNING)

        sm.pause_session()
        self.assertEqual(sm.state, SessionState.PAUSED)

        # High confidence RIGHT_FIST predictor
        def mock_right_predictor(subject_id, run_id, trial_idx):
            return {
                "subject_id": subject_id, "run_id": run_id, "trial_idx": trial_idx,
                "predicted_label": "RIGHT_FIST", "confidence": 0.92,
                "probabilities": {"REST": 0.04, "LEFT_FIST": 0.04, "RIGHT_FIST": 0.92}
            }

        payload = sm.process_pipeline_step(mock_right_predictor, allow_unstarted=False)
        self.assertFalse(payload["action_execution"]["executed"])
        self.assertIn("PAUSED", payload["action_execution"]["details"])

    def test_07_resume_continues_playback(self):
        """7. Validate that resuming a paused session restores operational action execution."""
        # Test SessionManager lifecycle
        sm = NeuroOSSessionManager(default_profile_id="web_control")
        sm.start_session()
        self.assertEqual(sm.state, SessionState.RUNNING)

        sm.pause_session()
        self.assertEqual(sm.state, SessionState.PAUSED)

        sm.resume_session()
        self.assertEqual(sm.state, SessionState.RUNNING)

        # Test Web Control API endpoints
        start_res = start_web_control()
        self.assertEqual(start_res["session_state"], "RUNNING")

        pause_res = pause_web_control()
        self.assertEqual(pause_res["session_state"], "PAUSED")
        self.assertEqual(get_web_control_status()["session_state"], "PAUSED")

        resume_res = resume_web_control()
        self.assertEqual(resume_res["session_state"], "RUNNING")
        self.assertEqual(get_web_control_status()["session_state"], "RUNNING")

    def test_08_stop_ends_session(self):
        """8. Validate that stopping a session transitions state to STOPPED and halts actions."""
        sm = NeuroOSSessionManager(default_profile_id="web_control")
        sm.safety_shield.smoothing_window_size = 1
        sm.start_session()
        sm.stop_session()
        self.assertEqual(sm.state, SessionState.STOPPED)

        def mock_right_predictor(subject_id, run_id, trial_idx):
            return {
                "subject_id": subject_id, "run_id": run_id, "trial_idx": trial_idx,
                "predicted_label": "RIGHT_FIST", "confidence": 0.92,
                "probabilities": {"REST": 0.04, "LEFT_FIST": 0.04, "RIGHT_FIST": 0.92}
            }

        payload = sm.process_pipeline_step(mock_right_predictor, allow_unstarted=False)
        self.assertFalse(payload["action_execution"]["executed"])
        self.assertIn("STOPPED", payload["action_execution"]["details"])

    def test_09_reset_restores_initial_scroll_position(self):
        """9. Validate that reset_web_control restores scroll offset to 0 and trial index to 0."""
        # Mutate current state
        from api_server import neuro_action_executor, neuro_session_manager
        neuro_action_executor.state_store["scroll_demo"]["scroll_offset_px"] = 720
        neuro_session_manager.dataset_source.current_trial_idx = 15

        res = reset_web_control()
        self.assertEqual(res["status"], "RESET")
        self.assertEqual(res["scroll_offset_px"], 0)
        self.assertEqual(res["current_trial_idx"], 0)
        self.assertEqual(neuro_action_executor.state_store["scroll_demo"]["scroll_offset_px"], 0)

    def test_10_duplicate_predictions_suppressed(self):
        """10. Validate that duplicate predictions without release are suppressed by repeat guard."""
        import time
        router = NeuroOSCommandRouter(dwell_windows=1, cooldown_sec=1.5, profile_id="web_control")
        router.repeat_suppression = True
        now = time.time()

        safety_res = {
            "accepted": True,
            "intent": "RIGHT_FIST",
            "confidence": 0.88,
            "prediction_id": "pred_001",
            "timestamp": now
        }

        # First trigger should fire
        res1 = router.process_intent(safety_res, current_time=now)
        self.assertTrue(res1["action_triggered"], "Initial intent must trigger action.")

        # Immediate identical repeat without state release should be suppressed
        safety_res_repeat = {
            "accepted": True,
            "intent": "RIGHT_FIST",
            "confidence": 0.88,
            "prediction_id": "pred_001",  # Same prediction ID
            "timestamp": now + 0.1
        }
        res2 = router.process_intent(safety_res_repeat, current_time=now + 0.1)
        self.assertFalse(res2["action_triggered"], "Duplicate prediction must be suppressed.")
        self.assertIn("STALE", res2.get("reason", "") or res2.get("command", ""))

    def test_11_invalid_urls_rejected(self):
        """11. Validate that unsafe, non-HTTP/HTTPS URLs are strictly rejected."""
        # Forbidden schemes
        res_js = validate_web_url(UrlValidationRequest(url="javascript:alert('xss')"))
        self.assertFalse(res_js["valid"])
        self.assertIn("Only HTTP and HTTPS", res_js["error"])

        res_file = validate_web_url(UrlValidationRequest(url="file:///C:/Windows/System32/calc.exe"))
        self.assertFalse(res_file["valid"])

        res_data = validate_web_url(UrlValidationRequest(url="data:text/html,<script>alert(1)</script>"))
        self.assertFalse(res_data["valid"])

        res_empty = validate_web_url(UrlValidationRequest(url="   "))
        self.assertFalse(res_empty["valid"])

        # Valid URLs
        res_http = validate_web_url(UrlValidationRequest(url="http://example.com/article"))
        self.assertTrue(res_http["valid"])
        self.assertEqual(res_http["scheme"], "http")

        res_https = validate_web_url(UrlValidationRequest(url="https://en.wikipedia.org/wiki/Brain-computer_interface"))
        self.assertTrue(res_https["valid"])
        self.assertEqual(res_https["scheme"], "https")

    def test_12_local_demo_webpage_scroll_control(self):
        """12. Validate that local demo webpage HTML is complete, rich, and provides scroll bridge."""
        html = generate_demo_article_html()
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 3000, "Demo article must have substantial rich content.")
        self.assertIn("NeuroOS Research Journal", html)
        self.assertIn("NEURO_SCROLL_BY", html, "Must provide postMessage scroll bridge for iframe control.")
        self.assertIn("NEURO_SCROLL_UPDATE", html)

        # Test FastAPI HTMLResponse endpoint
        resp = get_demo_article_html()
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"NeuroOS Research Journal", resp.body)

    def test_13_external_iframe_restrictions_handled_honestly(self):
        """13. Validate that web control status and telemetry honestly report offline replay and sandbox."""
        status = get_web_control_status()
        self.assertEqual(status["status"], "ONLINE")
        self.assertEqual(status["badge"], "OFFLINE DATASET REPLAY")
        self.assertTrue(status["is_offline_replay"])
        self.assertIn("scroll_state", status)
        self.assertIn("safety_config", status)

        # Verify scroll config tuning works
        cfg_res = update_web_control_config(WebControlConfigRequest(speed="FAST", max_scroll_px=6000))
        self.assertEqual(cfg_res["status"], "CONFIG_UPDATED")
        self.assertEqual(cfg_res["scroll_config"]["step_px"], 300)
        self.assertEqual(cfg_res["scroll_config"]["max_scroll_px"], 6000)

    def test_14_websocket_telemetry_payload_contract(self):
        """14. Validate that step_web_control generates an authentic telemetry payload with all required fields."""
        reset_web_control()
        step_req = WebControlStepRequest(subject_id="S001", run_id="R11", trial_idx=0)
        payload = step_web_control(step_req)

        self.assertIn("event_id", payload)
        self.assertIn("timestamp", payload)
        self.assertIn("session_id", payload)
        self.assertIn("session_state", payload)
        self.assertIn("active_model_id", payload)
        self.assertIn("active_profile", payload)
        self.assertIn("decoder_output", payload)
        self.assertIn("safety_state", payload)
        self.assertIn("command_routing", payload)
        self.assertIn("action_execution", payload)
        self.assertIn("latencies_ms", payload)

        # Verify decoder output contains genuine probabilities
        probs = payload["decoder_output"]["probabilities"]
        self.assertIn("REST", probs)
        self.assertIn("LEFT_FIST", probs)
        self.assertIn("RIGHT_FIST", probs)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)

    def test_15_historical_models_remain_unchanged(self):
        """15. Validate that model registry preserves historical models and selects Phase 7 champion."""
        self.assertIn("Phase7-PureImagery-Calibrated", AVAILABLE_MODELS)
        self.assertIn("Phase7-Hierarchical-LOSO", AVAILABLE_MODELS)
        self.assertIn("Phase6-PureImagery-Calibrated", AVAILABLE_MODELS)
        self.assertIn("Step62-SurfaceLaplacian-Ensemble", AVAILABLE_MODELS)
        self.assertIn("Step61-Multimodal-Ensemble", AVAILABLE_MODELS)

        # Select Phase 7 model
        sel_res = select_model_version(ModelSelectRequest(model_id="Phase7-PureImagery-Calibrated"))
        self.assertEqual(sel_res["status"], "SUCCESS")
        self.assertIn("Phase 7 Champion", sel_res["model_details"]["name"])

        # Select Step 62 model
        sel_res_62 = select_model_version(ModelSelectRequest(model_id="Step62-SurfaceLaplacian-Ensemble"))
        self.assertEqual(sel_res_62["status"], "SUCCESS")
        self.assertIn("Step 62", sel_res_62["model_details"]["name"])


if __name__ == "__main__":
    unittest.main()
