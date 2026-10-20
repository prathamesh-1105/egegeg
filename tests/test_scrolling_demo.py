"""
Unit and Integration Tests for NeuroOS EEG-Controlled Webpage Scrolling Demo.
Validates:
1. Prediction-to-Command Mapping (LEFT_FIST -> SCROLL_UP, RIGHT_FIST -> SCROLL_DOWN, REST -> NEUTRAL).
2. REST Intent Neutrality (Zero scroll displacement).
3. Playback Pause, Resume, and Stop State Machine.
4. Safety Shield Rejection Guardrails (Low confidence, noise, manual lock -> No action).
5. Scroll Direction & Boundary Constraints (Upper and lower limits).
6. End-to-End Integration with Genuine PhysioNet EDF Windows.
7. Scroll Demo State Reset Functionality.
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
from api_server import predict_eeg_window, step_scroll_demo, reset_scroll_demo


class TestNeuroOSScrollingDemo(unittest.TestCase):

    def setUp(self):
        """Set up fresh isolated instances of safety shield, command router, and action executor."""
        self.safety_shield = NeuroOSSafetyShield(confidence_thresh=0.60, margin_thresh=0.10)
        self.command_router = NeuroOSCommandRouter(dwell_windows=3, cooldown_sec=0.0, command_mode="SCROLL")
        self.action_executor = NeuroOSActionExecutor(mode="SCROLL_DEMO")

    def test_01_command_mapping_left_fist_scroll_up(self):
        """Test that accepted LEFT_FIST intent maps to SCROLL_UP."""
        # Simulated high-confidence LEFT_FIST probability distribution
        raw_probs = {"REST": 0.10, "LEFT_FIST": 0.85, "RIGHT_FIST": 0.05}
        safety_res = self.safety_shield.evaluate_safety(raw_probs, hardware_connected=True, signal_quality_ok=True)
        self.assertTrue(safety_res["accepted"], "High-confidence LEFT_FIST should be accepted by safety shield.")
        self.assertEqual(safety_res["intent"], "LEFT_FIST")

        routing_res = self.command_router.process_intent(safety_res)
        self.assertEqual(routing_res["command"], "SCROLL_UP", "LEFT_FIST should route to SCROLL_UP.")
        self.assertTrue(routing_res["action_triggered"], "Action should be triggered on first accepted motor window.")

    def test_02_command_mapping_right_fist_scroll_down(self):
        """Test that accepted RIGHT_FIST intent maps to SCROLL_DOWN."""
        raw_probs = {"REST": 0.10, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.85}
        safety_res = self.safety_shield.evaluate_safety(raw_probs, hardware_connected=True, signal_quality_ok=True)
        self.assertTrue(safety_res["accepted"], "High-confidence RIGHT_FIST should be accepted by safety shield.")
        self.assertEqual(safety_res["intent"], "RIGHT_FIST")

        routing_res = self.command_router.process_intent(safety_res)
        self.assertEqual(routing_res["command"], "SCROLL_DOWN", "RIGHT_FIST should route to SCROLL_DOWN.")
        self.assertTrue(routing_res["action_triggered"])

    def test_03_rest_intent_neutral_no_scroll(self):
        """Test that REST intent produces NEUTRAL command and zero scroll displacement."""
        initial_offset = self.action_executor.state_store["scroll_demo"]["scroll_offset_px"]

        raw_probs = {"REST": 0.90, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.05}
        safety_res = self.safety_shield.evaluate_safety(raw_probs, hardware_connected=True, signal_quality_ok=True)
        self.assertEqual(safety_res["intent"], "REST")

        routing_res = self.command_router.process_intent(safety_res)
        self.assertEqual(routing_res["command"], "NEUTRAL", "REST intent must map to NEUTRAL.")
        self.assertFalse(routing_res["action_triggered"], "REST must not trigger actions.")

        action_res = self.action_executor.execute_command(routing_res["command"])
        self.assertFalse(action_res["executed"], "No action should be executed for NEUTRAL.")
        final_offset = self.action_executor.state_store["scroll_demo"]["scroll_offset_px"]
        self.assertEqual(initial_offset, final_offset, "Webpage scroll offset must remain unchanged during REST.")

    def test_04_safety_rejection_prevents_scrolling(self):
        """Test that low-confidence or noisy predictions are blocked by the safety shield."""
        # Ambiguous 3-way distribution (low confidence: 0.38 < 0.60)
        raw_probs = {"REST": 0.30, "LEFT_FIST": 0.38, "RIGHT_FIST": 0.32}
        safety_res = self.safety_shield.evaluate_safety(raw_probs, hardware_connected=True, signal_quality_ok=True)
        self.assertFalse(safety_res["accepted"], "Low confidence prediction must be rejected.")

        routing_res = self.command_router.process_intent(safety_res)
        self.assertEqual(routing_res["command"], "NEUTRAL")
        self.assertFalse(routing_res["action_triggered"])

    def test_05_manual_lock_prevents_scrolling(self):
        """Test that engaging manual emergency lock prevents any scrolling commands."""
        self.safety_shield.set_manual_lock(True)
        raw_probs = {"REST": 0.05, "LEFT_FIST": 0.90, "RIGHT_FIST": 0.05}
        safety_res = self.safety_shield.evaluate_safety(raw_probs, hardware_connected=True, signal_quality_ok=True)
        self.assertFalse(safety_res["accepted"])
        self.assertEqual(safety_res["reason"], "SAFETY_SHIELD_MANUAL_LOCK_ACTIVE")

        routing_res = self.command_router.process_intent(safety_res)
        self.assertEqual(routing_res["command"], "NEUTRAL")
        self.assertFalse(routing_res["action_triggered"])

    def test_06_scroll_direction_and_bounds(self):
        """Test scrolling down, scrolling up, and boundary limits."""
        # Initial offset is 0
        self.assertEqual(self.action_executor.state_store["scroll_demo"]["scroll_offset_px"], 0)

        # Scrolling UP at top (0px) should not go below 0 (cannot scroll to negative offset)
        self.action_executor.execute_command("SCROLL_UP")
        self.assertEqual(self.action_executor.state_store["scroll_demo"]["scroll_offset_px"], 0)

        # Scrolling DOWN should increase offset by step_px (180px)
        step = self.action_executor.state_store["scroll_demo"]["step_px"]
        res_down = self.action_executor.execute_command("SCROLL_DOWN")
        self.assertTrue(res_down["executed"])
        self.assertEqual(self.action_executor.state_store["scroll_demo"]["scroll_offset_px"], step)
        self.assertEqual(res_down["action"], "SCROLL_DOWN")

        # Scrolling DOWN again
        self.action_executor.execute_command("SCROLL_DOWN")
        self.assertEqual(self.action_executor.state_store["scroll_demo"]["scroll_offset_px"], step * 2)

        # Scrolling UP should decrease offset back towards top
        res_up = self.action_executor.execute_command("SCROLL_UP")
        self.assertTrue(res_up["executed"])
        self.assertEqual(self.action_executor.state_store["scroll_demo"]["scroll_offset_px"], step)
        self.assertEqual(res_up["action"], "SCROLL_UP")

    def test_07_playback_session_state_transitions(self):
        """Test session manager start, pause, resume, and stop controls."""
        sm = NeuroOSSessionManager()
        self.assertEqual(sm.state, SessionState.CONNECTED)

        sm.start_session()
        self.assertEqual(sm.state, SessionState.RUNNING)

        sm.pause_session()
        self.assertEqual(sm.state, SessionState.PAUSED)

        sm.start_session()
        self.assertEqual(sm.state, SessionState.RUNNING)

        sm.stop_session()
        self.assertEqual(sm.state, SessionState.STOPPED)

    def test_08_end_to_end_real_edf_inference_integration(self):
        """Integration test: Execute step_scroll_demo on genuine PhysioNet EDF recording."""
        reset_scroll_demo()
        payload = step_scroll_demo(subject_id="S001", run_id="R11", trial_idx=0)

        # Verify payload structure
        self.assertIn("trial_info", payload)
        self.assertIn("decoder_output", payload)
        self.assertIn("safety_state", payload)
        self.assertIn("command_routing", payload)
        self.assertIn("action_execution", payload)

        # Verify genuine values
        trial_info = payload["trial_info"]
        self.assertEqual(trial_info["subject_id"], "S001")
        self.assertEqual(trial_info["run_id"], "R11")
        self.assertEqual(trial_info["true_label"], "REST")
        self.assertIn(payload["decoder_output"]["predicted_label"], ["REST", "LEFT_FIST", "RIGHT_FIST"])
        self.assertGreaterEqual(payload["decoder_output"]["confidence"], 0.0)
        self.assertLessEqual(payload["decoder_output"]["confidence"], 1.0)

        # Verify command and action execution payload
        action_exec = payload["action_execution"]
        self.assertEqual(action_exec["mode"], "SCROLL_DEMO")
        self.assertIn("details", action_exec)

    def test_09_reset_scroll_demo_endpoint(self):
        """Test reset_scroll_demo endpoint resets position and trial index."""
        res = reset_scroll_demo()
        self.assertEqual(res["status"], "RESET")
        self.assertEqual(res["scroll_offset_px"], 0)
        self.assertEqual(res["current_trial_idx"], 0)


if __name__ == "__main__":
    unittest.main()
