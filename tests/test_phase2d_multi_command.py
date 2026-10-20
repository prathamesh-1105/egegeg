"""
Unit and Integration Tests for NeuroOS Phase 2D: Safe Multi-Command BCI Interface.

Validates:
1. Configurable Command Profiles (Profile A Scrolling, Profile B Document Navigation, Profile C Presentation).
2. Profile Mapping Integrity (All three classes: LEFT_FIST, RIGHT_FIST, REST mapped accurately).
3. REST Intent Neutrality across all profiles (Zero state displacement/action).
4. Safety Shield Rejection Guardrails (Low confidence, noise, disconnections -> Zero action).
5. Stale Prediction Guardrail (Duplicate prediction IDs / expired timestamps cannot re-trigger actions).
6. Emergency Lock Guardrail (Immediate lockout suppresses all actions).
7. Cooldown & Temporal Confirmation Guardrails.
8. Session State Lifecycle (Start, Pause, Resume, Stop).
9. Profile Switching State Isolation (Switching does not corrupt session state or counters).
10. Safe Isolated Execution Targets (Scrolling, Section Reader, Slide Viewer boundaries).
"""

import os
import sys
import unittest
import time
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.neuro_os.command_profiles import (
    CommandProfile,
    COMMAND_PROFILES,
    get_command_profile,
    list_available_profiles,
    PROFILE_SCROLLING,
    PROFILE_DOCUMENT_NAVIGATION,
    PROFILE_PRESENTATION
)
from src.neuro_os.safety_shield import NeuroOSSafetyShield
from src.neuro_os.command_router import NeuroOSCommandRouter
from src.neuro_os.action_executor import NeuroOSActionExecutor
from src.neuro_os.session_manager import NeuroOSSessionManager, SessionState
from api_server import (
    predict_eeg_window,
    step_multi_command,
    reset_multi_command,
    step_scroll_demo,
    reset_scroll_demo
)


class TestNeuroOSPhase2DMultiCommand(unittest.TestCase):

    def setUp(self):
        """Set up fresh isolated instances of safety shield, router, and executor."""
        self.safety_shield = NeuroOSSafetyShield(confidence_thresh=0.60, margin_thresh=0.10)
        self.command_router = NeuroOSCommandRouter(
            dwell_windows=3,
            cooldown_sec=0.0,  # 0.0 for immediate mapping tests unless testing cooldown
            profile_id="scrolling",
            temporal_confirmation_windows=1
        )
        self.action_executor = NeuroOSActionExecutor(mode="SCROLL_DEMO")
        self.session_manager = NeuroOSSessionManager(default_profile_id="scrolling")

    # --------------------------------------------------------------------------
    # 1. PROFILE MAPPINGS INTEGRITY
    # --------------------------------------------------------------------------

    def test_01_profile_a_scrolling_mappings(self):
        """Verify Profile A — Scrolling maps LEFT_FIST -> SCROLL_UP, RIGHT_FIST -> SCROLL_DOWN, REST -> NEUTRAL."""
        self.command_router.set_profile("scrolling")
        self.action_executor.set_adapter_mode("SCROLL_DEMO")

        # LEFT_FIST -> SCROLL_UP
        self.safety_shield.reset()
        raw_left = {"REST": 0.05, "LEFT_FIST": 0.90, "RIGHT_FIST": 0.05}
        s_left = self.safety_shield.evaluate_safety(raw_left)
        r_left = self.command_router.process_intent(s_left)
        self.assertEqual(r_left["command"], "SCROLL_UP")
        self.assertTrue(r_left["action_triggered"])
        a_left = self.action_executor.execute_command(r_left["command"])
        self.assertEqual(a_left["action_name"], "SCROLL_UP")

        # RIGHT_FIST -> SCROLL_DOWN
        self.safety_shield.reset()
        raw_right = {"REST": 0.05, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.90}
        s_right = self.safety_shield.evaluate_safety(raw_right)
        r_right = self.command_router.process_intent(s_right)
        self.assertEqual(r_right["command"], "SCROLL_DOWN")
        self.assertTrue(r_right["action_triggered"])
        a_right = self.action_executor.execute_command(r_right["command"])
        self.assertEqual(a_right["action_name"], "SCROLL_DOWN")

        # REST -> NEUTRAL
        self.safety_shield.reset()
        raw_rest = {"REST": 0.90, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.05}
        s_rest = self.safety_shield.evaluate_safety(raw_rest)
        r_rest = self.command_router.process_intent(s_rest)
        self.assertEqual(r_rest["command"], "NEUTRAL")
        self.assertFalse(r_rest["action_triggered"])
        a_rest = self.action_executor.execute_command(r_rest["command"])
        self.assertFalse(a_rest["executed"])

    def test_02_profile_b_document_navigation_mappings(self):
        """Verify Profile B — Document Navigation maps LEFT_FIST -> PREVIOUS_SECTION, RIGHT_FIST -> NEXT_SECTION, REST -> NEUTRAL."""
        self.command_router.set_profile("document_navigation")
        self.action_executor.set_adapter_mode("DOCUMENT_NAVIGATION")

        # LEFT_FIST -> PREVIOUS_SECTION
        self.safety_shield.reset()
        raw_left = {"REST": 0.05, "LEFT_FIST": 0.90, "RIGHT_FIST": 0.05}
        s_left = self.safety_shield.evaluate_safety(raw_left)
        r_left = self.command_router.process_intent(s_left)
        self.assertEqual(r_left["command"], "PREVIOUS_SECTION")
        self.assertTrue(r_left["action_triggered"])
        a_left = self.action_executor.execute_command(r_left["command"])
        self.assertEqual(a_left["action_name"], "PREVIOUS_SECTION")

        # RIGHT_FIST -> NEXT_SECTION
        self.safety_shield.reset()
        raw_right = {"REST": 0.05, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.90}
        s_right = self.safety_shield.evaluate_safety(raw_right)
        r_right = self.command_router.process_intent(s_right)
        self.assertEqual(r_right["command"], "NEXT_SECTION")
        self.assertTrue(r_right["action_triggered"])
        a_right = self.action_executor.execute_command(r_right["command"])
        self.assertEqual(a_right["action_name"], "NEXT_SECTION")

        # REST -> NEUTRAL
        self.safety_shield.reset()
        raw_rest = {"REST": 0.90, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.05}
        s_rest = self.safety_shield.evaluate_safety(raw_rest)
        r_rest = self.command_router.process_intent(s_rest)
        self.assertEqual(r_rest["command"], "NEUTRAL")
        self.assertFalse(r_rest["action_triggered"])
        a_rest = self.action_executor.execute_command(r_rest["command"])
        self.assertFalse(a_rest["executed"])

    def test_03_profile_c_presentation_mappings(self):
        """Verify Profile C — Presentation Demo maps LEFT_FIST -> PREVIOUS_SLIDE, RIGHT_FIST -> NEXT_SLIDE, REST -> NEUTRAL."""
        self.command_router.set_profile("presentation")
        self.action_executor.set_adapter_mode("PRESENTATION")

        # LEFT_FIST -> PREVIOUS_SLIDE
        self.safety_shield.reset()
        raw_left = {"REST": 0.05, "LEFT_FIST": 0.90, "RIGHT_FIST": 0.05}
        s_left = self.safety_shield.evaluate_safety(raw_left)
        r_left = self.command_router.process_intent(s_left)
        self.assertEqual(r_left["command"], "PREVIOUS_SLIDE")
        self.assertTrue(r_left["action_triggered"])
        a_left = self.action_executor.execute_command(r_left["command"])
        self.assertEqual(a_left["action_name"], "PREVIOUS_SLIDE")

        # RIGHT_FIST -> NEXT_SLIDE
        self.safety_shield.reset()
        raw_right = {"REST": 0.05, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.90}
        s_right = self.safety_shield.evaluate_safety(raw_right)
        r_right = self.command_router.process_intent(s_right)
        self.assertEqual(r_right["command"], "NEXT_SLIDE")
        self.assertTrue(r_right["action_triggered"])
        a_right = self.action_executor.execute_command(r_right["command"])
        self.assertEqual(a_right["action_name"], "NEXT_SLIDE")

        # REST -> NEUTRAL
        self.safety_shield.reset()
        raw_rest = {"REST": 0.90, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.05}
        s_rest = self.safety_shield.evaluate_safety(raw_rest)
        r_rest = self.command_router.process_intent(s_rest)
        self.assertEqual(r_rest["command"], "NEUTRAL")
        self.assertFalse(r_rest["action_triggered"])
        a_rest = self.action_executor.execute_command(r_rest["command"])
        self.assertFalse(a_rest["executed"])

    # --------------------------------------------------------------------------
    # 2. PROFILE SWITCHING & STATE ISOLATION
    # --------------------------------------------------------------------------

    def test_04_profile_switching_preserves_session_state(self):
        """Verify profile switching does not corrupt session state, counters, or demo targets."""
        sm = self.session_manager
        sm.start_session()
        self.assertEqual(sm.state, SessionState.RUNNING)
        sm.session_counters["actions_executed_count"] = 5

        # Switch to document_navigation
        sm.set_profile("document_navigation")
        self.assertEqual(sm.command_router.profile_id, "document_navigation")
        self.assertEqual(sm.action_executor.mode, "DOCUMENT_NAVIGATION")
        self.assertEqual(sm.state, SessionState.RUNNING, "Session state must remain RUNNING across profile switches.")
        self.assertEqual(sm.session_counters["actions_executed_count"], 5, "Counters must not be reset on profile switch.")

        # Switch to presentation
        sm.set_profile("presentation")
        self.assertEqual(sm.command_router.profile_id, "presentation")
        self.assertEqual(sm.action_executor.mode, "PRESENTATION")
        self.assertEqual(sm.state, SessionState.RUNNING)

        # Switch back to scrolling
        sm.set_profile("scrolling")
        self.assertEqual(sm.command_router.profile_id, "scrolling")
        self.assertEqual(sm.action_executor.mode, "SCROLL_DEMO")

    # --------------------------------------------------------------------------
    # 3. SAFETY & REST NEUTRALITY
    # --------------------------------------------------------------------------

    def test_05_rest_intent_produces_no_action_across_all_profiles(self):
        """Verify REST intent strictly produces NEUTRAL and zero action in every profile."""
        profiles = ["scrolling", "document_navigation", "presentation"]
        raw_rest = {"REST": 0.95, "LEFT_FIST": 0.03, "RIGHT_FIST": 0.02}
        s_res = self.safety_shield.evaluate_safety(raw_rest)

        for prof in profiles:
            self.command_router.set_profile(prof)
            self.action_executor.set_adapter_mode(self.command_router.active_profile.adapter_mode)
            routing = self.command_router.process_intent(s_res)
            self.assertEqual(routing["command"], "NEUTRAL", f"REST must be NEUTRAL in {prof}")
            self.assertFalse(routing["action_triggered"], f"REST must not trigger action in {prof}")
            action = self.action_executor.execute_command(routing["command"])
            self.assertFalse(action["executed"], f"Action must not execute for REST in {prof}")

    def test_06_rejected_predictions_produce_no_action(self):
        """Verify that low-confidence or noisy predictions trigger no actions."""
        # Ambiguous distribution: confidence 0.38 < 0.60
        raw_noisy = {"REST": 0.31, "LEFT_FIST": 0.38, "RIGHT_FIST": 0.31}
        s_res = self.safety_shield.evaluate_safety(raw_noisy)
        self.assertFalse(s_res["accepted"])

        routing = self.command_router.process_intent(s_res)
        self.assertEqual(routing["command"], "NEUTRAL")
        self.assertFalse(routing["action_triggered"])
        action = self.action_executor.execute_command(routing["command"])
        self.assertFalse(action["executed"])

    # --------------------------------------------------------------------------
    # 4. STALE PREDICTION & REPEAT ACTION GUARDRAILS
    # --------------------------------------------------------------------------

    def test_07_stale_predictions_cannot_trigger_repeated_commands(self):
        """Verify that duplicate prediction IDs or flagged stale predictions are suppressed."""
        pred_id = "pred_S001_R11_001_seq10"
        s_res = {
            "accepted": True,
            "intent": "LEFT_FIST",
            "confidence": 0.88,
            "prediction_id": pred_id,
            "timestamp": time.time(),
            "is_stale": False
        }

        # First consumption: Action should trigger
        r1 = self.command_router.process_intent(s_res)
        self.assertTrue(r1["action_triggered"], "First fresh prediction should trigger action.")
        self.assertEqual(r1["command"], "SCROLL_UP")

        # Second consumption of identical prediction_id: MUST BE SUPPRESSED
        r2 = self.command_router.process_intent(s_res)
        self.assertFalse(r2["action_triggered"], "Duplicate prediction ID must be suppressed as stale.")
        self.assertEqual(r2["command"], "NEUTRAL")
        self.assertIn("STALE_DUPLICATE_PREDICTION_SUPPRESSED", r2["reason"])

        # Flagged stale prediction: MUST BE SUPPRESSED
        stale_explicit = {
            "accepted": True,
            "intent": "RIGHT_FIST",
            "confidence": 0.90,
            "prediction_id": "pred_new_002",
            "is_stale": True
        }
        r3 = self.command_router.process_intent(stale_explicit)
        self.assertFalse(r3["action_triggered"], "Explicit is_stale=True must be suppressed.")
        self.assertEqual(r3["command"], "NEUTRAL")
        self.assertIn("STALE_PREDICTION_SUPPRESSED", r3["reason"])

    # --------------------------------------------------------------------------
    # 5. EMERGENCY LOCK & DISENGAGE
    # --------------------------------------------------------------------------

    def test_08_emergency_lock_blocks_execution(self):
        """Verify manual emergency lock immediately prevents any action execution."""
        self.safety_shield.set_manual_lock(True)

        raw_motor = {"REST": 0.01, "LEFT_FIST": 0.98, "RIGHT_FIST": 0.01}
        s_res = self.safety_shield.evaluate_safety(raw_motor)
        self.assertFalse(s_res["accepted"], "Safety shield must reject predictions while emergency locked.")
        self.assertEqual(s_res["reason"], "SAFETY_SHIELD_MANUAL_LOCK_ACTIVE")

        routing = self.command_router.process_intent(s_res)
        self.assertEqual(routing["command"], "NEUTRAL")
        self.assertFalse(routing["action_triggered"])

        action = self.action_executor.execute_command(routing["command"])
        self.assertFalse(action["executed"])

        # Release lock
        self.safety_shield.set_manual_lock(False)
        s_unlocked = self.safety_shield.evaluate_safety(raw_motor)
        self.assertTrue(s_unlocked["accepted"])

    # --------------------------------------------------------------------------
    # 6. COOLDOWN & TEMPORAL CONFIRMATION
    # --------------------------------------------------------------------------

    def test_09_cooldown_guardrail_prevents_rapid_fire(self):
        """Verify configurable cooldown blocks subsequent triggers within cooldown window."""
        self.command_router.cooldown_sec = 0.5  # 500 ms cooldown
        raw_left = {"REST": 0.05, "LEFT_FIST": 0.90, "RIGHT_FIST": 0.05}

        # Step 1: fires command
        s1 = self.safety_shield.evaluate_safety(raw_left)
        s1["prediction_id"] = "p1"
        r1 = self.command_router.process_intent(s1)
        self.assertTrue(r1["action_triggered"])

        # Step 2: immediate second trigger (<0.5s) must be blocked by cooldown
        s2 = self.safety_shield.evaluate_safety(raw_left)
        s2["prediction_id"] = "p2"
        r2 = self.command_router.process_intent(s2)
        self.assertFalse(r2["action_triggered"], "Rapid command within cooldown window must be blocked.")
        self.assertIn("COOLDOWN_ACTIVE", r2["reason"])

    def test_10_temporal_confirmation_guardrail(self):
        """Verify optional temporal confirmation requires N consecutive windows before firing."""
        self.command_router.cooldown_sec = 0.0
        self.command_router.set_temporal_confirmation_windows(2)  # Requires 2 consecutive windows

        raw_right = {"REST": 0.05, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.90}

        # Window 1: Pending confirmation (1/2)
        s1 = self.safety_shield.evaluate_safety(raw_right)
        s1["prediction_id"] = "w1"
        r1 = self.command_router.process_intent(s1)
        self.assertFalse(r1["action_triggered"], "Window 1 should be pending confirmation.")
        self.assertIn("TEMPORAL_CONFIRMATION_PENDING", r1["reason"])

        # Window 2: Consecutive match (2/2) -> Action fires
        s2 = self.safety_shield.evaluate_safety(raw_right)
        s2["prediction_id"] = "w2"
        r2 = self.command_router.process_intent(s2)
        self.assertTrue(r2["action_triggered"], "Window 2 matching consecutive intent must fire.")
        self.assertEqual(r2["command"], "SCROLL_DOWN")

    # --------------------------------------------------------------------------
    # 7. SESSION LIFECYCLE (START, PAUSE, RESUME, STOP)
    # --------------------------------------------------------------------------

    def test_11_session_state_lifecycle_and_action_gating(self):
        """Verify session manager start, pause, resume, stop lifecycle and action blocking."""
        sm = self.session_manager
        sm.stop_session()
        self.assertEqual(sm.state, SessionState.STOPPED)

        # While STOPPED, pipeline step does NOT execute actions
        res_stopped = sm.process_pipeline_step(predict_eeg_window, allow_unstarted=False)
        self.assertFalse(res_stopped["action_execution"]["executed"])
        self.assertIn("Action blocked: Session is STOPPED", res_stopped["action_execution"]["details"])

        # Start Session -> RUNNING
        sm.start_session()
        self.assertEqual(sm.state, SessionState.RUNNING)

        # Pause Session -> PAUSED
        sm.pause_session()
        self.assertEqual(sm.state, SessionState.PAUSED)
        res_paused = sm.process_pipeline_step(predict_eeg_window, allow_unstarted=False)
        self.assertFalse(res_paused["action_execution"]["executed"])
        self.assertIn("Action blocked: Session is PAUSED", res_paused["action_execution"]["details"])

        # Resume Session -> RUNNING
        sm.resume_session()
        self.assertEqual(sm.state, SessionState.RUNNING)

        # Stop Session -> STOPPED
        sm.stop_session()
        self.assertEqual(sm.state, SessionState.STOPPED)

    # --------------------------------------------------------------------------
    # 8. SAFE ISOLATED DEMO TARGETS EXECUTION & BOUNDS
    # --------------------------------------------------------------------------

    def test_12_safe_isolated_targets_bounds(self):
        """Verify boundaries for Document Reader (Sections 1-6) and Slide Viewer (Slides 1-5)."""
        # Section Reader
        self.action_executor.set_adapter_mode("DOCUMENT_NAVIGATION")
        doc_st = self.action_executor.state_store["document_navigation"]
        doc_st["active_section_idx"] = 0

        # Cannot navigate before Section 0
        self.action_executor.execute_command("PREVIOUS_SECTION")
        self.assertEqual(doc_st["active_section_idx"], 0)

        # Advance to Section 1
        res = self.action_executor.execute_command("NEXT_SECTION")
        self.assertTrue(res["executed"])
        self.assertEqual(doc_st["active_section_idx"], 1)

        # Presentation Slide Viewer
        self.action_executor.set_adapter_mode("PRESENTATION")
        pres_st = self.action_executor.state_store["presentation"]
        pres_st["active_slide"] = 1

        # Cannot move before Slide 1
        self.action_executor.execute_command("PREVIOUS_SLIDE")
        self.assertEqual(pres_st["active_slide"], 1)

        # Advance to Slide 2
        res_slide = self.action_executor.execute_command("NEXT_SLIDE")
        self.assertTrue(res_slide["executed"])
        self.assertEqual(pres_st["active_slide"], 2)

    # --------------------------------------------------------------------------
    # 9. INTEGRATION ENDPOINTS
    # --------------------------------------------------------------------------

    def test_13_multi_command_endpoints_integration(self):
        """Test step_multi_command and reset_multi_command API functions on real EDF windows."""
        reset_res = reset_multi_command()
        self.assertEqual(reset_res["status"], "RESET")

        # Step Profile B on genuine EDF recording
        step_res = step_multi_command(subject_id="S001", run_id="R11", trial_idx=0, profile_id="document_navigation")
        self.assertIn("trial_info", step_res)
        self.assertIn("decoder_output", step_res)
        self.assertIn("safety_state", step_res)
        self.assertIn("command_routing", step_res)
        self.assertIn("action_execution", step_res)
        self.assertIn("session_counters", step_res)
        self.assertEqual(step_res["command_routing"]["profile_id"], "document_navigation")


if __name__ == "__main__":
    unittest.main()
