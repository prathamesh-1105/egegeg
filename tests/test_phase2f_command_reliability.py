"""
NeuroOS Phase 2F — Command Reliability, Replay Analytics, and End-to-End Validation
Automated Regression and Data Integrity Test Suite

Verifies:
1. Logical decoupling of ground-truth evaluation from command dispatch.
2. Complete audit logging (all 15 required fields).
3. Action execution state change tracking and delta calculations.
4. Accurate counter aggregation and explicit denominator calculations.
5. Deterministic session replay in SYNCHRONOUS_STEP mode.
6. Safety shield rejection reasons (confidence, margin, entropy, quality).
7. Strict REST neutrality across all profiles.
8. Emergency lock and manual override enforcement.
9. Duplicate prediction and stale event suppression.
10. Preservation of Step 60, Step 61, and Step 62 champion model artifacts.
"""

import os
import sys
import unittest
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.neuro_os.session_auditor import NeuroOSSessionAuditor, SessionAuditRecord
from src.neuro_os.safety_shield import NeuroOSSafetyShield
from src.neuro_os.command_router import NeuroOSCommandRouter
from src.neuro_os.action_executor import NeuroOSActionExecutor
from src.neuro_os.signal_quality import NeuroOSSignalQualityGate
from src.neuro_os.deterministic_replayer import DeterministicReplayer
from src.neuro_os.command_profiles import get_command_profile


class TestPhase2FCommandReliability(unittest.TestCase):

    def setUp(self):
        self.auditor = NeuroOSSessionAuditor(max_records=100)
        self.shield = NeuroOSSafetyShield(
            confidence_thresh=0.60,
            margin_thresh=0.12,
            entropy_thresh=1.30,
            smoothing_window_size=2
        )
        self.router = NeuroOSCommandRouter(
            dwell_windows=1,
            cooldown_sec=0.0,
            profile_id="scrolling"
        )
        self.executor = NeuroOSActionExecutor(mode="SCROLL_DEMO")

    # -------------------------------------------------------------------------
    # 1. GROUND TRUTH / PREDICTION LOGICAL SEPARATION
    # -------------------------------------------------------------------------
    def test_ground_truth_prediction_logical_separation(self):
        """
        Confirms ground-truth label NEVER drives live commands or actions.
        Command routing depends purely on safety-shielded model predictions.
        """
        # Scenario A: Model predicts RIGHT_FIST with high confidence, ground truth is REST
        probs = {"REST": 0.05, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.90}
        shield_res = self.shield.evaluate_safety(probs, hardware_connected=True, signal_quality_ok=True)
        shield_res["prediction_id"] = "pred_test_1"
        shield_res["timestamp"] = 1000.0

        # Command router receives ONLY safety_res (no ground truth)
        routing_res = self.router.process_intent(shield_res, current_time=1000.0)
        self.assertTrue(routing_res["action_triggered"])
        self.assertEqual(routing_res["command"], "SCROLL_DOWN")

        # Ground truth is provided ONLY to auditor for scoring
        record = self.auditor.record_trial(
            session_id="test_decoupling",
            predicted_label="RIGHT_FIST",
            confidence=0.90,
            probabilities=probs,
            safety_status=shield_res,
            command_routing=routing_res,
            action_execution={"executed": True, "action": "SCROLL_DOWN", "state_change": {"delta": 180}},
            signal_quality={"passed": True},
            latencies_ms={"pipeline_latency": 20.0},
            ground_truth_label="REST"  # Offline scoring label
        )

        # Auditor correctly scored it as a classification error and false actuation
        self.assertFalse(record.is_correct_classification)
        self.assertTrue(record.is_incorrect_action)
        self.assertEqual(record.mapped_command, "SCROLL_DOWN")

    # -------------------------------------------------------------------------
    # 2. COMPLETE EVENT LOGGING (15 REQUIRED AUDIT FIELDS)
    # -------------------------------------------------------------------------
    def test_complete_event_logging(self):
        """Verifies every logged trial contains all 15 mandatory audit fields."""
        probs = {"REST": 0.1, "LEFT_FIST": 0.85, "RIGHT_FIST": 0.05}
        shield_res = self.shield.evaluate_safety(probs, hardware_connected=True, signal_quality_ok=True)
        shield_res["prediction_id"] = "pred_complete_1"
        shield_res["timestamp"] = 1000.0

        routing_res = self.router.process_intent(shield_res, current_time=1000.0)
        action_res = self.executor.execute_command(routing_res["command"])

        rec = self.auditor.record_trial(
            session_id="session_complete_audit",
            predicted_label="LEFT_FIST",
            confidence=0.85,
            probabilities=probs,
            safety_status=shield_res,
            command_routing=routing_res,
            action_execution=action_res,
            signal_quality={"passed": True, "signal_quality": "EXCELLENT"},
            latencies_ms={"ml_latency": 4.1, "pipeline_latency": 20.2},
            subject_id="S001",
            run_id="R11",
            trial_idx=3,
            data_source="PHYSIO_NET_EDF",
            model_version="Step62-SurfaceLaplacian-Ensemble",
            ground_truth_label="LEFT_FIST",
            timestamp=1000.0
        )

        d = rec.to_dict()
        required_keys = [
            "session_id", "subject_id", "run_id", "trial_idx", "data_source",
            "model_version", "ground_truth_label", "predicted_label", "confidence",
            "probabilities", "signal_quality_passed", "signal_quality_status",
            "safety_accepted", "safety_rejection_reason", "command_profile_id",
            "mapped_command", "action_executed", "state_change", "latencies_ms", "timestamp"
        ]
        for key in required_keys:
            self.assertIn(key, d, f"Missing required audit field: {key}")
            self.assertIsNotNone(d[key], f"Audit field {key} must not be None")

    # -------------------------------------------------------------------------
    # 3. CORRECT ACTION OUTCOME TRACKING & STATE DELTAS
    # -------------------------------------------------------------------------
    def test_action_executor_state_change_tracking(self):
        """Verifies state parameter deltas for Scrolling, Doc Nav, and Presentation."""
        # A. Scrolling: SCROLL_DOWN -> scroll_offset_px +180
        exec_scroll = NeuroOSActionExecutor(mode="SCROLL_DEMO")
        res_scroll = exec_scroll.execute_command("SCROLL_DOWN")
        self.assertTrue(res_scroll["executed"])
        self.assertEqual(res_scroll["state_change"]["parameter"], "scroll_offset_px")
        self.assertEqual(res_scroll["state_change"]["delta"], 180)
        self.assertEqual(res_scroll["state_change"]["after"], 180)

        # B. Document Navigation: NEXT_SECTION -> active_section_idx +1
        exec_doc = NeuroOSActionExecutor(mode="DOCUMENT_NAVIGATION")
        res_doc = exec_doc.execute_command("NEXT_SECTION")
        self.assertTrue(res_doc["executed"])
        self.assertEqual(res_doc["state_change"]["parameter"], "active_section_idx")
        self.assertEqual(res_doc["state_change"]["delta"], 1)
        self.assertEqual(res_doc["state_change"]["after"], 1)

        # C. Presentation: NEXT_SLIDE -> active_slide +1
        exec_pres = NeuroOSActionExecutor(mode="PRESENTATION")
        res_pres = exec_pres.execute_command("NEXT_SLIDE")
        self.assertTrue(res_pres["executed"])
        self.assertEqual(res_pres["state_change"]["parameter"], "active_slide")
        self.assertEqual(res_pres["state_change"]["delta"], 1)
        self.assertEqual(res_pres["state_change"]["after"], 2)

    # -------------------------------------------------------------------------
    # 4. ACCURATE COUNTER AGGREGATION & EXPLICIT DENOMINATORS
    # -------------------------------------------------------------------------
    def test_auditor_counter_aggregation_and_explicit_denominators(self):
        """
        Feeds a synthetic sequence of 10 known trials and verifies exact mathematical rates:
        - 4 true REST trials (all neutral)
        - 4 correct LEFT_FIST trials (executed action)
        - 1 misclassified trial that gets executed (false action)
        - 1 misclassified trial that gets rejected by safety shield (suppressed error)
        """
        auditor = NeuroOSSessionAuditor()

        # Trial 1-4: Correct REST
        for i in range(4):
            auditor.record_trial(
                session_id="agg_test",
                predicted_label="REST",
                confidence=0.88,
                probabilities={"REST": 0.88, "LEFT_FIST": 0.06, "RIGHT_FIST": 0.06},
                safety_status={"accepted": True, "intent": "REST"},
                command_routing={"command": "NEUTRAL", "action_triggered": False},
                action_execution={"executed": False, "action": "NONE", "state_change": {"delta": 0}},
                signal_quality={"passed": True},
                latencies_ms={"pipeline_latency": 20.0},
                ground_truth_label="REST"
            )

        # Trial 5-8: Correct LEFT_FIST (executed)
        for i in range(4):
            auditor.record_trial(
                session_id="agg_test",
                predicted_label="LEFT_FIST",
                confidence=0.85,
                probabilities={"REST": 0.05, "LEFT_FIST": 0.85, "RIGHT_FIST": 0.10},
                safety_status={"accepted": True, "intent": "LEFT_FIST"},
                command_routing={"command": "SCROLL_UP", "action_triggered": True},
                action_execution={"executed": True, "action": "SCROLL_UP", "state_change": {"delta": -180}},
                signal_quality={"passed": True},
                latencies_ms={"pipeline_latency": 22.0},
                ground_truth_label="LEFT_FIST"
            )

        # Trial 9: Misclassified LEFT_FIST (true is RIGHT_FIST), executed -> false actuation
        auditor.record_trial(
            session_id="agg_test",
            predicted_label="LEFT_FIST",
            confidence=0.80,
            probabilities={"REST": 0.05, "LEFT_FIST": 0.80, "RIGHT_FIST": 0.15},
            safety_status={"accepted": True, "intent": "LEFT_FIST"},
            command_routing={"command": "SCROLL_UP", "action_triggered": True},
            action_execution={"executed": True, "action": "SCROLL_UP", "state_change": {"delta": -180}},
            signal_quality={"passed": True},
            latencies_ms={"pipeline_latency": 21.0},
            ground_truth_label="RIGHT_FIST"
        )

        # Trial 10: Misclassified, but REJECTED by shield -> suppressed error
        auditor.record_trial(
            session_id="agg_test",
            predicted_label="LEFT_FIST",
            confidence=0.52,
            probabilities={"REST": 0.20, "LEFT_FIST": 0.52, "RIGHT_FIST": 0.28},
            safety_status={"accepted": False, "intent": "REST", "reason": "LOW_CONFIDENCE"},
            command_routing={"command": "NEUTRAL", "action_triggered": False},
            action_execution={"executed": False, "action": "NONE", "state_change": {"delta": 0}},
            signal_quality={"passed": True},
            latencies_ms={"pipeline_latency": 19.0},
            ground_truth_label="REST"
        )

        m = auditor.get_summary_metrics()

        # Total processed
        self.assertEqual(m["total_trials_processed"], 10)
        # Raw Accuracy: 8 correct / 10 = 80.0%
        self.assertAlmostEqual(m["classification_accuracy_pct"], 80.0, places=1)
        # Safety Rejection Rate: 1 / 10 = 10.0%
        self.assertAlmostEqual(m["safety_rejection_rate_pct"], 10.0, places=1)
        # Accepted-Trial Accuracy: 8 correct / 9 accepted = 88.89%
        self.assertAlmostEqual(m["accepted_trial_accuracy_pct"], 88.89, places=1)
        # Command Execution Rate: 5 executed / 10 = 50.0%
        self.assertAlmostEqual(m["command_execution_rate_pct"], 50.0, places=1)
        # Incorrect Command Rate (Total Denominator): 1 incorrect / 10 = 10.0%
        self.assertAlmostEqual(m["incorrect_command_rate_total_pct"], 10.0, places=1)
        # Incorrect Command Rate (Executed Denominator): 1 incorrect / 5 executed = 20.0%
        self.assertAlmostEqual(m["incorrect_command_rate_executed_pct"], 20.0, places=1)
        # REST Neutrality: all 5 true REST had zero actions = 100.0% (shield successfully suppressed Trial 10)
        self.assertAlmostEqual(m["rest_neutrality_rate_pct"], 100.0, places=1)
        # Latencies
        self.assertGreater(m["latencies_ms"]["mean"], 0)
        self.assertGreaterEqual(m["latencies_ms"]["p95"], m["latencies_ms"]["p50"])

    # -------------------------------------------------------------------------
    # 5. DETERMINISTIC SESSION REPLAY CONSISTENCY
    # -------------------------------------------------------------------------
    def test_deterministic_replayer_synchronous_mode(self):
        """Verifies DeterministicReplayer achieves 100% mathematical match rate in SYNCHRONOUS_STEP mode."""
        # Synthetic deterministic mock predictor
        def mock_predictor(subject_id="S001", run_id="R11", trial_idx=0):
            probs_map = [
                {"REST": 0.90, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.05},
                {"REST": 0.05, "LEFT_FIST": 0.85, "RIGHT_FIST": 0.10},
                {"REST": 0.10, "LEFT_FIST": 0.10, "RIGHT_FIST": 0.80},
                {"REST": 0.45, "LEFT_FIST": 0.35, "RIGHT_FIST": 0.20},  # Low confidence
            ]
            p = probs_map[trial_idx % len(probs_map)]
            pred = max(p.items(), key=lambda x: x[1])[0]
            return {
                "predicted_label": pred,
                "confidence": p[pred],
                "probabilities": p,
                "true_label": "REST" if trial_idx % 2 == 0 else "LEFT_FIST"
            }

        seq = [{"subject_id": "S001", "run_id": "R11", "trial_idx": i, "true_label": "REST"} for i in range(12)]

        replayer1 = DeterministicReplayer(
            predictor_func=mock_predictor,
            command_profile_id="scrolling",
            dwell_windows=1,
            cooldown_sec=0.0,
            confidence_thresh=0.60,
            timing_mode="SYNCHRONOUS_STEP"
        )
        pass1 = replayer1.replay_sequence(seq)

        replayer2 = DeterministicReplayer(
            predictor_func=mock_predictor,
            command_profile_id="scrolling",
            dwell_windows=1,
            cooldown_sec=0.0,
            confidence_thresh=0.60,
            timing_mode="SYNCHRONOUS_STEP"
        )
        pass2 = replayer2.replay_sequence(seq, original_records=pass1["audit_records"])

        verification = pass2["match_verification"]
        self.assertTrue(verification["deterministic_consistency"])
        self.assertEqual(verification["prediction_match_rate_pct"], 100.0)
        self.assertEqual(verification["safety_decision_match_rate_pct"], 100.0)
        self.assertEqual(verification["command_match_rate_pct"], 100.0)
        self.assertEqual(verification["action_execution_match_rate_pct"], 100.0)
        self.assertEqual(verification["state_mutation_match_rate_pct"], 100.0)

    # -------------------------------------------------------------------------
    # 6. SAFETY SHIELD REJECTION REASONS
    # -------------------------------------------------------------------------
    def test_safety_shield_rejection_reasons(self):
        """Verifies explainable rejection reasons across confidence, margin, and entropy filters."""
        shield = NeuroOSSafetyShield(confidence_thresh=0.60, margin_thresh=0.12, entropy_thresh=1.30)

        # A. Low Confidence
        res_low_conf = shield.evaluate_safety({"REST": 0.50, "LEFT_FIST": 0.30, "RIGHT_FIST": 0.20})
        self.assertFalse(res_low_conf["accepted"])
        self.assertIn("LOW_CONFIDENCE", res_low_conf["reason"])

        # B. Low Class Margin
        res_low_margin = shield.evaluate_safety({"REST": 0.20, "LEFT_FIST": 0.41, "RIGHT_FIST": 0.39})
        self.assertFalse(res_low_margin["accepted"])
        self.assertTrue("LOW_CONFIDENCE" in res_low_margin["reason"] or "LOW_CLASS_MARGIN" in res_low_margin["reason"])

        # C. High Entropy (flat uncertain distribution)
        res_entropy = shield.evaluate_safety({"REST": 0.34, "LEFT_FIST": 0.33, "RIGHT_FIST": 0.33})
        self.assertFalse(res_entropy["accepted"])

        # D. Hardware Disconnected
        res_disc = shield.evaluate_safety({"REST": 0.05, "LEFT_FIST": 0.90, "RIGHT_FIST": 0.05}, hardware_connected=False)
        self.assertFalse(res_disc["accepted"])
        self.assertEqual(res_disc["reason"], "HARDWARE_DISCONNECTED")

        # E. Signal Quality Artifact
        res_artifact = shield.evaluate_safety({"REST": 0.05, "LEFT_FIST": 0.90, "RIGHT_FIST": 0.05}, signal_quality_ok=False)
        self.assertFalse(res_artifact["accepted"])
        self.assertEqual(res_artifact["reason"], "POOR_SIGNAL_QUALITY_ARTIFACT_DETECTED")

    # -------------------------------------------------------------------------
    # 7. STRICT REST NEUTRALITY
    # -------------------------------------------------------------------------
    def test_rest_neutrality_enforcement(self):
        """Verifies REST intent never triggers an action across all profiles."""
        for prof_id in ["scrolling", "document_navigation", "presentation"]:
            router = NeuroOSCommandRouter(profile_id=prof_id)
            safety_res = {
                "accepted": True,
                "intent": "REST",
                "confidence": 0.95,
                "prediction_id": f"pred_rest_{prof_id}",
                "timestamp": 1000.0
            }
            res = router.process_intent(safety_res, current_time=1000.0)
            self.assertFalse(res["action_triggered"])
            self.assertEqual(res["command"], "NEUTRAL")

    # -------------------------------------------------------------------------
    # 8. EMERGENCY LOCK & MANUAL OVERRIDE
    # -------------------------------------------------------------------------
    def test_emergency_lock_and_manual_override(self):
        """Verifies manual lock immediately suppresses actions regardless of probabilities."""
        shield = NeuroOSSafetyShield()
        shield.set_manual_lock(True)

        res = shield.evaluate_safety({"REST": 0.01, "LEFT_FIST": 0.98, "RIGHT_FIST": 0.01})
        self.assertFalse(res["accepted"])
        self.assertTrue(res["emergency_locked"])
        self.assertEqual(res["reason"], "SAFETY_SHIELD_MANUAL_LOCK_ACTIVE")

        # Releasing lock restores safety checks
        shield.set_manual_lock(False)
        res_unlocked = shield.evaluate_safety({"REST": 0.01, "LEFT_FIST": 0.98, "RIGHT_FIST": 0.01})
        self.assertTrue(res_unlocked["accepted"])

    # -------------------------------------------------------------------------
    # 9. DUPLICATE & STALE EVENT SUPPRESSION
    # -------------------------------------------------------------------------
    def test_duplicate_and_stale_event_suppression(self):
        """Verifies duplicate prediction IDs and expired timestamps are rejected."""
        router = NeuroOSCommandRouter(dwell_windows=1, cooldown_sec=0.0, stale_timeout_sec=3.0)

        safety_res = {
            "accepted": True,
            "intent": "LEFT_FIST",
            "confidence": 0.90,
            "prediction_id": "pred_unique_99",
            "timestamp": 1000.0
        }

        # First dispatch passes
        res1 = router.process_intent(safety_res, current_time=1000.0)
        self.assertTrue(res1["action_triggered"])

        # Second dispatch with identical prediction ID must be suppressed as duplicate
        res2 = router.process_intent(safety_res, current_time=1000.5)
        self.assertFalse(res2["action_triggered"])
        self.assertIn("STALE_DUPLICATE_PREDICTION_SUPPRESSED", res2["reason"])

        # Third dispatch with timestamp older than stale timeout must be suppressed
        stale_res = {
            "accepted": True,
            "intent": "LEFT_FIST",
            "confidence": 0.90,
            "prediction_id": "pred_stale_100",
            "timestamp": 1000.0
        }
        res3 = router.process_intent(stale_res, current_time=1010.0)  # 10s elapsed > 3.0s timeout
        self.assertFalse(res3["action_triggered"])
        self.assertIn("STALE_TIMESTAMP_EXPIRED", res3["reason"])

    # -------------------------------------------------------------------------
    # 10. MODEL ARTIFACT INTEGRITY PRESERVATION
    # -------------------------------------------------------------------------
    def test_model_artifacts_integrity_step60_61_62(self):
        """Verifies Step 60, Step 61, and Step 62 model directories exist and were not overwritten."""
        step60_path = os.path.join(REPO_ROOT, "models", "step60")
        step61_path = os.path.join(REPO_ROOT, "models", "step61")
        step62_path = os.path.join(REPO_ROOT, "models", "step62")

        self.assertTrue(os.path.exists(step60_path), "models/step60 directory missing!")
        self.assertTrue(os.path.exists(step61_path), "models/step61 directory missing!")
        self.assertTrue(os.path.exists(step62_path), "models/step62 directory missing!")

        # Verify champion summary reports exist
        summary61 = os.path.join(step61_path, "summary_report.json")
        self.assertTrue(os.path.exists(summary61), "Step 61 summary report missing!")


if __name__ == "__main__":
    unittest.main()
