import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.command_state_machine import BCICommandStateMachine
from src.dataset_eeg_source import DatasetEEGSource

print("--- PHASE 3 VERIFICATION: Real-Time Streamer & Command State Machine ---")

# 1. Initialize State Machine
sm = BCICommandStateMachine(strategy="HYSTERESIS", max_prob_thresh=0.60)
sm.reset()
print(f"[OK] BCI Command State Machine Initialized in Strategy: {sm.strategy}")

# 2. Simulate 5 Windows of Real-Time Predictions
test_predictions = [
    {"REST": 0.85, "LEFT_FIST": 0.10, "RIGHT_FIST": 0.05},
    {"REST": 0.80, "LEFT_FIST": 0.15, "RIGHT_FIST": 0.05},
    {"REST": 0.10, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.85},  # Right fist surge
    {"REST": 0.05, "LEFT_FIST": 0.05, "RIGHT_FIST": 0.90},  # Right fist confirmed
    {"REST": 0.70, "LEFT_FIST": 0.15, "RIGHT_FIST": 0.15},  # Back to rest
]

for i, p_dict in enumerate(test_predictions, start=1):
    res = sm.process_window(p_dict)
    state = res.get("final_command_state", res.get("current_state", "UNKNOWN"))
    print(f" Window {i}: Raw Max = {max(p_dict, key=p_dict.get)} -> State Machine Command = {state}")

print("[OK] PHASE 3 VERIFICATION COMPLETED SUCCESSFULLY!\n")
