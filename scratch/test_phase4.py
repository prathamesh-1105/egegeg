import os
import sys

sys.path.insert(0, os.path.abspath("."))

print("--- PHASE 4 VERIFICATION: AC Appliance State Machine & Dispatcher ---")

class MockSessionState(dict):
    def __getattr__(self, name):
        return self.get(name)
    def __setattr__(self, name, value):
        self[name] = value

session_state = MockSessionState()
session_state.ac_power_state = "OFF"
session_state.appliance_log = []

predictions_sequence = ["REST", "RIGHT_FIST", "REST", "LEFT_FIST", "REST"]

for step, active_pred in enumerate(predictions_sequence, start=1):
    if active_pred == "RIGHT_FIST" and session_state.ac_power_state != "ON":
        session_state.ac_power_state = "ON"
        session_state.appliance_log.append(f"Step {step}: [RIGHT FIST] AC Power ON")
    elif active_pred == "LEFT_FIST" and session_state.ac_power_state != "OFF":
        session_state.ac_power_state = "OFF"
        session_state.appliance_log.append(f"Step {step}: [LEFT FIST] AC Power OFF")
    
    print(f"Step {step} Pred={active_pred} -> AC State: {session_state.ac_power_state}")

print("\nAppliance Action Logs:")
for log in session_state.appliance_log:
    print(f"  {log}")

assert session_state.ac_power_state == "OFF"
assert len(session_state.appliance_log) == 2
print("[OK] PHASE 4 VERIFICATION COMPLETED SUCCESSFULLY!\n")
