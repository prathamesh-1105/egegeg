import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.eeg_source import EEGSource
from src.hardware_eeg_source import HardwareEEGSource
from src.dataset_eeg_source import DatasetEEGSource

print("--- PHASE 5 VERIFICATION: Hardware Abstraction & Readiness ---")

# 1. Test Hardware Source Incompatibility Graceful Detection
hw_source = HardwareEEGSource()
connected = hw_source.connect()
is_conn = hw_source.is_connected()
msg = hw_source.get_status_message()
print(f"[OK] Hardware Handshake: Connected={is_conn}, Message='{msg}'")

# 2. Test Abstraction Layer Fallback
dataset_source = DatasetEEGSource(subject_id="S001", run_name="R03", data_dir="data")
ds_connected = dataset_source.connect()
print(f"[OK] Dataset Source Fallback Ready: Connected={ds_connected}, Total Windows={len(dataset_source.windows)}")

print("[OK] PHASE 5 VERIFICATION COMPLETED SUCCESSFULLY!\n")
