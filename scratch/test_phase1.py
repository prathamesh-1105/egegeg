import os
import sys
import numpy as np
import mne

sys.path.insert(0, os.path.abspath("."))

from src.load_eeg import load_eeg_raw, get_eeg_info
from src.events import get_eeg_events

print("--- PHASE 1 VERIFICATION: Data Pipeline & Preprocessing ---")

edf_file = "data/S001R03.edf"
if not os.path.exists(edf_file):
    print(f"❌ EDF file not found: {edf_file}")
    sys.exit(1)

# 1. Load Raw EDF
raw = load_eeg_raw(edf_file)
info = get_eeg_info(raw, edf_file)
print(f"✓ EDF Loaded: {info['n_channels']} channels, Sampling Rate: {info['sfreq']} Hz, Duration: {info['duration_sec']} sec")

# 2. Extract Events & Map Classes
df_events = get_eeg_events(raw)
print(f"✓ Total Events Extracted: {len(df_events)}")
print(f"Event Distribution:\n{df_events['event_name'].value_counts()}")

# 3. Test Bandpass Filtering (8-30 Hz)
raw_filtered = raw.copy().filter(l_freq=8.0, h_freq=30.0, fir_design='firwin', verbose=False)
print("✓ Bandpass Filter Applied (8 - 30 Hz)")

print("✅ PHASE 1 VERIFICATION COMPLETED SUCCESSFULLY!\n")
