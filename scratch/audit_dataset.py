import os
import glob
import numpy as np
import mne
mne.set_log_level('ERROR')

from src.step8_pipeline import find_subject_runs
from src.step14_pipeline import extract_event_aligned_windows

edf_files_all = glob.glob("data/**/*.edf", recursive=True)
print(f"Total EDF files found in data/: {len(edf_files_all)}")

subs = [f"S{i:03d}" for i in range(1, 11)]
print(f"Subjects checked: {subs}")

runs_found = set()
for f in edf_files_all: AZx
    base = os.path.basename(f)
    for r in ["R03", "R07", "R11"]:
        if r in base:
            runs_found.add(r)
print(f"Runs found in filenames: {sorted(list(runs_found))}")

# Check trial count and details across S001-S010 for R03, R07, R11
total_trials = 0
r03_r07_trials = 0
r11_trials = 0

class_counts = {"REST": 0, "LEFT_FIST": 0, "RIGHT_FIST": 0}
r11_class_counts = {"REST": 0, "LEFT_FIST": 0, "RIGHT_FIST": 0}

for sub_id in subs:
    found_runs = find_subject_runs(sub_id, data_dir="data")
    r03_path = found_runs.get("R03")
    r07_path = found_runs.get("R07")
    r11_path = found_runs.get("R11")
    
    X_3, y_3, _ = extract_event_aligned_windows(r03_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
    X_7, y_7, _ = extract_event_aligned_windows(r07_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
    X_11, y_11, _ = extract_event_aligned_windows(r11_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)

    r03_r07_trials += len(y_3) + len(y_7)
    r11_trials += len(y_11)

    for y in list(y_3) + list(y_7) + list(y_11):
        class_counts[y] = class_counts.get(y, 0) + 1

    for y in y_11:
        r11_class_counts[y] = r11_class_counts.get(y, 0) + 1

total_trials = r03_r07_trials + r11_trials
print(f"Total Trials across S001-S010 (R03, R07, R11): {total_trials}")
print(f"R03 + R07 Trials (Calibration/Dev): {r03_r07_trials}")
print(f"R11 Trials (Testing): {r11_trials}")
print(f"Class distribution across all runs: {class_counts}")
print(f"Class distribution in R11: {r11_class_counts}")
