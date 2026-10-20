import glob
import os

edf_files = glob.glob("data/**/*.edf", recursive=True)
print(f"TOTAL_EDF_COUNT={len(edf_files)}")

# Break down by location
raw_edfs = glob.glob("data/raw/*.edf")
print(f"RAW_DIR_EDF_COUNT={len(raw_edfs)}")

mne_edfs = glob.glob("data/raw/MNE-eegbci-data/**/*.edf", recursive=True)
print(f"MNE_DIR_EDF_COUNT={len(mne_edfs)}")

root_edfs = glob.glob("data/*.edf")
print(f"ROOT_DATA_EDF_COUNT={len(root_edfs)}")

for f in sorted(edf_files):
    print(f)
