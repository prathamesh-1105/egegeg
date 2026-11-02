import os
import sys
import shutil
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_S3_URL = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0"
RAW_DIR = os.path.abspath("data/raw")
MNE_DIR = os.path.abspath("data/raw/MNE-eegbci-data/files/eegmmidb/1.0.0")

def sync_existing_mne_files():
    """Copies R04, R08, R12 from MNE nested folder into data/raw."""
    copied = 0
    if os.path.exists(MNE_DIR):
        for sub in os.listdir(MNE_DIR):
            sub_path = os.path.join(MNE_DIR, sub)
            if os.path.isdir(sub_path):
                for f in os.listdir(sub_path):
                    if f.endswith(".edf"):
                        src = os.path.join(sub_path, f)
                        dst = os.path.join(RAW_DIR, f)
                        if not os.path.exists(dst) or os.path.getsize(dst) < 100000:
                            shutil.copy2(src, dst)
                            copied += 1
    print(f"[Sync] Copied {copied} files from MNE nested storage to {RAW_DIR}")

def download_file(sub: str, run: str) -> tuple:
    filename = f"{sub}{run}.edf"
    target_path = os.path.join(RAW_DIR, filename)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 100000:
        return (filename, True, "already exists", os.path.getsize(target_path))

    url = f"{BASE_S3_URL}/{sub}/{filename}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response, open(target_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        size = os.path.getsize(target_path)
        return (filename, True, "downloaded", size)
    except Exception as e:
        if os.path.exists(target_path):
            os.remove(target_path)
        return (filename, False, str(e), 0)

def expand_dataset(subjects: list, runs: list, max_workers: int = 5):
    os.makedirs(RAW_DIR, exist_ok=True)
    sync_existing_mne_files()

    tasks = []
    for sub in subjects:
        for run in runs:
            tasks.append((sub, run))

    print(f"\n[Dataset Expansion] Checking & downloading {len(tasks)} EDF recordings across {len(subjects)} subjects...")
    t0 = time.time()
    success_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(download_file, s, r): (s, r) for s, r in tasks}
        for future in as_completed(future_to_file):
            filename, success, status, size = future.result()
            if success:
                success_count += 1
                if status == "downloaded":
                    print(f"  [OK] {filename} downloaded ({size / (1024*1024):.2f} MB)", flush=True)
            else:
                print(f"  [FAIL] {filename} failed: {status}", flush=True)

    t_total = time.time() - t0
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".edf")]
    print(f"\n[Completed] {success_count}/{len(tasks)} valid files ready in {t_total:.1f}s.")
    print(f"[Total Raw EDF Recordings in {RAW_DIR}]: {len(raw_files)}")

if __name__ == "__main__":
    # Subjects S001 through S020 (double cohort)
    all_subjects = [f"S{i:03d}" for i in range(1, 21)]
    # All primary motor execution and motor imagery runs
    target_runs = ["R03", "R04", "R07", "R08", "R11", "R12"]
    expand_dataset(all_subjects, target_runs, max_workers=6)
