import os
import urllib.request
import time

def download_physionet_dataset(subjects: list = None, runs: list = None, output_dir: str = "data/raw"):
    if subjects is None:
        subjects = [f"S{i:03d}" for i in range(1, 11)]
    if runs is None:
        runs = ["R03", "R07", "R11"]

    os.makedirs(output_dir, exist_ok=True)
    base_url = "https://physionet.org/files/eegmmidb/1.0.0"

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)

    downloaded = 0
    total = len(subjects) * len(runs)

    for sub in subjects:
        for run in runs:
            filename = f"{sub}{run}.edf"
            filepath = os.path.join(output_dir, filename)

            if not os.path.exists(filepath) or os.path.getsize(filepath) < 100000:
                file_url = f"{base_url}/{sub}/{filename}"
                print(f"[{downloaded+1}/{total}] Downloading {filename} from {file_url}...")
                try:
                    urllib.request.urlretrieve(file_url, filepath)
                    print(f"  -> Saved {filename} ({os.path.getsize(filepath)} bytes)")
                    time.sleep(0.1)
                except Exception as e:
                    print(f"  -> Failed {filename}: {e}")
            else:
                print(f"[{downloaded+1}/{total}] Already exists: {filename}")
            
            downloaded += 1

    print(f"\nFinished! Files in {output_dir}: {len(os.listdir(output_dir))}")

if __name__ == "__main__":
    download_physionet_dataset()
