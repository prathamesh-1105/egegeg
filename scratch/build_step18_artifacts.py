import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

sys.path.insert(0, os.path.abspath("."))

from src.step7_pipeline import discover_available_subjects, EXPECTED_CLASSES
from src.step8_pipeline import find_subject_runs
from src.step14_pipeline import extract_event_aligned_windows
from src.step17b_pipeline import extract_fbcsp_features, FILTER_BANKS

def build_step18_artifacts():
    print("Building Step 18 Integration Artifacts...")
    src_step17b_dir = "models/step17b/selected_model"
    target_step18_dir = "models/step18/selected_model"
    os.makedirs(target_step18_dir, exist_ok=True)

    subjects = discover_available_subjects("data")
    if not subjects:
        subjects = [f"S{i:03d}" for i in range(1, 11)]

    all_actuals, all_preds = [], []

    for sub_id in subjects:
        sub_src_dir = os.path.join(src_step17b_dir, sub_id)
        sub_target_dir = os.path.join(target_step18_dir, sub_id)
        os.makedirs(sub_target_dir, exist_ok=True)

        model = joblib.load(os.path.join(sub_src_dir, "model.pkl"))
        scaler = joblib.load(os.path.join(sub_src_dir, "scaler.pkl"))
        csp_dict = joblib.load(os.path.join(sub_src_dir, "csp.pkl"))

        joblib.dump(model, os.path.join(sub_target_dir, "model.pkl"))
        joblib.dump(scaler, os.path.join(sub_target_dir, "scaler.pkl"))
        joblib.dump(csp_dict, os.path.join(sub_target_dir, "csp.pkl"))

        with open(os.path.join(sub_src_dir, "metadata.json"), "r") as f:
            meta = json.load(f)

        found_runs = find_subject_runs(sub_id, data_dir="data")
        r03_path = found_runs.get("R03")
        r07_path = found_runs.get("R07")
        r11_path = found_runs.get("R11")

        if not r03_path or not r07_path or not r11_path:
            avail = sorted(list(found_runs.keys()))
            if len(avail) >= 3:
                r03_path = found_runs[avail[0]]
                r07_path = found_runs[avail[1]]
                r11_path = found_runs[avail[-1]]

        X_r03, y_r03, _ = extract_event_aligned_windows(r03_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
        X_r07, y_r07, _ = extract_event_aligned_windows(r07_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
        X_r11, y_r11, _ = extract_event_aligned_windows(r11_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)

        X_calib = np.concatenate([X_r03, X_r07], axis=0)
        y_calib = np.concatenate([y_r03, y_r07], axis=0)

        # Extract FBCSP features on calibration data using saved CSP transformers
        X_calib_fbcsp, _ = extract_fbcsp_features(X_calib, csp_dict=csp_dict, is_train=False)

        # Fit SelectKBest (k=16) strictly on calibration data
        k_best = min(16, X_calib_fbcsp.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k_best)
        selector.fit(X_calib_fbcsp, y_calib)

        joblib.dump(selector, os.path.join(sub_target_dir, "selector.pkl"))

        # Verify on R11 holdout
        X_r11_fbcsp, _ = extract_fbcsp_features(X_r11, csp_dict=csp_dict, is_train=False)
        X_r11_fbcsp_sel = selector.transform(X_r11_fbcsp)
        X_r11_sc = scaler.transform(X_r11_fbcsp_sel)
        preds_r11 = model.predict(X_r11_sc)

        all_actuals.extend(y_r11)
        all_preds.extend(preds_r11)

        acc_r11 = float(np.mean(preds_r11 == y_r11) * 100.0)
        meta["step18_verified"] = True
        meta["r11_holdout_accuracy"] = round(acc_r11, 2)

        with open(os.path.join(sub_target_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[{sub_id}] Selector saved. R11 Holdout Accuracy: {acc_r11:.2f}%")

    overall_acc = float(np.mean(np.array(all_preds) == np.array(all_actuals)) * 100.0)
    print("==================================================")
    print(f"Overall Aggregated R11 Holdout Accuracy: {overall_acc:.2f}%")
    print("Step 18 Integration Artifacts Build Complete.")
    print("==================================================")

if __name__ == "__main__":
    build_step18_artifacts()
