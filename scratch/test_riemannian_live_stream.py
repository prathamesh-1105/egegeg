import os
import sys
sys.path.insert(0, ".")
import json
import joblib
import numpy as np

from src.step7_pipeline import discover_available_subjects
from src.step8_pipeline import find_subject_runs
from src.step9_pipeline import extract_raw_multichannel_windows
from src.step11_pipeline import apply_causal_temporal_smoothing

def test_riemannian_stream():
    subject_id = "S001"
    models_dir = "models/online_calibration"
    sub_model_dir = os.path.join(models_dir, subject_id)
    
    if not os.path.exists(sub_model_dir):
        print(f"Model dir {sub_model_dir} missing.")
        return

    cov_est = joblib.load(os.path.join(sub_model_dir, "cov.pkl"))
    ts_trans = joblib.load(os.path.join(sub_model_dir, "riemannian.pkl"))
    scaler = joblib.load(os.path.join(sub_model_dir, "scaler.pkl"))
    model = joblib.load(os.path.join(sub_model_dir, "model.pkl"))
    
    with open(os.path.join(sub_model_dir, "metadata.json"), "r") as f:
        meta = json.load(f)

    print("==================================================")
    print("RIEMANNIAN LIVE BCI STREAM DIAGNOSTIC TEST")
    print("==================================================")
    print(f"Subject: {subject_id}")
    print(f"Model Type: {type(model).__name__}")
    print(f"Model Classes: {model.classes_}")
    print(f"Calibration Source: {meta.get('calibration_source')}")
    
    found_runs = find_subject_runs(subject_id, data_dir="data")
    eval_run_file = found_runs.get("R11", None)
    if not eval_run_file:
        print("Run R11 not found for S001.")
        return
        
    X_eval, y_eval, infos = extract_raw_multichannel_windows(eval_run_file, channel_group="Group B", window_sec=3.0, overlap=0.5)
    print(f"Extracted {len(X_eval)} multi-channel windows (Shape: {X_eval.shape})")
    
    covs = cov_est.transform(X_eval)
    X_ts = ts_trans.transform(covs)
    X_sc = scaler.transform(X_ts)
    
    raw_preds = model.predict(X_sc)
    raw_probs = model.predict_proba(X_sc)
    
    raw_history = []
    smoothed_preds = []
    
    print("\n--- WINDOW-BY-WINDOW INFERENCE DEBUG LOG ---")
    for i in range(min(20, len(X_eval))):
        win_x = X_eval[i]  # Shape: (15, 480)
        h_val = hash(win_x.tobytes())
        min_v = np.min(win_x)
        max_v = np.max(win_x)
        mean_v = np.mean(win_x)
        std_v = np.std(win_x)
        
        raw_p = raw_preds[i]
        raw_history.append(raw_p)
        smooth_p = apply_causal_temporal_smoothing(raw_history, window_size=3)
        smoothed_preds.append(smooth_p)
        
        prob_dict = {cls: float(p) for cls, p in zip(model.classes_, raw_probs[i])}
        
        info_i = infos[i]
        print(f"Win {i:02d} [{info_i['window_start']:5.1f}s-{info_i['window_end']:5.1f}s] | Actual: {y_eval[i]:10s} | RawPred: {raw_p:10s} | SmoothPred: {smooth_p:10s} | Mean: {mean_v:6.2f} | Std: {std_v:6.2f} | Hash: {h_val}")
        print(f"    Probabilities -> REST: {prob_dict.get('REST',0):.3f}, LEFT: {prob_dict.get('LEFT_FIST',0):.3f}, RIGHT: {prob_dict.get('RIGHT_FIST',0):.3f}")

if __name__ == "__main__":
    test_riemannian_stream()
