import os
import sys
sys.path.insert(0, ".")
import json
import joblib
import numpy as np
import pandas as pd

from src.load_eeg import load_eeg_raw
from src.events import get_eeg_events
from src.bci_stream import RealEEGStreamer
from src.feature_extraction import compute_feature_vector
from src.train_model import predict_eeg, FEATURE_COLUMNS, EXPECTED_CLASSES
from src.eeg_source import DatasetEEGSource
from src.step9_pipeline import extract_raw_multichannel_windows

print("==================================================")
print("1. INSPECTING STEP 5 / DEFAULT MODEL (models/eeg_model.pkl)")
print("==================================================")

model_path = "models/eeg_model.pkl"
scaler_path = "models/scaler.pkl"
meta_path = "models/model_metadata.json"

if os.path.exists(model_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model loaded:", type(model))
    print("Model classes_:", getattr(model, "classes_", "N/A"))
    if hasattr(model, "n_features_in_"):
        print("Feature count:", model.n_features_in_)

if os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    print("Metadata classes:", meta.get("classes"))
    print("Metadata best_metrics:", meta.get("best_metrics"))

print("\n==================================================")
print("2. TESTING RealEEGStreamer WINDOWS & STEP 5 MODEL PREDICTIONS")
print("==================================================")

streamer = RealEEGStreamer(edf_path="data/S001R03.edf", window_sec=2.0, overlap=0.5)
total_wins = streamer.get_total_windows()
print(f"Total windows extracted by RealEEGStreamer: {total_wins}")

stream_preds = []
for idx in range(min(15, total_wins)):
    win = streamer.get_window(idx)
    sig = win["signal_uv"]
    h_val = hash(sig.tobytes())
    mean_val = np.mean(sig)
    std_val = np.std(sig)
    min_val = np.min(sig)
    max_val = np.max(sig)
    
    feats = compute_feature_vector(sig, streamer.sfreq)
    pred_res = predict_eeg(feats)
    
    print(f"Win {idx:02d} [{win['window_start']:5.1f}s-{win['window_end']:5.1f}s] | Label: {win['actual_label']:10s} | Pred: {pred_res['predicted_class']:10s} | Conf: {pred_res['confidence']:.3f} | Mean: {mean_val:7.2f} | Std: {std_val:6.2f} | Hash: {h_val}")
    if pred_res['probabilities']:
        print(f"    Probs -> REST: {pred_res['probabilities'].get('REST',0):.3f}, LEFT_FIST: {pred_res['probabilities'].get('LEFT_FIST',0):.3f}, RIGHT_FIST: {pred_res['probabilities'].get('RIGHT_FIST',0):.3f}")

print("\n==================================================")
print("3. INSPECTING STEP 11 RIEMANNIAN ONLINE CALIBRATION MODEL")
print("==================================================")

online_dir = "models/online_calibration/S001"
if os.path.exists(online_dir):
    cov_p = os.path.join(online_dir, "cov.pkl")
    riem_p = os.path.join(online_dir, "riemannian.pkl")
    sc_p = os.path.join(online_dir, "scaler.pkl")
    mod_p = os.path.join(online_dir, "model.pkl")
    meta_p = os.path.join(online_dir, "metadata.json")
    
    cov_est = joblib.load(cov_p)
    ts_trans = joblib.load(riem_p)
    scaler_online = joblib.load(sc_p)
    model_online = joblib.load(mod_p)
    with open(meta_p, "r") as f:
        meta_online = json.load(f)
        
    print("Online Model loaded:", type(model_online))
    print("Online Model classes_:", getattr(model_online, "classes_", "N/A"))
    print("Metadata sample_predictions count:", len(meta_online.get("sample_predictions", [])))
    
    # Extract multi-channel windows for S001 R11 holdout
    edf_r11 = "data/S001R11.edf"
    if not os.path.exists(edf_r11):
        edf_r11 = "data/raw/S001R11.edf"
        
    if os.path.exists(edf_r11):
        X_r11, y_r11, info_r11 = extract_raw_multichannel_windows(edf_r11, channel_group="Group B", window_sec=3.0, overlap=0.5)
        print(f"S001 R11 Multi-channel windows extracted: shape={X_r11.shape}, labels count={len(y_r11)}")
        
        covs_eval = cov_est.transform(X_r11)
        X_eval_ts = ts_trans.transform(covs_eval)
        X_eval_sc = scaler_online.transform(X_eval_ts)
        y_eval_pred = model_online.predict(X_eval_sc)
        
        if hasattr(model_online, "predict_proba"):
            y_eval_prob = model_online.predict_proba(X_eval_sc)
        else:
            y_eval_prob = None
            
        print("\nTesting Step 11 Multi-Channel Riemannian Model on S001 R11:")
        for idx in range(min(15, len(y_r11))):
            act_l = y_r11[idx]
            pred_l = y_eval_pred[idx]
            probs_str = ""
            if y_eval_prob is not None:
                probs_dict = {cls: float(p) for cls, p in zip(model_online.classes_, y_eval_prob[idx])}
                probs_str = f" | Probs -> REST: {probs_dict.get('REST',0):.3f}, LEFT: {probs_dict.get('LEFT_FIST',0):.3f}, RIGHT: {probs_dict.get('RIGHT_FIST',0):.3f}"
            print(f"Win {idx:02d} [{info_r11[idx]['window_start']:5.1f}s-{info_r11[idx]['window_end']:5.1f}s] | Actual: {act_l:10s} | Pred: {pred_l:10s}{probs_str}")
