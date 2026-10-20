import os
import sys
sys.path.insert(0, os.path.abspath("."))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.step8_pipeline import find_subject_runs
from src.step14_pipeline import extract_event_aligned_windows
from src.step27_pipeline import FILTER_BANK_CANDIDATES, prefilter_window_bank, extract_fbcsp_features_transform
from src.step33_pipeline import extract_contralateral_asymmetry_features, extract_welch_psd_features

CLASSES = ["REST", "LEFT_FIST", "RIGHT_FIST"]
filter_bank = FILTER_BANK_CANDIDATES["Bank_A"]
data_dir = "data"
step24_dir = "models/step24"

def align_probs(probs: np.ndarray, classes_from_clf) -> np.ndarray:
    classes_from_clf = list(classes_from_clf)
    aligned = np.zeros((len(probs), 3))
    for c_i, c_name in enumerate(CLASSES):
        if c_name in classes_from_clf:
            orig_i = classes_from_clf.index(c_name)
            aligned[:, c_i] = probs[:, orig_i]
    return aligned

all_subs = [f"S{i:03d}" for i in range(1, 11)]

y_true_all = []
y_pred_s24 = []
y_pred_s33 = []

for sub_id in all_subs:
    found_runs = find_subject_runs(sub_id, data_dir=data_dir)
    r03_path = found_runs.get("R03", list(found_runs.values())[0])
    r07_path = found_runs.get("R07", list(found_runs.values())[1])
    r11_path = found_runs.get("R11", list(found_runs.values())[-1])

    X_r03, y_r03, _ = extract_event_aligned_windows(r03_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
    X_r07, y_r07, _ = extract_event_aligned_windows(r07_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)
    X_r11, y_r11, _ = extract_event_aligned_windows(r11_path, channel_group="Group B", window_sec=3.0, onset_offset_sec=0.5)

    X_cal = np.concatenate([X_r03, X_r07], axis=0)
    y_cal = np.concatenate([y_r03, y_r07], axis=0)

    s_model_dir = os.path.join(step24_dir, "selected_model", sub_id)
    csp_dict = joblib.load(os.path.join(s_model_dir, "csp.pkl"))
    sel = joblib.load(os.path.join(s_model_dir, "feature_selector.pkl"))
    sc = joblib.load(os.path.join(s_model_dir, "scaler.pkl"))
    clf = joblib.load(os.path.join(s_model_dir, "model.pkl"))

    filt_r11 = prefilter_window_bank(X_r11, filter_bank)
    X_r11_fbcsp = extract_fbcsp_features_transform(filt_r11, csp_dict)
    X_r11_sc = sc.transform(sel.transform(X_r11_fbcsp))

    p_base_raw = clf.predict_proba(X_r11_sc)
    p_base = align_probs(p_base_raw, clf.classes_)
    pred_base = [CLASSES[i] for i in np.argmax(p_base, axis=1)]

    # Fit Asymmetry submodel on calibration data
    f_cal_asym = extract_contralateral_asymmetry_features(X_cal)
    sc_b = StandardScaler().fit(f_cal_asym)
    clf_b = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced').fit(sc_b.transform(f_cal_asym), y_cal)
    p_b_raw = clf_b.predict_proba(sc_b.transform(extract_contralateral_asymmetry_features(X_r11)))
    p_b = align_probs(p_b_raw, clf_b.classes_)

    # Fit PSD submodel on calibration data
    f_cal_psd = extract_welch_psd_features(X_cal)
    sc_c = StandardScaler().fit(f_cal_psd)
    clf_c = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced').fit(sc_c.transform(f_cal_psd), y_cal)
    p_c_raw = clf_c.predict_proba(sc_c.transform(extract_welch_psd_features(X_r11)))
    p_c = align_probs(p_c_raw, clf_c.classes_)

    p_fused = 0.6 * p_base + 0.2 * p_b + 0.2 * p_c

    cand_preds = []
    for k in range(len(y_r11)):
        pa = p_base[k]
        pf = p_fused[k]
        if pa[0] > 0.65: # Confident REST (REST index is 0 in aligned probs!)
            cand_preds.append("REST")
        else:
            cand_preds.append(CLASSES[np.argmax(pf)])

    y_true_all.extend(y_r11)
    y_pred_s24.extend(pred_base)
    y_pred_s33.extend(cand_preds)

print("Step 24 Accuracy:", round(accuracy_score(y_true_all, y_pred_s24) * 100.0, 2), "% Macro F1:", round(f1_score(y_true_all, y_pred_s24, average="macro") * 100.0, 2), "%")
print("Step 33 Accuracy:", round(accuracy_score(y_true_all, y_pred_s33) * 100.0, 2), "% Macro F1:", round(f1_score(y_true_all, y_pred_s33, average="macro") * 100.0, 2), "%")
