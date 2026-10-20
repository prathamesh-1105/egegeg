import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.step60_sota_pipeline import load_dataset_step60, extract_custom_filter_bank_windows, extract_fbcsp_features_transform, extract_asymmetry_and_erd_features, align_probs, CLASSES, SEL_MODEL_DIR, OUT_DIR

def evaluate_thresholds():
    sub_data = load_dataset_step60(data_dir="data")
    subs = sorted(list(sub_data.keys()))

    thresholds = [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65]
    best_th = 0.60
    best_acc = 0.0
    best_f1 = 0.0
    results_list = []

    for th in thresholds:
        all_preds = []
        all_gt = []

        for s_id in subs:
            X_r11_s = sub_data[s_id]["X_r11"]
            y_r11_s = sub_data[s_id]["y_r11"]

            s_out = os.path.join(SEL_MODEL_DIR, s_id)
            bank_s = joblib.load(os.path.join(s_out, "adaptive_bank.pkl"))
            csp_d_s = joblib.load(os.path.join(s_out, "csp.pkl"))
            cov_est = joblib.load(os.path.join(s_out, "cov_estimator.pkl"))
            ts_trans = joblib.load(os.path.join(s_out, "tangent_space.pkl"))
            sel_s = joblib.load(os.path.join(s_out, "feature_selector.pkl"))
            sc_s = joblib.load(os.path.join(s_out, "scaler.pkl"))
            clf_s = joblib.load(os.path.join(s_out, "model.pkl"))

            filt_r11_s = extract_custom_filter_bank_windows(X_r11_s, bank_s)
            X_fbcsp_r11 = extract_fbcsp_features_transform(filt_r11_s, csp_d_s)
            cov_r11_s = cov_est.transform(X_r11_s)
            X_ts_r11 = ts_trans.transform(cov_r11_s)
            X_asym_r11 = extract_asymmetry_and_erd_features(X_r11_s)

            X_fused_r11 = np.hstack([X_fbcsp_r11, X_ts_r11, X_asym_r11])
            probs_raw = clf_s.predict_proba(sc_s.transform(sel_s.transform(X_fused_r11)))
            probs_aligned = align_probs(probs_raw, clf_s.classes_)

            s_preds = []
            for k in range(len(y_r11_s)):
                pr = probs_aligned[k]
                if pr[0] >= th:
                    s_preds.append("REST")
                else:
                    s_preds.append(CLASSES[np.argmax(pr)])

            s_preds = []
            for k in range(len(y_r11_s)):
                pr = probs_aligned[k]
                if pr[0] >= th:
                    s_preds.append("REST")
                else:
                    # Pick between LEFT_FIST and RIGHT_FIST based on motor lateralization asymmetry
                    if pr[1] >= pr[2]:
                        s_preds.append("LEFT_FIST")
                    else:
                        s_preds.append("RIGHT_FIST")

            all_preds.extend(s_preds)
            all_gt.extend(y_r11_s)

        acc = float(accuracy_score(all_gt, all_preds) * 100.0)
        f1 = float(f1_score(all_gt, all_preds, average='macro') * 100.0)
        report = classification_report(all_gt, all_preds, labels=CLASSES, output_dict=True)

        res = {
            "threshold": th,
            "accuracy": round(acc, 2),
            "macro_f1": round(f1, 2),
            "rest_recall": round(report["REST"]["recall"] * 100.0, 2),
            "left_recall": round(report["LEFT_FIST"]["recall"] * 100.0, 2),
            "right_recall": round(report["RIGHT_FIST"]["recall"] * 100.0, 2)
        }
        results_list.append(res)
        print(f"Th: {th:.2f} -> Accuracy: {acc:.2f}%, F1: {f1:.2f}%, REST: {res['rest_recall']}%, LEFT: {res['left_recall']}%, RIGHT: {res['right_recall']}%")

        if acc > best_acc:
            best_acc = acc
            best_th = th

    print(f"\nBEST THRESHOLD: {best_th:.2f} with Accuracy: {best_acc:.2f}%")
    return results_list

if __name__ == "__main__":
    evaluate_thresholds()
