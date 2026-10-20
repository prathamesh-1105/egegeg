import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.step17b_pipeline import extract_fbcsp_features


class PersonalFBCSPTrainer:
    """
    Personalized FBCSP Model Trainer for Subject-Specific Hardware Calibration.
    Trains a subject-specific model using recorded EEG calibration windows with chronological train/validation splitting.
    """

    def __init__(self, output_dir: str = "models/step21/personal_model"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model_info = None

    def train_personal_model(self, X_epochs: np.ndarray, y_labels: np.ndarray, sfreq: float = 160.0) -> dict:
        """
        Trains subject-specific FBCSP model on recorded calibration epochs.
        X_epochs shape: (n_epochs, 15, n_samples)
        y_labels shape: (n_epochs,) with classes e.g. ['REST', 'LEFT_FIST', 'RIGHT_FIST']

        Uses chronological train/val split (first 70% train, last 30% val) to prevent temporal leakage.
        """
        n_epochs = len(X_epochs)
        if n_epochs < 6:
            return {
                "status": "ERROR",
                "message": f"Insufficient calibration windows ({n_epochs} windows available; minimum 6 required)."
            }

        # Chronological Split (70% train, 30% val)
        split_idx = int(n_epochs * 0.7)
        if split_idx < 3 or (n_epochs - split_idx) < 3:
            split_idx = int(n_epochs * 0.5)

        X_train, X_val = X_epochs[:split_idx], X_epochs[split_idx:]
        y_train, y_val = y_labels[:split_idx], y_labels[split_idx:]

        classes_train = np.unique(y_train)
        classes_val = np.unique(y_val)

        if len(classes_train) < 2 or len(classes_val) < 2:
            # Fallback to stratified K-Fold indices if chronological split produces single class in val set
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=3, shuffle=False)
            train_idx, val_idx = next(skf.split(X_epochs, y_labels))
            X_train, X_val = X_epochs[train_idx], X_epochs[val_idx]
            y_train, y_val = y_labels[train_idx], y_labels[val_idx]

        # Fit FBCSP Feature Extractor
        X_train_fbcsp, csp_dict = extract_fbcsp_features(X_train, y_labels=y_train, sfreq=sfreq, is_train=True)
        X_val_fbcsp, _ = extract_fbcsp_features(X_val, csp_dict=csp_dict, sfreq=sfreq, is_train=False)

        # Feature Selection (SelectKBest k=min(16, n_features))
        k_best = min(16, X_train_fbcsp.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k_best)
        X_train_sel = selector.fit_transform(X_train_fbcsp, y_train)
        X_val_sel = selector.transform(X_val_fbcsp)

        # Scaler
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_sel)
        X_val_sc = scaler.transform(X_val_sel)

        # Class-Balanced Logistic Regression
        clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        clf.fit(X_train_sc, y_train)

        # Evaluate on Validation Set
        y_pred = clf.predict(X_val_sc)

        acc = float(accuracy_score(y_val, y_pred))
        macro_f1 = float(f1_score(y_val, y_pred, average='macro', zero_division=0))

        # Per-class recall
        unique_cls = list(clf.classes_)
        recall_dict = {}
        for cls in unique_cls:
            mask = (y_val == cls)
            if np.sum(mask) > 0:
                rec = float(np.mean(y_pred[mask] == cls))
            else:
                rec = 0.0
            recall_dict[str(cls)] = round(rec * 100.0, 2)

        # Save Personal Model Artifacts
        joblib.dump(clf, os.path.join(self.output_dir, "model.pkl"))
        joblib.dump(csp_dict, os.path.join(self.output_dir, "csp.pkl"))
        joblib.dump(selector, os.path.join(self.output_dir, "selector.pkl"))
        joblib.dump(scaler, os.path.join(self.output_dir, "scaler.pkl"))

        metrics_report = {
            "status": "TRAINED",
            "total_windows": n_epochs,
            "train_windows": len(X_train),
            "validation_windows": len(X_val),
            "accuracy": round(acc * 100.0, 2),
            "macro_f1": round(macro_f1 * 100.0, 2),
            "per_class_recall": recall_dict,
            "classes": [str(c) for c in unique_cls],
            "model_dir": self.output_dir
        }

        with open(os.path.join(self.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_report, f, indent=2)

        self.model_info = metrics_report
        return metrics_report
