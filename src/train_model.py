import os
import json
import joblib
import datetime
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

FEATURE_COLUMNS = [
    "mean", "variance", "std", "rms", "min", "max", "peak_to_peak", "energy",
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"
]

EXPECTED_CLASSES = ["REST", "LEFT_FIST", "RIGHT_FIST"]


def load_and_validate_dataset(csv_path: str = "data/processed/eeg_features.csv") -> pd.DataFrame:
    """
    Loads eeg_features.csv and validates target labels and feature columns.

    Parameters:
        csv_path (str): Path to processed CSV file.

    Returns:
        pd.DataFrame: Validated DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Feature dataset not found at '{csv_path}'. Please run STEP 4 dataset build first.")

    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = ["subject", "run", "window_start", "window_end", "label"] + FEATURE_COLUMNS
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from {csv_path}")

    # Validate target labels
    actual_labels = set(df["label"].unique())
    unexpected = actual_labels - set(EXPECTED_CLASSES)
    if unexpected:
        raise ValueError(f"[STOP] Unexpected labels found in dataset: {unexpected}. Expected only {EXPECTED_CLASSES}.")

    return df


def subject_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
    """
    Splits dataset strictly by subject ID to prevent data leakage between windows.

    Parameters:
        df (pd.DataFrame): Dataset DataFrame.
        train_ratio (float): Fraction of subjects to allocate for training (~0.8).

    Returns:
        tuple: (df_train, df_test, train_subjects, test_subjects)
    """
    unique_subs = list(df["subject"].unique())
    if len(unique_subs) < 2:
        raise ValueError("At least 2 subjects are required for subject-based train/test splitting.")

    n_train = max(1, int(len(unique_subs) * train_ratio))
    train_subs = unique_subs[:n_train]
    test_subs = unique_subs[n_train:]

    df_train = df[df["subject"].isin(train_subs)].copy()
    df_test = df[df["subject"].isin(test_subs)].copy()

    # Ensure zero overlap
    assert len(set(train_subs).intersection(set(test_subs))) == 0, "[FAIL] Subject overlap detected between train and test sets!"

    return df_train, df_test, train_subs, test_subs


def train_and_evaluate_models(
    csv_path: str = "data/processed/eeg_features.csv",
    models_dir: str = "models"
) -> dict:
    """
    Trains Support Vector Machine (SVM) and Random Forest classifiers on training subjects,
    evaluates on completely unseen test subjects, serializes the best model, scaler, and metadata.

    Returns:
        dict: Complete evaluation results dictionary.
    """
    df = load_and_validate_dataset(csv_path)
    df_train, df_test, train_subs, test_subs = subject_train_test_split(df, train_ratio=0.8)

    X_train = df_train[FEATURE_COLUMNS].values
    y_train = df_train["label"].values

    X_test = df_test[FEATURE_COLUMNS].values
    y_test = df_test["label"].values

    # Preprocessing: StandardScaler fitted ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Support Vector Machine (SVM)
    svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)

    svm_acc = float(accuracy_score(y_test, y_pred_svm))
    svm_prec = float(precision_score(y_test, y_pred_svm, average="macro", zero_division=0))
    svm_rec = float(recall_score(y_test, y_pred_svm, average="macro", zero_division=0))
    svm_f1 = float(f1_score(y_test, y_pred_svm, average="macro", zero_division=0))
    svm_cm = confusion_matrix(y_test, y_pred_svm, labels=EXPECTED_CLASSES).tolist()
    svm_report = classification_report(y_test, y_pred_svm, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)

    # 2. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

    rf_acc = float(accuracy_score(y_test, y_pred_rf))
    rf_prec = float(precision_score(y_test, y_pred_rf, average="macro", zero_division=0))
    rf_rec = float(recall_score(y_test, y_pred_rf, average="macro", zero_division=0))
    rf_f1 = float(f1_score(y_test, y_pred_rf, average="macro", zero_division=0))
    rf_cm = confusion_matrix(y_test, y_pred_rf, labels=EXPECTED_CLASSES).tolist()
    rf_report = classification_report(y_test, y_pred_rf, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0)

    # Compare models based on test F1-score (or accuracy if tied)
    if rf_f1 > svm_f1:
        best_model_name = "Random Forest"
        best_model = rf
        best_metrics = {"accuracy": rf_acc, "precision": rf_prec, "recall": rf_rec, "f1": rf_f1}
    else:
        best_model_name = "Support Vector Machine (SVM)"
        best_model = svm
        best_metrics = {"accuracy": svm_acc, "precision": svm_prec, "recall": svm_rec, "f1": svm_f1}

    # Save models and metadata
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "eeg_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_name": best_model_name,
        "feature_names": FEATURE_COLUMNS,
        "classes": EXPECTED_CLASSES,
        "training_subjects": train_subs,
        "testing_subjects": test_subs,
        "training_date": datetime.datetime.now().isoformat(),
        "best_metrics": best_metrics,
        "svm_metrics": {"accuracy": svm_acc, "precision": svm_prec, "recall": svm_rec, "f1": svm_f1, "confusion_matrix": svm_cm, "report": svm_report},
        "rf_metrics": {"accuracy": rf_acc, "precision": rf_prec, "recall": rf_rec, "f1": rf_f1, "confusion_matrix": rf_cm, "report": rf_report},
        "dataset_info": {
            "total_windows": len(df),
            "train_windows": len(df_train),
            "test_windows": len(df_test)
        }
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def predict_eeg(feature_input, models_dir: str = "models") -> dict:
    """
    Inference function for real-time BCI prediction.

    Parameters:
        feature_input (list or np.ndarray or dict): 13 feature values.
        models_dir (str): Models directory.

    Returns:
        dict: Contains 'predicted_class', 'confidence', 'probabilities'.
    """
    model_path = os.path.join(models_dir, "eeg_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Trained model or scaler file missing in 'models/'. Please run training first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Format input array
    if isinstance(feature_input, dict):
        # Extract features in exact order
        vec = [feature_input.get(col, 0.0) for col in FEATURE_COLUMNS]
    elif isinstance(feature_input, (list, tuple)):
        vec = list(feature_input)
    elif isinstance(feature_input, np.ndarray):
        vec = feature_input.flatten().tolist()
    else:
        raise ValueError("feature_input must be dict, list, tuple, or numpy array")

    if len(vec) != len(FEATURE_COLUMNS):
        raise ValueError(f"Feature vector length must be exactly {len(FEATURE_COLUMNS)}, got {len(vec)}")

    vec_2d = np.array(vec, dtype=float).reshape(1, -1)
    vec_scaled = scaler.transform(vec_2d)

    predicted_class = str(model.predict(vec_scaled)[0])

    if hasattr(model, "predict_proba"):
        proba_vals = model.predict_proba(vec_scaled)[0]
        classes = list(model.classes_)
        probabilities = {cls: float(p) for cls, p in zip(classes, proba_vals)}
        confidence = float(np.max(proba_vals))
    else:
        probabilities = None
        confidence = None

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }


if __name__ == "__main__":
    print("Training EEG Movement Recognition models...")
    res = train_and_evaluate_models()
    print("Training complete! Metadata:")
    print(json.dumps(res, indent=2))
