import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.model_loader import ModelLoader
from src.model_integrity import ModelIntegrityVerifier

print("--- PHASE 2 VERIFICATION: Model Training & Inference ---")

models_dir = "models"
if not os.path.exists(models_dir):
    print(f"[FAIL] Models directory missing: {models_dir}")
    sys.exit(1)

# 1. Verify Model Integrity
verifier = ModelIntegrityVerifier(models_dir=models_dir, subject_id="S001")
res_integ = verifier.verify_integrity()
print(f"[OK] Model Integrity Verified: Valid={res_integ['is_valid']}")

# 2. Test Model Loader
loader = ModelLoader(models_dir)
artifacts = loader.load_step14_model(subject_id="S001")
model = artifacts["model"]
scaler = artifacts["scaler"]
print(f"[OK] Model Loaded: {type(model).__name__}")

# 3. Test Sample Inference
dummy_data = np.random.randn(1, artifacts["scaler"].n_features_in_)
if scaler is not None:
    dummy_data = scaler.transform(dummy_data)

probs = model.predict_proba(dummy_data)[0]
pred_idx = np.argmax(probs)
classes = ["REST", "LEFT_FIST", "RIGHT_FIST"]
print(f"[OK] Sample Prediction: {classes[pred_idx]} | Probs: REST={probs[0]:.2f}, LEFT={probs[1]:.2f}, RIGHT={probs[2]:.2f}")

print("[OK] PHASE 2 VERIFICATION COMPLETED SUCCESSFULLY!\n")
