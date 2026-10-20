import os
import json
import joblib


class ModelLoader:
    """
    Model Artifact Loader.
    Loads frozen Riemannian BCI models (Step 14 champion, Step 13B baseline).
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir

    def load_step14_model(self, subject_id: str = "S001") -> dict:
        """
        Load Step 14 Event-Aligned Riemannian model artifacts.
        """
        step14_dir = os.path.join(self.models_dir, "step14", "selected_model", subject_id)
        if not os.path.exists(step14_dir):
            step14_dir = os.path.join(self.models_dir, "step14", "selected_model", "S001")

        cov_p = os.path.join(step14_dir, "cov.pkl")
        ts_p = os.path.join(step14_dir, "riemannian.pkl")
        sc_p = os.path.join(step14_dir, "scaler.pkl")
        m_p = os.path.join(step14_dir, "model.pkl")
        meta_p = os.path.join(step14_dir, "metadata.json")

        if not (os.path.exists(cov_p) and os.path.exists(ts_p) and os.path.exists(sc_p) and os.path.exists(m_p)):
            # Fallback to Step 13B model if Step 14 model artifact not found
            step13b_dir = os.path.join(self.models_dir, "step13b", "selected_model", subject_id)
            if not os.path.exists(step13b_dir):
                step13b_dir = os.path.join(self.models_dir, "step13b", "selected_model", "S001")
            cov_p = os.path.join(step13b_dir, "cov.pkl")
            ts_p = os.path.join(step13b_dir, "riemannian.pkl")
            sc_p = os.path.join(step13b_dir, "scaler.pkl")
            m_p = os.path.join(step13b_dir, "model.pkl")
            meta_p = os.path.join(step13b_dir, "metadata.json")

        cov_est = joblib.load(cov_p)
        ts_trans = joblib.load(ts_p)
        scaler = joblib.load(sc_p)
        model = joblib.load(m_p)

        meta = {}
        if os.path.exists(meta_p):
            with open(meta_p, "r") as f:
                meta = json.load(f)

        return {
            "cov_estimator": cov_est,
            "tangent_space": ts_trans,
            "scaler": scaler,
            "model": model,
            "metadata": meta,
            "model_name": "STEP 14 Event-Aligned Riemannian"
        }

    def load_step60_model(self) -> dict:
        """
        Load Step 60 SOTA Riemannian Spatial-Temporal Cascade Engine artifact.
        """
        step60_dir = os.path.join(self.models_dir, "step60")
        model_p = os.path.join(step60_dir, "selected_model", "step60_pipeline.joblib")
        meta_p = os.path.join(step60_dir, "summary_report.json")

        if not os.path.exists(model_p):
            raise FileNotFoundError(f"Step 60 model pipeline artifact not found at {model_p}")

        pipeline = joblib.load(model_p)
        meta = {}
        if os.path.exists(meta_p):
            with open(meta_p, "r") as f:
                meta = json.load(f)

        return {
            "pipeline": pipeline,
            "metadata": meta,
            "model_name": "Step 60 SOTA Riemannian Spatial-Temporal Engine"
        }

