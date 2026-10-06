"""Backend configuration — paths and MLflow tracking URI, reusing the same
tracking store the pipelines already write to (see `pipeline/mlflow_setup.py`)
so the API always sees the champion models the pipelines/retraining scripts
promote."""
from inference.config import ROOT, TRACKING_URI

__all__ = ["ROOT", "TRACKING_URI"]
