"""Shared MLflow tracking configuration for all pipelines."""

from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"


def configure(experiment_name: str) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
