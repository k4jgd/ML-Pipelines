"""Validate logic for utilization."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import PRIMARY_METRIC


def bounded_predictions(preds):
    values = np.asarray(preds, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite utilization prediction")
    return values.clip(0, 100)


def evaluate(y, preds):
    preds = bounded_predictions(preds)
    return {
        "MAE": float(mean_absolute_error(y, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
        "R2": float(r2_score(y, preds)),
    }


def evaluate_primary(y, preds):
    return evaluate(y, preds)[PRIMARY_METRIC]
