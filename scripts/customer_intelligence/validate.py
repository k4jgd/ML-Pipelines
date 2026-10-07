"""Validate logic for customer_intelligence."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import PRIMARY_METRIC


def evaluate(y_test, preds) -> dict:
    return {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": r2_score(y_test, preds),
    }


def evaluate_primary(y_test, preds) -> float:
    return evaluate(y_test, preds)[PRIMARY_METRIC]
