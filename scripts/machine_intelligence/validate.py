"""Validate logic for machine_intelligence."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_test, preds, proba=None) -> dict:
    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1": f1_score(y_test, preds, zero_division=0),
    }
    if proba is not None:
        try:
            metrics["ROC_AUC"] = roc_auc_score(y_test, proba)
        except ValueError:
            metrics["ROC_AUC"] = float("nan")
    return metrics


def evaluate_primary(y_test, preds) -> float:
    return f1_score(y_test, preds, zero_division=0)
