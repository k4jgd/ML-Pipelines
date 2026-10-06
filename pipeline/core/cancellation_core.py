"""Shared logic for the Cancellation capability: same Demand & Operations
preprocessing/feature engineering as `demand_operations_core`, targeting
next-day cancellation risk (whether the location's next recorded day has
any cancellations) as a binary classification problem.

Used by `pipeline/cancellation.py` (initial training + EDA).
"""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from . import demand_operations_core as _base
from .demand_operations_core import (  # noqa: F401 (re-exported for the training script)
    FEATURE_COLS,
    REQUIRED_COLUMNS,
    add_features,
    chronological_split,
    compute_location_hist_mean,
    load_data,
    preprocess,
)

RANDOM_STATE = 42
REGISTERED_MODEL_NAME = "cancellation_model"
TARGET_COL = "NextCancellationFlag"
PRIMARY_METRIC = "F1"
HIGHER_IS_BETTER = True


def engineer_features(df):
    model_df = _base.engineer_features(df, target_col=TARGET_COL)
    model_df[TARGET_COL] = model_df[TARGET_COL].astype(int)
    return model_df


def get_candidate_models(y_train, random_state: int = RANDOM_STATE) -> dict:
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced",
                                                  random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=4, class_weight="balanced",
                                                random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                        random_state=random_state),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                  scale_pos_weight=pos_weight, random_state=random_state,
                                  verbosity=0, eval_metric="logloss"),
    }


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
