"""Shared logic for the Machine Intelligence pipeline: preprocessing, feature
engineering, chronological split, candidate models, and evaluation.

Used by `pipeline/machine_intelligence.py` (initial training + EDA on the
full history).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
REGISTERED_MODEL_NAME = "machine_intelligence_model"
TARGET_COL = "NextFailure"
PRIMARY_METRIC = "F1"
HIGHER_IS_BETTER = True

REQUIRED_COLUMNS = [
    "MachineId", "Date", "SessionCount", "StartedSessionCount", "CompletedSessionCount",
    "AvgCycleDurationMinutes", "OnlineObserved", "FailureCount", "MaxConsecutiveFailures",
]
FEATURE_COLS = [
    "SessionCount", "StartedSessionCount", "CompletedSessionCount", "CompletionRate",
    "AvgCycleDurationMinutes", "OnlineObserved", "MaxConsecutiveFailures", "DayOfWeekNum",
    "Lag1_FailedToday", "Lag1_FailureCount", "RollMean3_FailureCount",
    "RollSum5_FailureCount", "DaysSincePrevService",
]


def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["MachineId", "Date"]).reset_index(drop=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["MachineId", "Date"]).copy()
    df["AvgCycleDurationMinutes"] = pd.to_numeric(df["AvgCycleDurationMinutes"], errors="coerce")
    df["AvgCycleDurationMinutes"] = df.groupby("MachineId")["AvgCycleDurationMinutes"].transform(
        lambda s: s.fillna(s.median())
    )
    df["AvgCycleDurationMinutes"] = df["AvgCycleDurationMinutes"].fillna(df["AvgCycleDurationMinutes"].median())
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds every engineered column + the target, WITHOUT dropping rows that
    are missing lag/rolling history or a target. Used by both `engineer_features`
    (training, which then drops those rows) and the serving layer (which wants
    exactly the row `engineer_features` would drop: each machine's latest
    service record, to predict its next one).
    """
    df = df.copy()
    df["DayOfWeekNum"] = df["Date"].dt.dayofweek
    df["CompletionRate"] = (df["CompletedSessionCount"] / df["StartedSessionCount"].replace(0, np.nan)).fillna(1.0)
    df["FailedToday"] = (df["FailureCount"] > 0).astype(int)

    g = df.groupby("MachineId")
    df["Lag1_FailedToday"] = g["FailedToday"].shift(1)
    df["Lag1_FailureCount"] = g["FailureCount"].shift(1)
    df["RollMean3_FailureCount"] = g["FailureCount"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    df["RollSum5_FailureCount"] = g["FailureCount"].shift(1).rolling(5).sum().reset_index(level=0, drop=True)
    df["DaysSincePrevService"] = g["Date"].diff().dt.days

    df[TARGET_COL] = g["FailedToday"].shift(-1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_features(df)
    model_df = df.dropna(subset=["Lag1_FailedToday", "RollMean3_FailureCount",
                                  "DaysSincePrevService", TARGET_COL]).copy()
    model_df = model_df.sort_values("Date").reset_index(drop=True)
    model_df[FEATURE_COLS] = model_df[FEATURE_COLS].fillna(0)
    return model_df


def chronological_split(model_df: pd.DataFrame, train_frac: float = 0.75):
    cutoff = model_df["Date"].quantile(train_frac, interpolation="nearest")
    train = model_df[model_df["Date"] <= cutoff].copy()
    test = model_df[model_df["Date"] > cutoff].copy()
    return train, test, cutoff


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
