"""Preprocess logic for machine_intelligence."""

import numpy as np
import pandas as pd

from .config import FEATURE_COLS, TARGET_COL


def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["MachineId", "Date"]).reset_index(drop=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["MachineId", "Date"]).copy()
    df["AvgCycleDurationMinutes"] = pd.to_numeric(
        df["AvgCycleDurationMinutes"], errors="coerce"
    )
    df["AvgCycleDurationMinutes"] = df.groupby("MachineId")[
        "AvgCycleDurationMinutes"
    ].transform(lambda s: s.fillna(s.median()))
    df["AvgCycleDurationMinutes"] = df["AvgCycleDurationMinutes"].fillna(
        df["AvgCycleDurationMinutes"].median()
    )
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
    df["CompletionRate"] = (
        df["CompletedSessionCount"] / df["StartedSessionCount"].replace(0, np.nan)
    ).fillna(1.0)
    df["FailedToday"] = (df["FailureCount"] > 0).astype(int)
    g = df.groupby("MachineId")
    df["Lag1_FailedToday"] = g["FailedToday"].shift(1)
    df["Lag1_FailureCount"] = g["FailureCount"].shift(1)
    df["RollMean3_FailureCount"] = (
        g["FailureCount"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    )
    df["RollSum5_FailureCount"] = (
        g["FailureCount"].shift(1).rolling(5).sum().reset_index(level=0, drop=True)
    )
    df["DaysSincePrevService"] = g["Date"].diff().dt.days
    df[TARGET_COL] = g["FailedToday"].shift(-1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_features(df)
    model_df = df.dropna(
        subset=[
            "Lag1_FailedToday",
            "RollMean3_FailureCount",
            "DaysSincePrevService",
            TARGET_COL,
        ]
    ).copy()
    model_df = model_df.sort_values("Date").reset_index(drop=True)
    model_df[FEATURE_COLS] = model_df[FEATURE_COLS].fillna(0)
    return model_df


def chronological_split(model_df: pd.DataFrame, train_frac: float = 0.75):
    cutoff = model_df["Date"].quantile(train_frac, interpolation="nearest")
    train = model_df[model_df["Date"] <= cutoff].copy()
    test = model_df[model_df["Date"] > cutoff].copy()
    return (train, test, cutoff)
