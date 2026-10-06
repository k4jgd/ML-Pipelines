"""Shared logic for the Customer Intelligence pipeline: preprocessing, feature
engineering, chronological split, candidate models, and evaluation.

Used by `pipeline/customer_intelligence.py` (initial training + EDA on
the full history).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42
REGISTERED_MODEL_NAME = "customer_intelligence_model"
TARGET_COL = "NextQuotedRevenue"
PRIMARY_METRIC = "RMSE"
HIGHER_IS_BETTER = False

REQUIRED_COLUMNS = [
    "ClientId", "Date", "BookingCount", "WalkInCount", "OnlineBookingCount",
    "CancellationCount", "QuotedRevenuePaise", "CourseCount", "MachineTypeCount",
]
NUM_COLS = [
    "BookingCount", "WalkInCount", "OnlineBookingCount", "CancellationCount",
    "QuotedRevenuePaise", "CourseCount", "MachineTypeCount",
]
FEATURE_COLS = [
    "BookingCount", "WalkInCount", "OnlineBookingCount", "CancellationCount",
    "CourseCount", "MachineTypeCount", "DayOfWeekNum", "IsWeekend",
    "CancellationRate", "OnlineShare", "DateOrdinal",
    "Lag1_QuotedRevenue", "Lag1_Converted", "RollMean3_QuotedRevenue",
    "ClientVisitNumber", "DaysSinceLastVisit",
]


def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["ClientId", "Date"]).reset_index(drop=True)
    df["Converted"] = (df["QuotedRevenuePaise"] > 0).astype(int)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["ClientId", "Date"]).copy()
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds every engineered column + the target, WITHOUT dropping rows that
    are missing lag/rolling history or a target. Used by both `engineer_features`
    (training, which then drops those rows) and the serving layer (which wants
    exactly the row `engineer_features` would drop: each client's latest visit,
    to predict its next one).
    """
    df = df.copy()
    df["DayOfWeekNum"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeekNum"].isin([5, 6]).astype(int)
    df["CancellationRate"] = (df["CancellationCount"] / df["BookingCount"].replace(0, np.nan)).fillna(0)
    df["OnlineShare"] = (df["OnlineBookingCount"] / df["BookingCount"].replace(0, np.nan)).fillna(0)
    df["DateOrdinal"] = (df["Date"] - df["Date"].min()).dt.days

    g = df.groupby("ClientId")
    df["Lag1_QuotedRevenue"] = g["QuotedRevenuePaise"].shift(1)
    df["Lag1_Converted"] = g["Converted"].shift(1)
    df["RollMean3_QuotedRevenue"] = g["QuotedRevenuePaise"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    df["ClientVisitNumber"] = g.cumcount()
    df["DaysSinceLastVisit"] = g["Date"].diff().dt.days

    df[TARGET_COL] = g["QuotedRevenuePaise"].shift(-1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_features(df)
    model_df = df.dropna(subset=["Lag1_QuotedRevenue", "RollMean3_QuotedRevenue",
                                  "DaysSinceLastVisit", TARGET_COL]).copy()
    model_df = model_df.sort_values("Date").reset_index(drop=True)
    model_df[FEATURE_COLS] = model_df[FEATURE_COLS].fillna(0)
    return model_df


def chronological_split(model_df: pd.DataFrame, train_frac: float = 0.75):
    cutoff = model_df["Date"].quantile(train_frac, interpolation="nearest")
    train = model_df[model_df["Date"] <= cutoff].copy()
    test = model_df[model_df["Date"] > cutoff].copy()
    return train, test, cutoff


def get_candidate_models(random_state: int = RANDOM_STATE) -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=4, random_state=random_state),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                       random_state=random_state),
        "XGBoost": XGBRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                 random_state=random_state, verbosity=0),
    }


def evaluate(y_test, preds) -> dict:
    return {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": r2_score(y_test, preds),
    }


def evaluate_primary(y_test, preds) -> float:
    return evaluate(y_test, preds)[PRIMARY_METRIC]
