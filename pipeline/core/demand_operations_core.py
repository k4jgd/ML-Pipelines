"""Shared logic for the Demand & Operations pipeline: preprocessing, feature
engineering, chronological split, candidate models, and evaluation.

Used by `pipeline/demand_operations.py` (initial training + EDA on the
full history).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42
REGISTERED_MODEL_NAME = "demand_operations_model"
TARGET_COL = "NextBookingCount"
PRIMARY_METRIC = "RMSE"
HIGHER_IS_BETTER = False

REQUIRED_COLUMNS = [
    "LocationId", "Date", "BookingCount", "WalkInCount", "OnlineBookingCount",
    "CancellationCount", "MachineTypeCount", "CourseCount",
]
NUM_COLS = [
    "BookingCount", "WalkInCount", "OnlineBookingCount",
    "CancellationCount", "MachineTypeCount", "CourseCount",
]
FEATURE_COLS = [
    "BookingCount", "WalkInCount", "OnlineBookingCount", "CancellationCount",
    "MachineTypeCount", "CourseCount", "DayOfWeekNum", "IsWeekend",
    "CancellationRate", "Lag1_BookingCount", "Lag2_BookingCount",
    "Lag1_CancellationCount", "RollMean3_BookingCount", "DaysSincePrevVisit",
    "LocationHistMeanBooking",
]


def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for c in df.select_dtypes(include="object"):
        df[c] = df[c].str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["LocationId", "Date"]).reset_index(drop=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["LocationId", "Date"]).copy()
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds every engineered column + the target (except `LocationHistMeanBooking`,
    which depends on which rows count as "history" and is computed separately by
    `compute_location_hist_mean`), WITHOUT dropping rows missing lag/rolling
    history or a target. Used by both `engineer_features` (training, which then
    drops those rows) and the serving layer (which wants exactly the row
    `engineer_features` would drop: each location's latest day, to predict its
    next one).
    """
    df = df.copy()
    df["DayOfWeekNum"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeekNum"].isin([5, 6]).astype(int)
    df["CancellationRate"] = (df["CancellationCount"] / df["BookingCount"].replace(0, np.nan)).fillna(0)

    g = df.groupby("LocationId")
    for lag in [1, 2]:
        df[f"Lag{lag}_BookingCount"] = g["BookingCount"].shift(lag)
        df[f"Lag{lag}_CancellationCount"] = g["CancellationCount"].shift(lag)
    df["RollMean3_BookingCount"] = g["BookingCount"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    df["DaysSincePrevVisit"] = g["Date"].diff().dt.days

    df[TARGET_COL] = g["BookingCount"].shift(-1)
    # Sibling targets for the other Demand & Operations capabilities (walk-in
    # forecast, online-booking forecast, cancellation risk), which share this
    # same feature set and preprocessing but predict a different next-day count.
    df["NextWalkInCount"] = g["WalkInCount"].shift(-1)
    df["NextOnlineBookingCount"] = g["OnlineBookingCount"].shift(-1)
    next_cancellation_count = g["CancellationCount"].shift(-1)
    df["NextCancellationFlag"] = np.where(
        next_cancellation_count.isna(), np.nan, (next_cancellation_count > 0).astype(float)
    )
    return df


def compute_location_hist_mean(df: pd.DataFrame) -> tuple[pd.Series, float]:
    """Per-location historical mean BookingCount + a global fallback mean for
    locations not present in `df`. Called on the train split during training
    (to avoid leakage) and on the full available history during serving.
    """
    loc_hist_mean = df.groupby("LocationId")["BookingCount"].mean()
    global_mean = df["BookingCount"].mean()
    return loc_hist_mean, global_mean


def engineer_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """`target_col` lets the sibling Demand & Operations pipelines (walk-in
    forecast, online-booking forecast, cancellation) reuse this exact
    preprocessing/feature-engineering while dropping rows for their own
    target's missing-history/no-next-row cases.
    """
    df = add_features(df)
    model_df = df.dropna(subset=["Lag1_BookingCount", "RollMean3_BookingCount",
                                  "DaysSincePrevVisit", target_col]).copy()
    return model_df.sort_values("Date").reset_index(drop=True)


def chronological_split(model_df: pd.DataFrame, train_frac: float = 0.75):
    cutoff = model_df["Date"].quantile(train_frac, interpolation="nearest")
    train = model_df[model_df["Date"] <= cutoff].copy()
    test = model_df[model_df["Date"] > cutoff].copy()

    # Location-level historical mean booking, fit on TRAIN ONLY to avoid leakage.
    loc_hist_mean, global_mean = compute_location_hist_mean(train)
    train["LocationHistMeanBooking"] = train["LocationId"].map(loc_hist_mean)
    test["LocationHistMeanBooking"] = test["LocationId"].map(loc_hist_mean).fillna(global_mean)

    train[FEATURE_COLS] = train[FEATURE_COLS].fillna(0)
    test[FEATURE_COLS] = test[FEATURE_COLS].fillna(0)
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
