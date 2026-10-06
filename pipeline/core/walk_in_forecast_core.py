"""Shared logic for the Walk-in Forecast capability: same Demand & Operations
preprocessing/feature engineering as `demand_operations_core`, targeting
next-day WalkInCount instead of next-day BookingCount.

Used by `pipeline/walk_in_forecast.py` (initial training + EDA).
"""
from functools import partial

from . import demand_operations_core as _base
from .demand_operations_core import (  # noqa: F401 (re-exported for the training script)
    FEATURE_COLS,
    REQUIRED_COLUMNS,
    add_features,
    chronological_split,
    compute_location_hist_mean,
    evaluate,
    evaluate_primary,
    get_candidate_models,
    load_data,
    preprocess,
)

RANDOM_STATE = 42
REGISTERED_MODEL_NAME = "walk_in_forecast_model"
TARGET_COL = "NextWalkInCount"
PRIMARY_METRIC = "RMSE"
HIGHER_IS_BETTER = False

engineer_features = partial(_base.engineer_features, target_col=TARGET_COL)
