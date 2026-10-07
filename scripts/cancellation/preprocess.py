"""Preprocess logic for cancellation."""

from ..demand_operations import preprocess as _base
from ..demand_operations.preprocess import add_features as add_features
from ..demand_operations.preprocess import chronological_split as chronological_split
from ..demand_operations.preprocess import (
    compute_location_hist_mean as compute_location_hist_mean,
)
from ..demand_operations.preprocess import load_data as load_data
from ..demand_operations.preprocess import preprocess as preprocess
from .config import TARGET_COL


def engineer_features(df):
    model_df = _base.engineer_features(df, target_col=TARGET_COL)
    model_df[TARGET_COL] = model_df[TARGET_COL].astype(int)
    return model_df
