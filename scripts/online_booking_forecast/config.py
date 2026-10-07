"""Configuration for the online_booking_forecast pipeline."""

from ..demand_operations.config import FEATURE_COLS as FEATURE_COLS
from ..demand_operations.config import REQUIRED_COLUMNS as REQUIRED_COLUMNS

RANDOM_STATE = 42
REGISTERED_MODEL_NAME = "online_booking_forecast_model"
TARGET_COL = "NextOnlineBookingCount"
PRIMARY_METRIC = "RMSE"
HIGHER_IS_BETTER = False
