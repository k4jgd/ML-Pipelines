"""Configuration for the cancellation pipeline."""

from ..demand_operations.config import FEATURE_COLS as FEATURE_COLS
from ..demand_operations.config import REQUIRED_COLUMNS as REQUIRED_COLUMNS

RANDOM_STATE = 42
REGISTERED_MODEL_NAME = "cancellation_model"
TARGET_COL = "NextCancellationFlag"
PRIMARY_METRIC = "F1"
HIGHER_IS_BETTER = True
