"""Configuration for the machine_intelligence pipeline."""

RANDOM_STATE = 42
REGISTERED_MODEL_NAME = "machine_intelligence_model"
TARGET_COL = "NextFailure"
PRIMARY_METRIC = "F1"
HIGHER_IS_BETTER = True
REQUIRED_COLUMNS = [
    "MachineId",
    "Date",
    "SessionCount",
    "StartedSessionCount",
    "CompletedSessionCount",
    "AvgCycleDurationMinutes",
    "OnlineObserved",
    "FailureCount",
    "MaxConsecutiveFailures",
]
FEATURE_COLS = [
    "SessionCount",
    "StartedSessionCount",
    "CompletedSessionCount",
    "CompletionRate",
    "AvgCycleDurationMinutes",
    "OnlineObserved",
    "MaxConsecutiveFailures",
    "DayOfWeekNum",
    "Lag1_FailedToday",
    "Lag1_FailureCount",
    "RollMean3_FailureCount",
    "RollSum5_FailureCount",
    "DaysSincePrevService",
]
