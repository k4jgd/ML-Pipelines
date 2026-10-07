"""Configuration for the utilization pipeline."""

REGISTERED_MODEL_NAME = "utilization_model"
TARGET_COL = "NextUtilizationPct"
PRIMARY_METRIC = "RMSE"
HIGHER_IS_BETTER = False
REQUIRED_COLUMNS = [
    "MachineId",
    "Date",
    "OccupiedMinutes",
    "UtilizationPct",
    "UnknownDuration",
]
RAW_COLUMNS = ["MachineId", "StartedAt", "EndedAt", "SessionMinutes"]
FEATURE_COLS = [
    "UtilizationPct",
    "UnknownDuration",
    "Lag1",
    "Lag7",
    "Mean3",
    "Mean7",
    "HistoryDays",
    "NextDayOfWeek",
    "NextIsWeekend",
]
