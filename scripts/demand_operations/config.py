"""Configuration for the demand_operations pipeline."""

RANDOM_STATE = 42
REGISTERED_MODEL_NAME = "demand_operations_model"
TARGET_COL = "NextBookingCount"
PRIMARY_METRIC = "RMSE"
HIGHER_IS_BETTER = False
REQUIRED_COLUMNS = [
    "LocationId",
    "Date",
    "BookingCount",
    "WalkInCount",
    "OnlineBookingCount",
    "CancellationCount",
    "MachineTypeCount",
    "CourseCount",
]
NUM_COLS = [
    "BookingCount",
    "WalkInCount",
    "OnlineBookingCount",
    "CancellationCount",
    "MachineTypeCount",
    "CourseCount",
]
FEATURE_COLS = [
    "BookingCount",
    "WalkInCount",
    "OnlineBookingCount",
    "CancellationCount",
    "MachineTypeCount",
    "CourseCount",
    "DayOfWeekNum",
    "IsWeekend",
    "CancellationRate",
    "Lag1_BookingCount",
    "Lag2_BookingCount",
    "Lag1_CancellationCount",
    "RollMean3_BookingCount",
    "DaysSincePrevVisit",
    "LocationHistMeanBooking",
]
