"""Configuration for the customer_intelligence pipeline."""

RANDOM_STATE = 42
REGISTERED_MODEL_NAME = "customer_intelligence_model"
TARGET_COL = "NextQuotedRevenue"
PRIMARY_METRIC = "RMSE"
HIGHER_IS_BETTER = False
REQUIRED_COLUMNS = [
    "ClientId",
    "Date",
    "BookingCount",
    "WalkInCount",
    "OnlineBookingCount",
    "CancellationCount",
    "QuotedRevenuePaise",
    "CourseCount",
    "MachineTypeCount",
]
NUM_COLS = [
    "BookingCount",
    "WalkInCount",
    "OnlineBookingCount",
    "CancellationCount",
    "QuotedRevenuePaise",
    "CourseCount",
    "MachineTypeCount",
]
FEATURE_COLS = [
    "BookingCount",
    "WalkInCount",
    "OnlineBookingCount",
    "CancellationCount",
    "CourseCount",
    "MachineTypeCount",
    "DayOfWeekNum",
    "IsWeekend",
    "CancellationRate",
    "OnlineShare",
    "DateOrdinal",
    "Lag1_QuotedRevenue",
    "Lag1_Converted",
    "RollMean3_QuotedRevenue",
    "ClientVisitNumber",
    "DaysSinceLastVisit",
]
