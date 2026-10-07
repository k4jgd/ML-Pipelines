"""Models logic for demand_operations."""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor

from .config import RANDOM_STATE


def get_candidate_models(random_state: int = RANDOM_STATE) -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=4, random_state=random_state
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.05, random_state=random_state
        ),
        "XGBoost": XGBRegressor(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.05,
            random_state=random_state,
            verbosity=0,
        ),
    }
