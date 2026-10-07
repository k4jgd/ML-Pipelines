"""Models logic for utilization."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline


def get_candidate_models():
    return {
        "MeanBaseline": DummyRegressor(strategy="mean"),
        "PersistenceBaseline": make_pipeline(
            ColumnTransformer([("current", "passthrough", ["UtilizationPct"])]),
            LinearRegression(),
        ),
        "Ridge": Ridge(alpha=10),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.03, random_state=42
        ),
    }


def fit_candidate(name, model, features, target):
    return model.fit(
        features, features.UtilizationPct if name == "PersistenceBaseline" else target
    )
