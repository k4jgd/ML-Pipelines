"""Models logic for machine_intelligence."""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def get_candidate_models(y_train, random_state: int = RANDOM_STATE) -> dict:
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            class_weight="balanced",
            random_state=random_state,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05, random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            random_state=random_state,
            verbosity=0,
            eval_metric="logloss",
        ),
    }
