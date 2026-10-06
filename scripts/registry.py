"""Shared MLflow Model Registry helpers: compare a newly trained challenger
against the currently promoted champion, and promote it only if it wins.

Used by the training scripts: there is usually no champion yet on the first
run, so that model is registered unconditionally, while later runs only
promote a challenger if it beats the existing champion.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import mlflow
from mlflow.tracking import MlflowClient

CHAMPION_ALIAS = "champion"


@dataclass
class PromotionDecision:
    promoted: bool
    reason: str
    new_version: Optional[str] = None


def champion_uri(registered_model_name: str) -> str:
    return f"models:/{registered_model_name}@{CHAMPION_ALIAS}"


def load_champion_metric(
    client: MlflowClient,
    registered_model_name: str,
    X_test,
    y_test,
    metric_fn: Callable,
) -> tuple[Optional[float], Optional[str]]:
    """Loads the current champion (if any) and scores it on THIS run's test
    set, so the comparison is apples-to-apples against the challenger even
    though the champion was originally trained/evaluated on older data.
    Returns (metric, champion_version) or (None, None) if no champion exists.
    """
    try:
        mv = client.get_model_version_by_alias(registered_model_name, CHAMPION_ALIAS)
    except Exception:
        return None, None

    model = mlflow.pyfunc.load_model(champion_uri(registered_model_name))
    preds = model.predict(X_test)
    return float(metric_fn(y_test, preds)), mv.version


def compare_and_promote(
    client: MlflowClient,
    registered_model_name: str,
    run_id: str,
    artifact_path: str,
    challenger_metric: float,
    champion_metric: Optional[float],
    higher_is_better: bool,
    min_improvement: float = 0.0,
    extra_tags: Optional[dict] = None,
) -> PromotionDecision:
    """Registers + promotes `run_id`'s model to `champion` only if it beats
    the current champion metric by at least `min_improvement`. If there is no
    champion yet, the challenger is promoted unconditionally (bootstrap case).
    """
    if champion_metric is None:
        should_promote = True
        reason = "no existing champion; registering as the initial champion"
    elif higher_is_better:
        should_promote = challenger_metric > champion_metric + min_improvement
        reason = "challenger beat champion" if should_promote else "challenger did not beat champion"
    else:
        should_promote = challenger_metric < champion_metric - min_improvement
        reason = "challenger beat champion" if should_promote else "challenger did not beat champion"

    if not should_promote:
        return PromotionDecision(promoted=False, reason=reason)

    mv = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", registered_model_name)
    client.set_registered_model_alias(registered_model_name, CHAMPION_ALIAS, mv.version)
    for key, value in (extra_tags or {}).items():
        client.set_model_version_tag(registered_model_name, mv.version, key, str(value))
    return PromotionDecision(promoted=True, reason=reason, new_version=mv.version)
