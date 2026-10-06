"""Compatibility facade for the original pipeline API; inference is delegated."""
from inference.data import entity_count, entity_history, list_entities
from inference.datasets import PipelineSpec
from inference.models import NoChampionError
from inference.registry import DOMAINS
from inference.service import predict
from backend.schemas import PredictionResult


def predict_entity(spec: PipelineSpec, entity_id: str) -> PredictionResult:
    domain = next(d for d in DOMAINS.values() if d.pipeline_key == spec.key)
    capability = next(c for c in domain.capabilities if c.registered_model_name == spec.core.REGISTERED_MODEL_NAME)
    result = predict(domain.key, capability.key, entity_id)
    if result.status == "no_champion":
        raise NoChampionError(result.message)
    if result.status != "ok":
        raise RuntimeError(result.message)
    return PredictionResult(
        pipeline_key=spec.key, entity_id=entity_id, as_of_date=result.as_of_date,
        task_type=spec.task_type, target_label=spec.target_label,
        predicted_value=float(result.prediction_value),
        predicted_label=int(result.prediction_value) if spec.task_type == "classification" else None,
        predicted_probability=result.uncertainty.value if result.uncertainty else None,
        model_version=result.model_version, model_registered_name=result.model_registered_name,
    )


def predict_all(spec: PipelineSpec) -> list[PredictionResult]:
    return [predict_entity(spec, entity.entity_id) for entity in list_entities(spec)]


__all__ = ["entity_count", "entity_history", "list_entities", "predict_entity", "predict_all"]
