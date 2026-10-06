from fastapi import APIRouter, HTTPException, Query

from backend.model_cache import NoChampionError, get_champion_run_metrics
from backend.pipeline_registry import PIPELINE_SPECS, get_spec
from backend.schemas import EntityHistoryPoint, EntitySummary, ModelInfo, PipelineInfo, PredictionResult
from backend.services import prediction_service as svc

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])


def _model_info(registered_model_name: str) -> ModelInfo:
    try:
        info = get_champion_run_metrics(registered_model_name)
        return ModelInfo(
            registered_model_name=registered_model_name,
            version=str(info["version"]),
            has_champion=True,
            metrics=info["metrics"],
            tags=info["tags"],
        )
    except Exception:
        return ModelInfo(registered_model_name=registered_model_name, has_champion=False)


@router.get("", response_model=list[PipelineInfo])
def list_pipelines() -> list[PipelineInfo]:
    result = []
    for spec in PIPELINE_SPECS.values():
        result.append(
            PipelineInfo(
                key=spec.key,
                display_name=spec.display_name,
                task_type=spec.task_type,
                target_label=spec.target_label,
                entity_col=spec.entity_col,
                entity_count=svc.entity_count(spec),
                model=_model_info(spec.core.REGISTERED_MODEL_NAME),
            )
        )
    return result


@router.get("/{key}/model", response_model=ModelInfo)
def get_model(key: str) -> ModelInfo:
    spec = _spec_or_404(key)
    return _model_info(spec.core.REGISTERED_MODEL_NAME)


@router.get("/{key}/entities", response_model=list[EntitySummary])
def get_entities(key: str, search: str = Query("", description="Filter entity ids by substring")) -> list[EntitySummary]:
    spec = _spec_or_404(key)
    return svc.list_entities(spec, search=search)


@router.get("/{key}/entities/{entity_id}/history", response_model=list[EntityHistoryPoint])
def get_entity_history(key: str, entity_id: str, limit: int = 20) -> list[EntityHistoryPoint]:
    spec = _spec_or_404(key)
    try:
        return svc.entity_history(spec, entity_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{key}/entities/{entity_id}/predict", response_model=PredictionResult)
def predict_entity(key: str, entity_id: str) -> PredictionResult:
    spec = _spec_or_404(key)
    try:
        return svc.predict_entity(spec, entity_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NoChampionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _spec_or_404(key: str):
    try:
        return get_spec(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
