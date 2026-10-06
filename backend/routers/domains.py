"""HTTP routing only; domain catalog and all inference live in inference/."""
from fastapi import APIRouter, HTTPException, Query

from inference import data, service
from inference.contracts import EntityHistoryPoint, EntitySummary, PredictionResponse, WorkspaceInfo
from inference.datasets import get_spec
from inference.registry import DOMAINS, get_domain

router = APIRouter(prefix="/api")


def _not_found(exc: KeyError) -> HTTPException:
    return HTTPException(status_code=404, detail=exc.args[0])


@router.get("/workspaces", response_model=list[WorkspaceInfo], tags=["workspaces"])
def list_workspaces():
    return [service.workspace(key) for key in DOMAINS]


def domain_router(domain_key: str) -> APIRouter:
    """Concrete domain paths in OpenAPI; capabilities share a stable route contract."""
    routes = APIRouter(prefix=f"/{domain_key}", tags=[domain_key])
    dataset = get_spec(get_domain(domain_key).pipeline_key)

    @routes.get("", response_model=WorkspaceInfo)
    def overview():
        return service.workspace(domain_key)

    @routes.get("/entities", response_model=list[EntitySummary])
    def entities(search: str = ""):
        return data.list_entities(dataset, search)

    @routes.get("/entities/{entity_id}/history", response_model=list[EntityHistoryPoint])
    def history(entity_id: str, limit: int = Query(20, ge=1, le=500)):
        try:
            return data.entity_history(dataset, entity_id, limit)
        except KeyError as exc:
            raise _not_found(exc) from exc

    # Create an explicit API endpoint for each registered capability.
    def prediction_endpoint(prediction_key: str):
        def endpoint(entity_id: str = Query(..., min_length=1)):
            try:
                return service.predict(domain_key, prediction_key, entity_id)
            except KeyError as exc:
                raise _not_found(exc) from exc
        return endpoint

    for capability in get_domain(domain_key).capabilities:
        routes.add_api_route(
            f"/{capability.key}/predict", prediction_endpoint(capability.key),
            methods=["GET"], response_model=PredictionResponse,
            name=f"{domain_key}_{capability.key}", summary=capability.name,
        )
    return routes


for domain in DOMAINS:
    router.include_router(domain_router(domain))
