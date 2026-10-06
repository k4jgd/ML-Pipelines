from typing import Literal, Optional

from pydantic import BaseModel
from inference.contracts import EntityHistoryPoint, EntitySummary


class ModelInfo(BaseModel):
    registered_model_name: str
    version: Optional[str] = None
    has_champion: bool
    metrics: dict[str, float] = {}
    tags: dict[str, str] = {}


class PipelineInfo(BaseModel):
    key: str
    display_name: str
    task_type: Literal["regression", "classification"]
    target_label: str
    entity_col: str
    entity_count: int
    model: ModelInfo


class PredictionResult(BaseModel):
    pipeline_key: str
    entity_id: str
    as_of_date: str
    task_type: Literal["regression", "classification"]
    target_label: str
    predicted_value: float
    predicted_label: Optional[int] = None
    predicted_probability: Optional[float] = None
    model_version: str
    model_registered_name: str
