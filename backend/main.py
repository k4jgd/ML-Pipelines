"""Unified prediction API for the Washtym ML pipelines.

Serves domain workspaces through the independent inference package while
preserving the original pipeline API. Capabilities are declared in
inference/domains/; training and retraining run separately.

Run: uv run uvicorn backend.main:app --reload --port 8123
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.pipelines import router as pipelines_router
from backend.routers.domains import router as domains_router

app = FastAPI(title="Washtym Prediction API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipelines_router)
app.include_router(domains_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
