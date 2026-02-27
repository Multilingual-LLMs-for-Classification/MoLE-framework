"""
Health and readiness endpoints for the expert worker service.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    model_key: str
    model_loaded: bool
    worker_id: str


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Basic health check — always returns 200 if the process is alive."""
    pool = request.app.state.worker_pool
    return HealthResponse(
        status="ok",
        model_key=pool.assigned_model_key,
        model_loaded=pool.is_ready(),
        worker_id=request.app.state.worker_id,
    )


@router.get("/health/ready", response_model=HealthResponse)
async def ready(request: Request):
    """
    Readiness probe — returns 200 only after the LLM has finished loading.
    Used by Docker health checks and the coordinator's startup dependency.
    """
    pool = request.app.state.worker_pool
    if not pool.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return HealthResponse(
        status="ready",
        model_key=pool.assigned_model_key,
        model_loaded=True,
        worker_id=request.app.state.worker_id,
    )
