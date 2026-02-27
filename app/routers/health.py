"""
Health check router for service monitoring.
"""

from fastapi import APIRouter

from app.config import settings
from app.schemas.responses import HealthResponse, ReadyResponse
from app.services.routing_service import routing_service

router = APIRouter(prefix="/api/v1/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.

    Returns 200 if the service is running.
    Does not check if models are loaded.
    """
    return HealthResponse(
        status="healthy",
        version=settings.api_version
    )


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check() -> ReadyResponse:
    """
    Readiness check endpoint.

    Returns 200 only if the service is ready to handle requests,
    i.e., models are loaded and initialized.
    """
    is_ready = routing_service.is_initialized

    details = None
    if is_ready:
        stats = routing_service.get_system_stats()
        details = {
            "domains": stats.get("total_domains", 0),
            "tasks": stats.get("total_tasks", 0),
            "languages": stats.get("supported_languages", 0)
        }

    return ReadyResponse(
        status="ready" if is_ready else "not_ready",
        models_loaded=is_ready,
        details=details
    )


@router.get("/live")
async def liveness_check() -> dict:
    """
    Liveness check endpoint for Kubernetes.

    Simple endpoint that returns 200 if the process is alive.
    """
    return {"status": "alive"}
