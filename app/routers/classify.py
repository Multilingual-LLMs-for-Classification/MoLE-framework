"""
Classification router for text classification endpoints.
"""

import time
from typing import List

from fastapi import APIRouter, HTTPException, Response, status

from app.config import settings
from app.dependencies import CurrentUser, RoutingServiceDep
from app.schemas.requests import ClassifyRequest, BatchClassifyRequest
from app.schemas.responses import (
    BatchClassifyResponse,
    ClassifyResponse,
    JobAcceptedResponse,
    SystemStatsResponse,
)
from app.services.analytics_service import analytics_service
from app.services.job_store import job_store

router = APIRouter(prefix="/api/v1/classify", tags=["Classification"])


@router.post("")
async def classify_text(
    request: ClassifyRequest,
    current_user: CurrentUser,
    routing_service: RoutingServiceDep,
    response: Response,
):
    """
    Classify a single text input.

    **Coordinator mode (default):** returns HTTP 202 immediately with a job_id.
    Poll ``GET /api/v1/classify/{job_id}`` for the result.

    **Monolithic mode:** blocks until inference completes and returns the full
    ClassifyResponse directly (original behaviour, unchanged).

    Requires authentication via Bearer token.
    """
    if not routing_service.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification service not ready. Models are still loading.",
        )

    if settings.service_mode == "coordinator":
        if not job_store.is_connected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Job queue unavailable (Redis not connected).",
            )

        try:
            gating = await routing_service.prepare_dispatch(request)
        except TimeoutError as exc:
            analytics_service.record_error()
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc))
        except Exception as exc:
            analytics_service.record_error()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Gating failed: {exc}",
            )

        worker_id = routing_service.get_worker_id_for_model(gating["base_model_key"])
        job_metadata = {
            "request_id": gating["request_id"],
            "language": gating["language"],
            "domain": gating["domain"],
            "task": gating["task"],
            "routing_path": gating["routing_path"],
        }
        await job_store.enqueue(worker_id, gating["request_id"], gating["payload"], job_metadata)

        response.status_code = status.HTTP_202_ACCEPTED
        return JobAcceptedResponse(
            job_id=gating["request_id"],
            status="queued",
            poll_url=f"/api/v1/classify/{gating['request_id']}",
        )

    # Monolithic mode — synchronous, unchanged
    try:
        result = await routing_service.classify(request)
        analytics_service.record_classification(result.model_dump())
        return result
    except TimeoutError as exc:
        analytics_service.record_error()
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc))
    except Exception as exc:
        analytics_service.record_error()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {exc}",
        )


@router.post("/test", response_model=ClassifyResponse)
async def classify_text_test(
    request: ClassifyRequest,
    current_user: CurrentUser,
    routing_service: RoutingServiceDep,
) -> ClassifyResponse:
    """
    Temporary mock classification endpoint for frontend development.
    Returns a hardcoded response using the new request shape.
    """
    
    print("test classify method is hitting....")
    response = ClassifyResponse(
        request_id="00000000-0000-0000-0000-000000000001",
        language="english",
        domain="finance",
        task="rating",
        result="4",
        confidence=0.94,
        routing_path="english → finance → rating",
        processing_time_ms=12.5,
        domain_probabilities={
            "finance": 0.94,
            "general": 0.06,
        },
        raw_response=request.text,
    )
    analytics_service.record_classification(response.model_dump())
    return response


@router.post("/batch", response_model=BatchClassifyResponse)
async def classify_batch(
    request: BatchClassifyRequest,
    current_user: CurrentUser,
    routing_service: RoutingServiceDep,
) -> BatchClassifyResponse:
    """
    Classify multiple text inputs in batch.

    Processes items sequentially (GPU constraint).
    Maximum 100 items per batch.

    Requires authentication via Bearer token.

    Args:
        request: Batch request with list of classification items
        current_user: Authenticated user (injected)
        routing_service: Routing service (injected)

    Returns:
        BatchClassifyResponse with all results
    """
    print("hittinggg....")
    if not routing_service.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification service not ready. Models are still loading."
        )

    start_time = time.perf_counter()
    results: List[ClassifyResponse] = []
    failed = 0

    for item in request.items:
        try:
            result = await routing_service.classify(item)
            analytics_service.record_classification(result.model_dump())
            results.append(result)
        except Exception:
            analytics_service.record_error()
            failed += 1
            # Continue processing remaining items

    total_time_ms = (time.perf_counter() - start_time) * 1000

    return BatchClassifyResponse(
        results=results,
        total_processing_time_ms=total_time_ms,
        successful=len(results),
        failed=failed
    )


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    current_user: CurrentUser,
    routing_service: RoutingServiceDep,
) -> SystemStatsResponse:
    """
    Get system statistics about the classification service.

    Returns information about supported domains, tasks, and languages.

    Requires authentication via Bearer token.

    Args:
        current_user: Authenticated user (injected)
        routing_service: Routing service (injected)

    Returns:
        SystemStatsResponse with system statistics
    """
    if not routing_service.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification service not ready"
        )

    stats = routing_service.get_system_stats()

    return SystemStatsResponse(
        total_domains=stats.get("total_domains", 0),
        total_tasks=stats.get("total_tasks", 0),
        supported_languages=stats.get("supported_languages", 0),
        all_languages=stats.get("all_languages", []),
        domains=stats.get("domains", [])
    )
