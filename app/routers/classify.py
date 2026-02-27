"""
Classification router for text classification endpoints.
"""

import time
from typing import List

from fastapi import APIRouter, HTTPException, status

from app.dependencies import CurrentUser, RoutingServiceDep
from app.schemas.requests import ClassifyRequest, BatchClassifyRequest
from app.schemas.responses import (
    ClassifyResponse,
    BatchClassifyResponse,
    SystemStatsResponse,
)
from app.services.analytics_service import analytics_service

router = APIRouter(prefix="/api/v1/classify", tags=["Classification"])


@router.post("", response_model=ClassifyResponse)
async def classify_text(
    request: ClassifyRequest,
    current_user: CurrentUser,
    routing_service: RoutingServiceDep,
) -> ClassifyResponse:
    """
    Classify a single text input.

    The description is combined with the text to form the prompt used
    for language/domain/task routing. The text is then passed separately
    to the selected expert for classification.

    Requires authentication via Bearer token.
    """
    if not routing_service.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification service not ready. Models are still loading."
        )

    try:
        result = await routing_service.classify(request)
        analytics_service.record_classification(result.model_dump())
        return result

    except TimeoutError as e:
        analytics_service.record_error()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(e)
        )
    except Exception as e:
        analytics_service.record_error()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
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
