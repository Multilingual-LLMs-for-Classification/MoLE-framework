"""
Response schemas for classification endpoints.
"""

from typing import Dict, Optional, Any, List

from pydantic import BaseModel, Field


class ClassifyResponse(BaseModel):
    """Response for single classification."""

    request_id: str = Field(..., description="Unique request identifier")
    language: str = Field(..., description="Detected language")
    domain: str = Field(..., description="Classified domain")
    task: str = Field(..., description="Selected task type")
    result: str = Field(..., description="Classification result from expert")
    confidence: Optional[float] = Field(
        None,
        description="Expert confidence score (0-1)"
    )
    routing_path: str = Field(
        ...,
        description="Full routing path taken",
        examples=["english → finance → rating"]
    )
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds"
    )

    # Optional fields based on request options
    domain_probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Domain probability distribution (if requested)"
    )
    raw_response: Optional[str] = Field(
        None,
        description="Raw expert model response (if requested)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": "550e8400-e29b-41d4-a716-446655440000",
                    "language": "english",
                    "domain": "finance",
                    "task": "rating",
                    "result": "5",
                    "confidence": 0.92,
                    "routing_path": "english → finance → rating",
                    "processing_time_ms": 234.5,
                    "domain_probabilities": {
                        "finance": 0.95,
                        "general": 0.05
                    }
                }
            ]
        }
    }


class BatchClassifyResponse(BaseModel):
    """Response for batch classification."""

    results: List[ClassifyResponse] = Field(
        ...,
        description="List of classification results"
    )
    total_processing_time_ms: float = Field(
        ...,
        description="Total batch processing time in milliseconds"
    )
    successful: int = Field(..., description="Number of successful classifications")
    failed: int = Field(..., description="Number of failed classifications")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class ReadyResponse(BaseModel):
    """Readiness check response."""

    status: str = Field(..., description="Service status")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional details about loaded components"
    )


class SystemStatsResponse(BaseModel):
    """System statistics response."""

    total_domains: int
    total_tasks: int
    supported_languages: int
    all_languages: List[str]
    domains: List[str]


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class RecentRequestItem(BaseModel):
    """Recent classification request in analytics."""

    timestamp: str
    request_id: str
    language: str
    domain: str
    task: str
    confidence: Optional[float] = None
    processing_time_ms: float


class AnalyticsSummaryResponse(BaseModel):
    """Analytics summary response."""

    total_requests: int
    total_errors: int
    error_rate: float
    avg_confidence: Optional[float] = None
    avg_processing_time_ms: Optional[float] = None
    language_distribution: Dict[str, int]
    domain_distribution: Dict[str, int]
    task_distribution: Dict[str, int]
    task_avg_confidence: Dict[str, float]
    recent_requests: List[RecentRequestItem]
