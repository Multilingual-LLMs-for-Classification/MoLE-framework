"""
Job status polling endpoint.

After POST /api/v1/classify returns 202, clients poll this endpoint
until status transitions from "queued" to "done" or "error".
"""

from fastapi import APIRouter, HTTPException, status

from app.dependencies import CurrentUser
from app.schemas.responses import JobStatusResponse
from app.services.job_store import job_store

router = APIRouter(prefix="/api/v1/classify", tags=["Classification"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, current_user: CurrentUser) -> JobStatusResponse:
    """
    Poll for the result of an async classification job.

    Returns the current job state:
    - ``status: "queued"``  — job is waiting in the worker queue
    - ``status: "done"``    — inference complete, result fields are populated
    - ``status: "error"``   — inference failed, ``detail`` field explains why

    Requires authentication via Bearer token.
    """
    if not job_store.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job queue unavailable (Redis not connected).",
        )

    record = await job_store.get_status(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found or expired.",
        )

    return JobStatusResponse(**record)
