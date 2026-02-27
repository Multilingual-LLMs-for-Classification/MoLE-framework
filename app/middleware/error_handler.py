"""
Global error handling middleware.
"""

import traceback
from typing import Callable

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.schemas.responses import ErrorResponse


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware to catch and handle unhandled exceptions.

    Converts exceptions to standardized JSON error responses.
    """

    async def dispatch(self, request: Request, call_next: Callable):
        try:
            response = await call_next(request)
            return response

        except Exception as e:
            # Log the full traceback
            error_trace = traceback.format_exc()
            print(f"Unhandled exception: {error_trace}")

            # Determine status code based on exception type
            status_code = self._get_status_code(e)

            # Build error response
            error_response = ErrorResponse(
                error=type(e).__name__,
                detail=str(e),
                request_id=getattr(request.state, "request_id", None)
            )

            return JSONResponse(
                status_code=status_code,
                content=error_response.model_dump()
            )

    def _get_status_code(self, exception: Exception) -> int:
        """Map exception types to HTTP status codes."""
        exception_status_map = {
            "ValueError": status.HTTP_400_BAD_REQUEST,
            "KeyError": status.HTTP_400_BAD_REQUEST,
            "TypeError": status.HTTP_400_BAD_REQUEST,
            "RuntimeError": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "TimeoutError": status.HTTP_504_GATEWAY_TIMEOUT,
            "PermissionError": status.HTTP_403_FORBIDDEN,
            "FileNotFoundError": status.HTTP_404_NOT_FOUND,
        }

        exception_name = type(exception).__name__

        # Check for CUDA/GPU related errors
        if "cuda" in exception_name.lower() or "gpu" in str(exception).lower():
            return status.HTTP_503_SERVICE_UNAVAILABLE

        return exception_status_map.get(
            exception_name,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        )


async def add_request_id(request: Request, call_next: Callable):
    """Middleware to add request ID to request state."""
    from uuid import uuid4
    request.state.request_id = str(uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response
