"""
Analytics router for classification metrics.
"""

from fastapi import APIRouter

from app.dependencies import CurrentUser
from app.schemas.responses import AnalyticsSummaryResponse
from app.services.analytics_service import analytics_service

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


@router.get("/summary", response_model=AnalyticsSummaryResponse)
async def get_analytics_summary(current_user: CurrentUser):
    """Aggregated classification analytics (in-memory, resets on restart)."""
    return analytics_service.get_summary()
