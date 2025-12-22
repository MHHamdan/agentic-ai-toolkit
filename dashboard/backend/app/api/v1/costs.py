"""Cost analysis API endpoints."""
from fastapi import APIRouter, Query
from typing import Optional

from app.schemas.metrics import CostBreakdown
from app.services.cost_service import cost_service

router = APIRouter()


@router.get("/breakdown", response_model=CostBreakdown)
async def get_cost_breakdown(
    evaluation_id: Optional[str] = Query(None),
):
    """Get 4-component cost breakdown."""
    return await cost_service.get_breakdown(evaluation_id)


@router.get("/history")
async def get_cost_history(
    period: str = Query("24h", description="Time period: 1h, 24h, 7d, 30d"),
    granularity: str = Query("1h", description="Data granularity"),
):
    """Get cost history over time."""
    return await cost_service.get_history(period, granularity)


@router.get("/by-model")
async def get_cost_by_model():
    """Get cost breakdown by model."""
    return await cost_service.get_by_model()


@router.get("/projections")
async def get_cost_projections(
    days: int = Query(30, ge=1, le=90),
):
    """Get cost projections for upcoming period."""
    return await cost_service.get_projections(days)


@router.get("/optimization")
async def get_optimization_suggestions():
    """Get cost optimization suggestions."""
    return await cost_service.get_optimization_suggestions()
