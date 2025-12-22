"""Metrics API endpoints."""
from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime

from app.schemas.metrics import CNSRMetrics, CostBreakdown, MetricsSummary, RollingMetrics
from app.services.metrics_service import metrics_service

router = APIRouter()


@router.get("/summary", response_model=MetricsSummary)
async def get_metrics_summary():
    """Get overall metrics summary for dashboard."""
    return await metrics_service.get_summary()


@router.get("/cnsr", response_model=CNSRMetrics)
async def get_cnsr_metrics(
    evaluation_id: Optional[str] = Query(None, description="Specific evaluation ID"),
):
    """Get CNSR (Cost-Normalized Success Rate) metrics."""
    return await metrics_service.get_cnsr(evaluation_id)


@router.get("/rolling", response_model=list[RollingMetrics])
async def get_rolling_metrics(
    window_size: int = Query(5, ge=1, le=100, description="Window size"),
    limit: int = Query(20, ge=1, le=100, description="Number of windows to return"),
):
    """Get rolling window metrics."""
    return await metrics_service.get_rolling_metrics(window_size, limit)


@router.get("/cost-breakdown", response_model=CostBreakdown)
async def get_cost_breakdown(
    evaluation_id: Optional[str] = Query(None, description="Specific evaluation ID"),
):
    """Get 4-component cost breakdown."""
    return await metrics_service.get_cost_breakdown(evaluation_id)


@router.get("/trends")
async def get_trends(
    metric: str = Query("success_rate", description="Metric to analyze"),
    period: str = Query("1h", description="Time period: 1h, 24h, 7d"),
):
    """Get trend analysis for a metric."""
    return await metrics_service.get_trends(metric, period)
