"""Metrics service wrapping toolkit evaluation metrics."""
from datetime import datetime
from typing import Optional, List

from app.schemas.metrics import (
    CNSRMetrics,
    CostBreakdown,
    MetricsSummary,
    RollingMetrics,
    TrendAnalysis,
)


class MetricsService:
    """Service for metrics calculations and retrieval."""

    def __init__(self):
        self._history: List[dict] = []
        self._rolling_history: List[float] = []
        self._last_cnsr: Optional[CNSRMetrics] = None

    def update_from_evaluation(self, result):
        """Update metrics from an evaluation result."""
        if result:
            self._last_cnsr = CNSRMetrics(
                cnsr=result.cnsr,
                success_rate=result.success_rate,
                mean_cost=result.mean_cost,
                total_tasks=result.total_tasks,
                total_successes=result.successful_tasks,
                cost_breakdown=CostBreakdown(
                    inference_cost=result.cost_breakdown.get("inference", 0),
                    tool_cost=result.cost_breakdown.get("tools", 0),
                    latency_cost=result.cost_breakdown.get("latency", 0),
                    human_cost=result.cost_breakdown.get("human", 0),
                    total_cost=result.total_cost,
                ),
                timestamp=datetime.now(),
            )
            self._rolling_history.append(result.success_rate)
            # Keep last 20 data points
            if len(self._rolling_history) > 20:
                self._rolling_history = self._rolling_history[-20:]

    async def get_summary(self) -> MetricsSummary:
        """Get overall metrics summary."""
        cnsr = await self.get_cnsr()

        return MetricsSummary(
            cnsr=cnsr,
            rolling_success_rate=self._rolling_history[-1] if self._rolling_history else cnsr.success_rate,
            rolling_history=self._rolling_history if self._rolling_history else [cnsr.success_rate],
            trends=[
                TrendAnalysis(
                    metric_name="success_rate",
                    trend="stable" if len(self._rolling_history) < 2 else (
                        "increasing" if self._rolling_history[-1] > self._rolling_history[-2] else "decreasing"
                    ),
                    current_value=cnsr.success_rate,
                    previous_value=self._rolling_history[-2] if len(self._rolling_history) > 1 else cnsr.success_rate,
                    change_percent=0.0,
                    is_positive=True,
                ),
                TrendAnalysis(
                    metric_name="mean_cost",
                    trend="stable",
                    current_value=cnsr.mean_cost,
                    previous_value=cnsr.mean_cost,
                    change_percent=0.0,
                    is_positive=True,
                ),
            ],
            last_updated=datetime.now(),
        )

    async def get_cnsr(self, evaluation_id: Optional[str] = None) -> CNSRMetrics:
        """Get CNSR metrics (Cost-Normalized Success Rate)."""
        # Return cached metrics if available
        if self._last_cnsr:
            return self._last_cnsr

        # Default demo data if no evaluations run yet
        cost_breakdown = CostBreakdown(
            inference_cost=0.0150,
            tool_cost=0.0050,
            latency_cost=0.0030,
            human_cost=0.0020,
            total_cost=0.0250,
        )

        success_rate = 0.85
        mean_cost = 0.025
        cnsr = success_rate / mean_cost if mean_cost > 0 else 0

        return CNSRMetrics(
            cnsr=round(cnsr, 2),
            success_rate=success_rate,
            mean_cost=mean_cost,
            total_tasks=121,
            total_successes=103,
            cost_breakdown=cost_breakdown,
            timestamp=datetime.now(),
        )

    async def get_rolling_metrics(
        self, window_size: int = 10, limit: int = 10
    ) -> List[RollingMetrics]:
        """Get rolling window metrics."""
        metrics = []
        history = self._rolling_history if self._rolling_history else [0.85]

        for i, sr in enumerate(history[-limit:]):
            metrics.append(
                RollingMetrics(
                    window_id=i,
                    window_size=window_size,
                    success_rate=sr,
                    mean_cost=0.025,
                    task_count=window_size,
                    timestamp=datetime.now(),
                )
            )
        return metrics

    async def get_cost_breakdown(
        self, evaluation_id: Optional[str] = None
    ) -> CostBreakdown:
        """Get 4-component cost breakdown."""
        if self._last_cnsr:
            return self._last_cnsr.cost_breakdown

        return CostBreakdown(
            inference_cost=0.0150,
            tool_cost=0.0050,
            latency_cost=0.0030,
            human_cost=0.0020,
            total_cost=0.0250,
        )

    async def get_trends(self, metric: str, period: str) -> dict:
        """Get trend analysis for a metric."""
        cnsr = await self.get_cnsr()
        return {
            "metric": metric,
            "period": period,
            "trend": "stable",
            "current_value": cnsr.success_rate if metric == "success_rate" else cnsr.mean_cost,
            "previous_value": cnsr.success_rate if metric == "success_rate" else cnsr.mean_cost,
            "change_percent": 0.0,
        }


# Singleton instance
metrics_service = MetricsService()
