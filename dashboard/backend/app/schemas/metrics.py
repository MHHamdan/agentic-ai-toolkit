"""Metrics schemas for the dashboard API."""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class CostBreakdown(BaseModel):
    """4-component cost breakdown (Equation 5 from paper)."""

    inference_cost: float = Field(default=0.0, description="Token inference costs")
    tool_cost: float = Field(default=0.0, description="Tool invocation costs")
    latency_cost: float = Field(default=0.0, description="Time-based costs")
    human_cost: float = Field(default=0.0, description="Human intervention costs")
    total_cost: float = Field(default=0.0, description="Sum of all costs")

    def calculate_total(self) -> float:
        """Calculate total cost from components."""
        self.total_cost = (
            self.inference_cost + self.tool_cost + self.latency_cost + self.human_cost
        )
        return self.total_cost


class CNSRMetrics(BaseModel):
    """Cost-Normalized Success Rate metrics (Equation 6 from paper)."""

    cnsr: float = Field(description="CNSR = Success Rate / Mean Cost")
    success_rate: float = Field(description="Proportion of successful tasks")
    mean_cost: float = Field(description="Average cost per task")
    total_tasks: int = Field(description="Total number of tasks")
    total_successes: int = Field(description="Number of successful tasks")
    cost_breakdown: CostBreakdown = Field(description="Average cost breakdown")
    timestamp: datetime = Field(default_factory=datetime.now)


class RollingMetrics(BaseModel):
    """Rolling window metrics."""

    window_id: int = Field(description="Window identifier")
    window_size: int = Field(description="Number of tasks in window")
    success_rate: float = Field(description="Success rate in this window")
    mean_cost: float = Field(description="Mean cost in this window")
    task_count: int = Field(description="Tasks completed in window")
    timestamp: datetime = Field(default_factory=datetime.now)


class TrendAnalysis(BaseModel):
    """Trend analysis for metrics."""

    metric_name: str
    trend: str = Field(description="stable, improving, or degrading")
    current_value: float
    previous_value: float
    change_percent: float
    is_positive: bool = Field(description="Whether the trend is good")


class MetricsSummary(BaseModel):
    """Overall metrics summary for dashboard."""

    cnsr: CNSRMetrics
    rolling_success_rate: float
    rolling_history: List[float] = Field(default_factory=list)
    trends: List[TrendAnalysis] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.now)
