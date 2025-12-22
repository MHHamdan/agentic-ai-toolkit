"""Cost analysis service."""
from datetime import datetime, timedelta
from typing import Optional, List
import random

from app.schemas.metrics import CostBreakdown
from app.config import settings


class CostService:
    """Service for cost analysis and projections."""

    def __init__(self):
        self._history: List[dict] = []

    async def get_breakdown(self, evaluation_id: Optional[str] = None) -> CostBreakdown:
        """Get 4-component cost breakdown."""
        return CostBreakdown(
            inference_cost=0.0156,
            tool_cost=0.0052,
            latency_cost=0.0031,
            human_cost=0.0021,
            total_cost=0.0260,
        )

    async def get_history(self, period: str, granularity: str) -> dict:
        """Get cost history over time."""
        # Generate demo data
        data_points = []
        periods = {"1h": 12, "24h": 24, "7d": 7, "30d": 30}
        count = periods.get(period, 24)

        base_cost = 0.02
        for i in range(count):
            data_points.append({
                "timestamp": (datetime.now() - timedelta(hours=count - i)).isoformat(),
                "total_cost": base_cost + random.uniform(-0.005, 0.01),
                "inference_cost": base_cost * 0.6 + random.uniform(-0.002, 0.004),
                "tool_cost": base_cost * 0.2 + random.uniform(-0.001, 0.002),
                "latency_cost": base_cost * 0.12 + random.uniform(-0.001, 0.001),
                "human_cost": base_cost * 0.08,
            })

        return {
            "period": period,
            "granularity": granularity,
            "data_points": data_points,
            "total_period_cost": sum(d["total_cost"] for d in data_points),
        }

    async def get_by_model(self) -> dict:
        """Get cost breakdown by model."""
        return {
            "models": [
                {
                    "model": "phi3:latest",
                    "total_cost": 0.156,
                    "tasks": 120,
                    "cost_per_task": 0.0013,
                    "token_rate": settings.token_rates.get("phi3:latest", 0.00001),
                },
                {
                    "model": "llama3.1:8b",
                    "total_cost": 0.089,
                    "tasks": 45,
                    "cost_per_task": 0.002,
                    "token_rate": settings.token_rates.get("llama3.1:8b", 0.00005),
                },
                {
                    "model": "mistral:latest",
                    "total_cost": 0.067,
                    "tasks": 35,
                    "cost_per_task": 0.0019,
                    "token_rate": settings.token_rates.get("mistral:latest", 0.00003),
                },
            ],
            "recommendation": "phi3:latest offers best cost efficiency for current workload",
        }

    async def get_projections(self, days: int) -> dict:
        """Get cost projections."""
        daily_cost = 0.5 + random.uniform(-0.1, 0.2)
        projected_total = daily_cost * days

        return {
            "period_days": days,
            "projected_total": round(projected_total, 2),
            "projected_daily_avg": round(daily_cost, 2),
            "confidence": 0.85,
            "breakdown": {
                "inference": round(projected_total * 0.6, 2),
                "tools": round(projected_total * 0.2, 2),
                "latency": round(projected_total * 0.12, 2),
                "human": round(projected_total * 0.08, 2),
            },
            "with_30_percent_buffer": round(projected_total * 1.3, 2),
        }

    async def get_optimization_suggestions(self) -> dict:
        """Get cost optimization suggestions."""
        return {
            "suggestions": [
                {
                    "category": "Model Selection",
                    "suggestion": "Switch to phi3:latest for simple queries",
                    "estimated_savings": "15-20%",
                    "impact": "low",
                },
                {
                    "category": "Prompt Optimization",
                    "suggestion": "Reduce average prompt length by 20%",
                    "estimated_savings": "10-15%",
                    "impact": "medium",
                },
                {
                    "category": "Caching",
                    "suggestion": "Implement response caching for repeated queries",
                    "estimated_savings": "25-35%",
                    "impact": "high",
                },
                {
                    "category": "Batching",
                    "suggestion": "Batch similar requests to reduce overhead",
                    "estimated_savings": "5-10%",
                    "impact": "low",
                },
            ],
            "total_potential_savings": "40-60%",
        }


# Singleton instance
cost_service = CostService()
