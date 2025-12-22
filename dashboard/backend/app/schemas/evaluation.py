"""Evaluation schemas for the dashboard API."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class EvaluationStatusEnum(str, Enum):
    """Evaluation status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationConfig(BaseModel):
    """Configuration for starting an evaluation."""

    name: str = Field(description="Evaluation name")
    model: str = Field(default="phi3:latest", description="Ollama model to use")
    num_tasks: int = Field(default=10, description="Number of tasks to run")
    task_type: str = Field(default="general", description="Type of tasks")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    window_size: int = Field(default=5, description="Rolling window size")


class TaskResult(BaseModel):
    """Result of a single task execution."""

    task_id: str
    success: bool
    cost: float
    duration_seconds: float
    steps_taken: int
    tokens_used: int
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class EvaluationStatus(BaseModel):
    """Current status of an evaluation."""

    evaluation_id: str
    name: str
    status: EvaluationStatusEnum
    progress: float = Field(description="Progress 0.0 to 1.0")
    tasks_completed: int
    tasks_total: int
    current_success_rate: float
    current_cnsr: float
    current_cost: float
    started_at: datetime
    estimated_completion: Optional[datetime] = None
    model: str = Field(default="gemma2:2b", description="Model being evaluated")


class EvaluationResult(BaseModel):
    """Final result of an evaluation."""

    evaluation_id: str
    name: str
    status: EvaluationStatusEnum
    config: EvaluationConfig

    # Metrics
    total_tasks: int
    successful_tasks: int
    success_rate: float
    cnsr: float
    total_cost: float
    mean_cost: float
    total_tokens: int

    # Cost breakdown
    cost_breakdown: Dict[str, float]

    # Task results
    task_results: List[TaskResult] = Field(default_factory=list)

    # Timing
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Recommendation
    recommendation: str = Field(default="PENDING", description="APPROVED, CONDITIONAL, or REJECTED")
    recommendation_reason: str = ""
