"""Incident schemas for the dashboard API."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


class IncidentType(str, Enum):
    """Types of incidents (from paper Section XV)."""

    HALLUCINATION = "hallucination"
    GOAL_DRIFT = "goal_drift"
    STATE_MISESTIMATION = "state_misestimation"
    CASCADING_FAILURE = "cascading_failure"
    TOOL_FAILURE = "tool_failure"
    TIMEOUT = "timeout"
    POLICY_VIOLATION = "policy_violation"
    HUMAN_INTERVENTION = "human_intervention"
    GUARDRAIL_ACTIVATION = "guardrail_activation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class IncidentSeverity(str, Enum):
    """Incident severity levels."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Incident(BaseModel):
    """A single incident record."""

    incident_id: str
    incident_type: IncidentType
    severity: IncidentSeverity
    description: str
    context: Dict = Field(default_factory=dict)
    resolved: bool = False
    resolution: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None


class IncidentStats(BaseModel):
    """Incident statistics summary."""

    total_incidents: int = 0
    unresolved_count: int = 0
    critical_count: int = 0
    incident_rate: float = Field(default=0.0, description="Incidents per hour")

    by_type: Dict[str, int] = Field(default_factory=dict)
    by_severity: Dict[str, int] = Field(default_factory=dict)

    recent_incidents: List[Incident] = Field(default_factory=list)
    trend: str = Field(default="stable", description="stable, increasing, or decreasing")
