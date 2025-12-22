"""Incident tracking service."""
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from app.schemas.incidents import (
    Incident,
    IncidentStats,
    IncidentType,
    IncidentSeverity,
)


class IncidentService:
    """Service for incident tracking."""

    def __init__(self):
        self._incidents: Dict[str, Incident] = {}
        self._init_demo_incidents()

    def _init_demo_incidents(self):
        """Initialize with demo incidents."""
        demo_incidents = [
            Incident(
                incident_id="inc_001",
                incident_type=IncidentType.GOAL_DRIFT,
                severity=IncidentSeverity.MEDIUM,
                description="Agent deviated from original objective during task execution",
                context={"task_id": "task_42", "drift_score": 0.35},
                resolved=False,
            ),
            Incident(
                incident_id="inc_002",
                incident_type=IncidentType.HALLUCINATION,
                severity=IncidentSeverity.HIGH,
                description="Agent referenced non-existent API endpoint",
                context={"endpoint": "/api/v2/invalid"},
                resolved=True,
                resolution="Added schema validation",
            ),
            Incident(
                incident_id="inc_003",
                incident_type=IncidentType.TIMEOUT,
                severity=IncidentSeverity.LOW,
                description="Task exceeded 30s timeout threshold",
                context={"task_id": "task_15", "duration": 32.5},
                resolved=True,
                resolution="Increased timeout for complex tasks",
            ),
        ]
        for inc in demo_incidents:
            self._incidents[inc.incident_id] = inc

    async def list_incidents(
        self,
        severity: Optional[IncidentSeverity],
        incident_type: Optional[IncidentType],
        resolved: Optional[bool],
        limit: int,
    ) -> List[Incident]:
        """List incidents with optional filters."""
        incidents = list(self._incidents.values())

        if severity:
            incidents = [i for i in incidents if i.severity == severity]
        if incident_type:
            incidents = [i for i in incidents if i.incident_type == incident_type]
        if resolved is not None:
            incidents = [i for i in incidents if i.resolved == resolved]

        return sorted(incidents, key=lambda x: x.created_at, reverse=True)[:limit]

    async def get_stats(self) -> IncidentStats:
        """Get incident statistics."""
        incidents = list(self._incidents.values())

        by_type = {}
        by_severity = {}

        for inc in incidents:
            by_type[inc.incident_type.value] = by_type.get(inc.incident_type.value, 0) + 1
            by_severity[inc.severity.value] = by_severity.get(inc.severity.value, 0) + 1

        unresolved = [i for i in incidents if not i.resolved]
        critical = [i for i in incidents if i.severity == IncidentSeverity.CRITICAL]

        return IncidentStats(
            total_incidents=len(incidents),
            unresolved_count=len(unresolved),
            critical_count=len(critical),
            incident_rate=len(incidents) / 24.0,  # per hour over 24h
            by_type=by_type,
            by_severity=by_severity,
            recent_incidents=sorted(incidents, key=lambda x: x.created_at, reverse=True)[:5],
            trend="stable",
        )

    async def get_pathology_summary(self) -> dict:
        """Get summary of 10 failure pathologies."""
        # 10 pathologies from Section XV
        pathologies = {
            "hallucinated_affordance": {
                "name": "Hallucinated Affordance",
                "description": "Agent claims non-existent capabilities",
                "detected": 2,
                "status": "monitored",
            },
            "state_misestimation": {
                "name": "State Misestimation",
                "description": "Incorrect tracking of environment state",
                "detected": 0,
                "status": "clear",
            },
            "cascading_tool_failure": {
                "name": "Cascading Tool Failure",
                "description": "Error propagation through tool chain",
                "detected": 1,
                "status": "monitored",
            },
            "action_observation_mismatch": {
                "name": "Action-Observation Mismatch",
                "description": "Disconnect between actions and observations",
                "detected": 0,
                "status": "clear",
            },
            "memory_poisoning": {
                "name": "Memory Poisoning",
                "description": "Corrupted memory affecting decisions",
                "detected": 0,
                "status": "clear",
            },
            "feedback_amplification": {
                "name": "Feedback Amplification",
                "description": "Positive feedback loops causing instability",
                "detected": 0,
                "status": "clear",
            },
            "goal_drift": {
                "name": "Goal Drift",
                "description": "Gradual deviation from original objective",
                "detected": 3,
                "status": "warning",
            },
            "planning_myopia": {
                "name": "Planning Myopia",
                "description": "Short-sighted planning decisions",
                "detected": 1,
                "status": "monitored",
            },
            "emergent_collusion": {
                "name": "Emergent Collusion",
                "description": "Unintended coordination between agents",
                "detected": 0,
                "status": "clear",
            },
            "consensus_deadlock": {
                "name": "Consensus Deadlock",
                "description": "Multi-agent agreement failure",
                "detected": 0,
                "status": "clear",
            },
        }
        return {
            "pathologies": pathologies,
            "total_detected": sum(p["detected"] for p in pathologies.values()),
            "warnings": len([p for p in pathologies.values() if p["status"] == "warning"]),
        }

    async def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get a specific incident."""
        return self._incidents.get(incident_id)

    async def resolve_incident(self, incident_id: str, resolution: str) -> Optional[Incident]:
        """Resolve an incident."""
        incident = self._incidents.get(incident_id)
        if incident:
            incident.resolved = True
            incident.resolution = resolution
            incident.resolved_at = datetime.now()
        return incident


# Singleton instance
incident_service = IncidentService()
