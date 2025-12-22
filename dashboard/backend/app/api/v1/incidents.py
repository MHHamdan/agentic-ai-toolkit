"""Incidents API endpoints."""
from fastapi import APIRouter, Query
from typing import List, Optional

from app.schemas.incidents import Incident, IncidentStats, IncidentType, IncidentSeverity
from app.services.incident_service import incident_service

router = APIRouter()


@router.get("/", response_model=List[Incident])
@router.get("", response_model=List[Incident], include_in_schema=False)
async def list_incidents(
    severity: Optional[IncidentSeverity] = None,
    incident_type: Optional[IncidentType] = None,
    resolved: Optional[bool] = None,
    limit: int = Query(50, ge=1, le=200),
):
    """List incidents with optional filters."""
    return await incident_service.list_incidents(severity, incident_type, resolved, limit)


@router.get("/stats", response_model=IncidentStats)
async def get_incident_stats():
    """Get incident statistics summary."""
    return await incident_service.get_stats()


@router.get("/pathologies")
async def get_pathology_summary():
    """Get summary of 10 failure pathologies (Section XV)."""
    return await incident_service.get_pathology_summary()


@router.get("/{incident_id}", response_model=Incident)
async def get_incident(incident_id: str):
    """Get a specific incident."""
    return await incident_service.get_incident(incident_id)


@router.patch("/{incident_id}/resolve")
async def resolve_incident(incident_id: str, resolution: str):
    """Mark an incident as resolved."""
    return await incident_service.resolve_incident(incident_id, resolution)
