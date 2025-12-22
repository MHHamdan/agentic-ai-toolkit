"""Safety and compliance API endpoints."""
from fastapi import APIRouter

from app.schemas.safety import ComplianceStatus, AutonomyCriteria
from app.services.safety_service import safety_service

router = APIRouter()


@router.get("/compliance", response_model=ComplianceStatus)
async def get_compliance_status():
    """Get overall safety compliance status."""
    return await safety_service.get_compliance_status()


@router.get("/requirements")
async def get_safety_requirements():
    """Get the 5 safety requirements and their status."""
    return await safety_service.get_requirements()


@router.get("/autonomy")
async def get_autonomy_assessment():
    """Get autonomy level assessment with 4 criteria."""
    return await safety_service.get_autonomy_assessment()


@router.get("/audit")
async def get_audit_log(limit: int = 100):
    """Get recent audit log entries."""
    return await safety_service.get_audit_log(limit)


@router.post("/check")
async def run_safety_check():
    """Run a fresh safety compliance check."""
    return await safety_service.run_compliance_check()
