"""Main API router for v1 endpoints."""
from fastapi import APIRouter

from app.api.v1 import metrics, evaluations, incidents, safety, costs

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])
api_router.include_router(evaluations.router, prefix="/evaluations", tags=["Evaluations"])
api_router.include_router(incidents.router, prefix="/incidents", tags=["Incidents"])
api_router.include_router(safety.router, prefix="/safety", tags=["Safety"])
api_router.include_router(costs.router, prefix="/costs", tags=["Costs"])


@api_router.get("/health")
async def health_check():
    """API health check."""
    return {"status": "healthy", "version": "1.0.0"}
