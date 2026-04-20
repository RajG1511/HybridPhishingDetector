"""
API Route — Health Check

GET /health — Returns service liveness status and version string.
Used by load balancers and container orchestration health checks.
"""

from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service liveness.

    Returns:
        HealthResponse with status "ok" and current version.
    """
    return HealthResponse(status="ok", version="0.1.0")
