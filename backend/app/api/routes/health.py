"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.schemas.response import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health and operational availability of the CycloSense AI API service.",
)
async def health_check() -> HealthResponse:
    """Return basic service health status without triggering ML inference or model reloads."""
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.version,
    )
