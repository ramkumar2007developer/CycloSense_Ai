"""API routes module."""

from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.prediction import router as prediction_router

__all__ = ["health_router", "prediction_router"]
