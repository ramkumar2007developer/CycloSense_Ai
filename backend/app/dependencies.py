"""FastAPI dependency injection utilities."""

from __future__ import annotations

import logging
from typing import Generator

from backend.app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

_prediction_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    """Dependency that returns the shared singleton PredictionService instance."""
    global _prediction_service
    if _prediction_service is None:
        logger.info("Initializing shared PredictionService singleton...")
        _prediction_service = PredictionService()
    return _prediction_service


def set_prediction_service(service: PredictionService | None) -> None:
    """Explicitly set or reset the PredictionService instance (used for testing)."""
    global _prediction_service
    _prediction_service = service
