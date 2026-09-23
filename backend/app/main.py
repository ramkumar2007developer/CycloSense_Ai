"""FastAPI application entrypoint for CycloSense AI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.prediction import router as prediction_router
from backend.app.core.config import settings
from backend.app.dependencies import get_prediction_service

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cyclose_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Loads and warms up the ML/DL models once at startup so requests
    do not incur disk-loading penalties or repeated initialization.
    """
    logger.info("Initializing CycloSense AI backend service (v%s)...", settings.version)
    try:
        # Pre-warm prediction service and load models once
        service = get_prediction_service()
        logger.info(
            "ML models and preprocessors successfully loaded into memory. Ready for inference."
        )
    except Exception as exc:
        logger.error("Failed to preload ML models during startup: %s", exc, exc_info=True)

    yield

    logger.info("CycloSense AI backend service shutting down.")


app = FastAPI(
    title=f"{settings.service_name} API",
    version=settings.version,
    description=(
        "Production-grade REST API exposing CycloSense AI's multimodal prediction pipeline. "
        "Integrates satellite infrared imagery analysis (CNN), numerical atmospheric indices (MLP), "
        "late-fusion risk scoring (0–100 Development/Risk Index), secondary LLM Vision Verification Gate "
        "(evaluating per-sample signals against the 50% threshold), and LLM-generated meteorological explanations."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure Cross-Origin Resource Sharing (CORS) for approved frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(prediction_router, prefix=settings.api_prefix)
