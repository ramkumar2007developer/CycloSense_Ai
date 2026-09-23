"""Backend configuration and settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _parse_cors_origins(raw: str | None) -> list[str]:
    default = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    if not raw:
        return default
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@dataclass(frozen=True)
class Settings:
    """Core settings for the FastAPI backend."""

    service_name: str = "CycloSense AI"
    version: str = "0.1.0"
    api_prefix: str = ""
    max_upload_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    allowed_extensions: set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png"}
    )
    allowed_content_types: set[str] = field(
        default_factory=lambda: {
            "image/jpeg",
            "image/png",
            "image/jpg",
            "application/octet-stream",
        }
    )
    cors_origins: list[str] = field(
        default_factory=lambda: _parse_cors_origins(
            os.environ.get("CYCLOSENSE_CORS_ORIGINS")
        )
    )


settings = Settings()
