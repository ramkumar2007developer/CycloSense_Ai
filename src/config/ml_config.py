"""Load ML configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.paths import find_project_root, resolve_path
from src.exceptions import ConfigurationError

_ENV_EXTERNAL_ROOT = "CYCLOSENSE_EXTERNAL_DATA_ROOT"
_DEFAULT = Path("configs") / "ml_config.json"


@dataclass(frozen=True)
class LLMVerificationConfig:
    enabled: bool
    provider: str
    model_name: str
    gate_threshold: float
    confidence_threshold: float
    timeout_seconds: float
    max_retries: int

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None = None) -> LLMVerificationConfig:
        raw = raw or {}
        # Environment variable overrides
        env_enabled = os.environ.get("CYCLOSENSE_LLM_VERIFICATION_ENABLED", os.environ.get("LLM_VERIFICATION_ENABLED"))
        if env_enabled is not None:
            enabled = env_enabled.lower() in ("true", "1", "yes")
        else:
            enabled = bool(raw.get("enabled", False))

        env_provider = os.environ.get("CYCLOSENSE_LLM_PROVIDER", os.environ.get("LLM_PROVIDER"))
        provider = str(env_provider or raw.get("provider", "mock"))

        env_model = os.environ.get("CYCLOSENSE_LLM_MODEL", os.environ.get("LLM_MODEL"))
        model_name = str(env_model or raw.get("model_name", "llama-3.2-11b-vision-preview"))

        env_gate = os.environ.get("CYCLOSENSE_LLM_GATE_THRESHOLD", os.environ.get("LLM_GATE_THRESHOLD"))
        gate_threshold = float(env_gate if env_gate is not None else raw.get("gate_threshold", 0.50))

        env_conf = os.environ.get("CYCLOSENSE_LLM_CONFIDENCE_THRESHOLD", os.environ.get("LLM_CONFIDENCE_THRESHOLD"))
        confidence_threshold = float(env_conf if env_conf is not None else raw.get("confidence_threshold", 0.70))

        timeout_seconds = float(raw.get("timeout_seconds", 15.0))
        max_retries = int(raw.get("max_retries", 2))

        return cls(
            enabled=enabled,
            provider=provider,
            model_name=model_name,
            gate_threshold=gate_threshold,
            confidence_threshold=confidence_threshold,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )


@dataclass(frozen=True)
class LLMExplanationConfig:
    enabled: bool
    provider: str
    model_name: str
    timeout_seconds: float
    max_retries: int

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None = None) -> LLMExplanationConfig:
        raw = raw or {}
        env_enabled = os.environ.get("CYCLOSENSE_LLM_EXPLANATION_ENABLED", os.environ.get("LLM_EXPLANATION_ENABLED"))
        if env_enabled is not None:
            enabled = env_enabled.lower() in ("true", "1", "yes")
        else:
            enabled = bool(raw.get("enabled", True))

        env_provider = os.environ.get("CYCLOSENSE_LLM_EXPLANATION_PROVIDER", os.environ.get("LLM_EXPLANATION_PROVIDER"))
        provider = str(env_provider or raw.get("provider", "mock"))

        env_model = os.environ.get("CYCLOSENSE_LLM_EXPLANATION_MODEL", os.environ.get("LLM_EXPLANATION_MODEL"))
        model_name = str(env_model or raw.get("model_name", "llama-3.3-70b-versatile"))

        timeout_seconds = float(raw.get("timeout_seconds", 15.0))
        max_retries = int(raw.get("max_retries", 2))

        return cls(
            enabled=enabled,
            provider=provider,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )


@dataclass(frozen=True)
class MLConfig:
    project_root: Path
    external_data_root: Path
    raw: dict[str, Any]

    @property
    def data_source_tag(self) -> str:
        return str(self.raw.get("data_source_tag", "synthetic_mvp"))

    @property
    def llm_verification(self) -> LLMVerificationConfig:
        return LLMVerificationConfig.from_raw(self.raw.get("llm_verification"))

    @property
    def llm_explanation(self) -> LLMExplanationConfig:
        return LLMExplanationConfig.from_raw(self.raw.get("llm_explanation"))

    def path(self, *parts: str) -> Path:
        return resolve_path(self.external_data_root, Path(*parts))


def load_ml_config(
    config_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> MLConfig:
    root = Path(project_root).resolve() if project_root else find_project_root()
    resolved = Path(config_path) if config_path else root / _DEFAULT
    if not resolved.is_absolute():
        resolved = root / resolved
    if not resolved.is_file():
        raise ConfigurationError(f"ML config not found: {resolved}")

    with resolved.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    external = os.environ.get(_ENV_EXTERNAL_ROOT, raw.get("external_data_root"))
    if not external:
        # Default to Downloads location discovered during audit
        default_downloads = Path.home() / "Downloads" / "dataset-main" / "dataset-main"
        env_csv = Path.home() / "Downloads" / "cyclosense_mvp_environmental_dataset.csv"
        if default_downloads.is_dir():
            external = str(default_downloads)
        elif env_csv.is_file():
            external = str(env_csv.parent)
        else:
            raise ConfigurationError(
                f"Set {_ENV_EXTERNAL_ROOT} or external_data_root in ml_config.json"
            )

    external_root = resolve_path(root, external).resolve()
    return MLConfig(project_root=root, external_data_root=external_root, raw=raw)
