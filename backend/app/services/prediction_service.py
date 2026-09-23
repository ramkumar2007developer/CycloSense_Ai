"""Prediction service orchestrating inference through the existing ML/DL & LLM pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.schemas.response import (
    ExplanationResult,
    PredictionResponse,
    PredictionResult,
    VerificationResult,
    ImagePredictionResponse,
    NumericalPredictionResponse,
    FusionPredictionResponse,
    RiskIndexResponse,
)
from src.inference.llm_explainer import LLMExplainer
from src.inference.llm_verifier import LLMVisionVerifier
from src.inference.predictor import CycloSensePredictor

logger = logging.getLogger(__name__)


class PredictionService:
    """Service wrapping the existing CycloSensePredictor pipeline."""

    def __init__(self, predictor: CycloSensePredictor | None = None) -> None:
        if predictor is not None:
            self._predictor = predictor
        else:
            logger.info("Initializing CycloSensePredictor from project root...")
            self._predictor = CycloSensePredictor.from_project()
            logger.info("CycloSensePredictor initialized successfully.")

    @property
    def predictor(self) -> CycloSensePredictor:
        return self._predictor

    def predict(
        self,
        image_path: Path,
        numerical_features: dict[str, float],
        case_id: str = "custom",
        verifier: LLMVisionVerifier | None = None,
        explainer: LLMExplainer | None = None,
    ) -> PredictionResponse:
        """Run complete ML prediction, verification gate, and LLM explanation.

        Args:
            image_path: Path to the validated satellite image file on disk.
            numerical_features: Dictionary of the 12 numerical features.
            case_id: Optional tracking identifier.
            verifier: Optional LLMVisionVerifier override (used in unit tests).
            explainer: Optional LLMExplainer override (used in unit tests).

        Returns:
            Structured PredictionResponse with ML, verification, and explanation details.
        """
        logger.info("Prediction requested for case_id=%s, image=%s", case_id, image_path.name)
        t_start = time.perf_counter()

        env_series = pd.Series(numerical_features)

        # Call existing predictor pipeline
        raw_result = self._predictor.predict_with_explanation(
            image_path=image_path,
            env_row=env_series,
            verifier=verifier,
            explainer=explainer,
            image_id=case_id,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Prediction completed for case_id=%s in %.2f ms (risk=%.1f, status=%s)",
            case_id,
            elapsed_ms,
            raw_result.get("development_risk_index", 0.0),
            raw_result.get("final_status", "UNKNOWN"),
        )

        # Map ML outputs
        prediction = PredictionResult(
            classification=str(raw_result.get("classification", "non_cyclone")),
            visual_signal=float(raw_result.get("image_probability", 0.0)),
            environmental_signal=float(raw_result.get("numerical_probability", 0.0)),
            fusion_signal=float(raw_result.get("fusion_probability", 0.0)),
            development_risk_index=float(raw_result.get("development_risk_index", 0.0)),
            confidence_pct=float(raw_result.get("confidence_pct", 0.0)),
            components=dict(raw_result.get("components", {})),
            data_source_tag=str(raw_result.get("data_source_tag", "synthetic_mvp")),
            note=str(raw_result.get("note", "Development/Risk Index — not a calibrated probability.")),
        )

        # Map Verification outputs
        verification = VerificationResult(
            gate_triggered=bool(raw_result.get("all_signals_above_50", False)),
            llm_called=bool(raw_result.get("llm_called", False)),
            status=str(raw_result.get("final_status", "LLM_NOT_REQUIRED")),
            diagnostic=str(raw_result.get("reason", "")),
            assessment=raw_result.get("llm_assessment"),
            confidence=raw_result.get("llm_confidence"),
            reason=raw_result.get("llm_reason"),
            needs_human_review=raw_result.get("needs_human_review"),
        )

        # Map Explanation outputs
        raw_expl = raw_result.get("llm_explanation", {})
        explanation = ExplanationResult(
            type=str(raw_expl.get("case_type", "suppressed_or_low_risk")),
            summary=str(raw_expl.get("summary", "")),
            atmospheric_rationale=str(raw_expl.get("meteorological_rationale", "")),
            primary_drivers=list(raw_expl.get("primary_drivers", [])),
            inhibiting_barriers=list(raw_expl.get("inhibiting_factors", [])),
            operational_guidance=str(raw_expl.get("operational_guidance", "")),
        )

        return PredictionResponse(
            case_id=case_id,
            prediction=prediction,
            verification=verification,
            explanation=explanation,
            timing_ms={"total_ms": round(elapsed_ms, 2)},
        )

    def predict_image(self, image_path: Path, case_id: str = "custom") -> ImagePredictionResponse:
        """Run isolated image inference."""
        t_start = time.perf_counter()
        prob = self._predictor.predict_image(image_path)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return ImagePredictionResponse(
            case_id=case_id,
            visual_signal=float(prob),
            timing_ms=round(elapsed_ms, 2)
        )

    def predict_numerical(self, numerical_features: dict[str, float], case_id: str = "custom") -> NumericalPredictionResponse:
        """Run isolated numerical inference."""
        t_start = time.perf_counter()
        env_series = pd.Series(numerical_features)
        prob = self._predictor.predict_numerical(env_series)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return NumericalPredictionResponse(
            case_id=case_id,
            environmental_signal=float(prob),
            timing_ms=round(elapsed_ms, 2)
        )

    def predict_fusion(self, image_path: Path, numerical_features: dict[str, float], case_id: str = "custom") -> FusionPredictionResponse:
        """Run isolated late-fusion inference."""
        t_start = time.perf_counter()
        env_series = pd.Series(numerical_features)
        prob = self._predictor.predict_fusion(image_path, env_series)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return FusionPredictionResponse(
            case_id=case_id,
            fusion_signal=float(prob),
            timing_ms=round(elapsed_ms, 2)
        )

    def calculate_risk_index(self, image_path: Path, numerical_features: dict[str, float], case_id: str = "custom") -> RiskIndexResponse:
        """Calculate isolated risk index and components."""
        t_start = time.perf_counter()
        env_series = pd.Series(numerical_features)
        
        result_dict = self._predictor.predict(image_path, env_series)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        
        return RiskIndexResponse(
            case_id=case_id,
            development_risk_index=float(result_dict["development_risk_index"]),
            components=result_dict["components"],
            timing_ms=round(elapsed_ms, 2)
        )
