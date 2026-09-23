"""Pydantic schemas for API responses."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field("ok", description="Service health status", examples=["ok"])
    service: str = Field("CycloSense AI", description="Service name", examples=["CycloSense AI"])
    version: str = Field("0.1.0", description="API version", examples=["0.1.0"])


class PredictionResult(BaseModel):
    """Core ML/DL prediction outputs and composite risk indicators."""

    classification: str = Field(..., description="Predicted class ('cyclone' or 'non_cyclone')", examples=["cyclone"])
    visual_signal: float = Field(..., description="Image CNN model probability (0.0 to 1.0)", examples=[0.830])
    environmental_signal: float = Field(..., description="Numerical MLP model probability (0.0 to 1.0)", examples=[0.910])
    fusion_signal: float = Field(..., description="Multimodal fusion model probability (0.0 to 1.0)", examples=[0.603])
    development_risk_index: float = Field(..., description="0-100 composite development risk index", examples=[77.5])
    confidence_pct: float = Field(..., description="Model classification confidence percentage", examples=[60.3])
    components: dict[str, float] = Field(..., description="Individual risk index component scores")
    data_source_tag: str = Field(..., description="Dataset provenance tag (e.g. synthetic_mvp)", examples=["synthetic_mvp"])
    note: str = Field(..., description="Cautionary note on score calibration", examples=["Development/Risk Index — not a calibrated probability."])


class VerificationResult(BaseModel):
    """Secondary LLM verification gate status and review findings."""

    gate_triggered: bool = Field(..., description="True if all signals >= 50% threshold", examples=[True])
    llm_called: bool = Field(..., description="True if vision verification LLM was invoked", examples=[True])
    status: str = Field(..., description="Verification policy status code", examples=["VERIFIED_CYCLONE"])
    diagnostic: str = Field(..., description="Internal gate decision diagnostic string", examples=["all_signals_above_threshold_and_llm_confirmed"])
    assessment: str | None = Field(default=None, description="LLM visual assessment ('cyclone_consistent', etc.)", examples=["cyclone_consistent"])
    confidence: float | None = Field(default=None, description="LLM verification confidence (0.0 to 1.0)", examples=[0.92])
    reason: str | None = Field(default=None, description="Detailed visual reasoning from verifier", examples=["Distinct eye and spiral curved convective banding."])
    needs_human_review: bool | None = Field(default=None, description="Flag indicating analyst manual triage is required", examples=[False])


class ExplanationResult(BaseModel):
    """Structured natural-language meteorological explanation."""

    type: str = Field(..., description="Explanation case type ('high_risk_verification' or 'suppressed_or_low_risk')", examples=["high_risk_verification"])
    summary: str = Field(..., description="Executive one-sentence meteorological summary")
    atmospheric_rationale: str = Field(..., description="Physics-based rationale synthesizing prediction matrices")
    primary_drivers: list[str] = Field(default_factory=list, description="List of favorable environmental cyclogenesis factors")
    inhibiting_barriers: list[str] = Field(default_factory=list, description="List of atmospheric suppression factors / hostile barriers")
    operational_guidance: str = Field(..., description="Actionable recommendation for forecaster triage")


class PredictionResponse(BaseModel):
    """Complete prediction, verification, and explanation API response."""

    case_id: str = Field(..., description="Observation or case identifier", examples=["TC-01"])
    prediction: PredictionResult
    verification: VerificationResult
    explanation: ExplanationResult
    timing_ms: dict[str, float] = Field(..., description="Execution duration metrics in milliseconds")


class ErrorResponse(BaseModel):
    """Standardized error response schema."""

    error: str = Field(..., description="Error classification string")
    detail: str = Field(..., description="Human-readable error description")


class ImagePredictionResponse(BaseModel):
    """Response schema for isolated Image CNN inference."""
    
    case_id: str = Field(..., description="Observation or case identifier", examples=["TC-01"])
    visual_signal: float = Field(..., description="Image CNN model probability (0.0 to 1.0)", examples=[0.830])
    timing_ms: float = Field(..., description="Execution duration in ms")


class NumericalPredictionResponse(BaseModel):
    """Response schema for isolated Numerical MLP inference."""
    
    case_id: str = Field(..., description="Observation or case identifier", examples=["TC-01"])
    environmental_signal: float = Field(..., description="Numerical MLP model probability (0.0 to 1.0)", examples=[0.910])
    timing_ms: float = Field(..., description="Execution duration in ms")


class FusionPredictionResponse(BaseModel):
    """Response schema for isolated Late-Fusion inference."""
    
    case_id: str = Field(..., description="Observation or case identifier", examples=["TC-01"])
    fusion_signal: float = Field(..., description="Multimodal fusion model probability (0.0 to 1.0)", examples=[0.603])
    timing_ms: float = Field(..., description="Execution duration in ms")


class RiskIndexResponse(BaseModel):
    """Response schema for isolated Risk Index calculation."""
    
    case_id: str = Field(..., description="Observation or case identifier", examples=["TC-01"])
    development_risk_index: float = Field(..., description="0-100 composite development risk index", examples=[77.5])
    components: dict[str, float] = Field(..., description="Individual risk index component scores")
    timing_ms: float = Field(..., description="Execution duration in ms")
