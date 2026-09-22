"""Secondary LLM/Vision Verification Gate and Decision Policy."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.config.ml_config import LLMVerificationConfig
from src.exceptions import ValidationError
from src.inference.llm_verifier import (
    LLMVerificationError,
    LLMVerificationResult,
    LLMVisionVerifier,
)

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    """Documented states for the secondary verification policy."""

    VERIFIED_CYCLONE = "VERIFIED_CYCLONE"
    SECONDARY_REVIEW_CONFLICT = "SECONDARY_REVIEW_CONFLICT"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    LLM_NOT_REQUIRED = "LLM_NOT_REQUIRED"
    LLM_VERIFICATION_FAILED = "LLM_VERIFICATION_FAILED"


@dataclass(frozen=True)
class GateSignals:
    """Per-sample ML probabilities extracted from CycloSense models."""

    image_probability: float
    numerical_probability: float
    fusion_probability: float

    def as_dict(self) -> dict[str, float]:
        return {
            "image_probability": self.image_probability,
            "numerical_probability": self.numerical_probability,
            "fusion_probability": self.fusion_probability,
        }

    def validate(self) -> None:
        for name, val in self.as_dict().items():
            if val is None or not (0.0 <= val <= 1.0):
                raise ValidationError(f"Signal {name} must be a probability in [0.0, 1.0], got {val}")


@dataclass(frozen=True)
class VerificationGateResult:
    """Complete result from the verification gate."""

    signals: GateSignals
    gate_threshold: float
    all_signals_above_threshold: bool
    llm_called: bool
    final_status: VerificationStatus
    reason: str
    llm_result: LLMVerificationResult | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "image_probability": self.signals.image_probability,
            "numerical_probability": self.signals.numerical_probability,
            "fusion_probability": self.signals.fusion_probability,
            "gate_threshold": self.gate_threshold,
            "all_signals_above_threshold": self.all_signals_above_threshold,
            "all_signals_above_50": self.all_signals_above_threshold,
            "llm_called": self.llm_called,
            "final_status": self.final_status.value,
            "reason": self.reason,
        }
        if self.llm_result:
            data["llm_assessment"] = self.llm_result.visual_assessment
            data["llm_confidence"] = self.llm_result.confidence
            data["llm_reason"] = self.llm_result.reason
            data["needs_human_review"] = self.llm_result.needs_human_review
        return data

    def to_log_entry(self, image_id: str | None = None) -> dict[str, Any]:
        """Produce structured log record (free of any secrets)."""
        entry: dict[str, Any] = {
            "image_id": image_id or "unknown",
            "image_probability": round(self.signals.image_probability, 4),
            "numerical_probability": round(self.signals.numerical_probability, 4),
            "fusion_probability": round(self.signals.fusion_probability, 4),
            "all_signals_above_50": self.all_signals_above_threshold,
            "llm_called": self.llm_called,
            "final_status": self.final_status.value,
        }
        if self.llm_called and self.llm_result:
            entry["llm_assessment"] = self.llm_result.visual_assessment
            entry["llm_confidence"] = round(self.llm_result.confidence, 4)
        elif not self.llm_called:
            entry["reason"] = self.reason
        elif self.final_status == VerificationStatus.LLM_VERIFICATION_FAILED:
            entry["reason"] = self.reason
        return entry


def evaluate_verification_gate(
    signals: GateSignals,
    image_path: Path,
    verifier: LLMVisionVerifier | None,
    config: LLMVerificationConfig,
    environmental_context: dict[str, float] | None = None,
    image_id: str | None = None,
) -> VerificationGateResult:
    """Evaluate the secondary LLM verification gate using the documented policy.

    Logic:
    - If ALL signals >= gate_threshold (default 0.50):
        Call verifier.
        - LLM says cyclone_consistent AND confidence >= confidence_threshold -> VERIFIED_CYCLONE
        - LLM says non_cyclone_consistent -> SECONDARY_REVIEW_CONFLICT
        - LLM says uncertain OR confidence < confidence_threshold OR needs_human_review -> HUMAN_REVIEW_REQUIRED
        - LLM failure (timeout / API error / parse failure) -> LLM_VERIFICATION_FAILED (preserves ML output)
    - If ANY signal < gate_threshold:
        DO NOT call verifier -> LLM_NOT_REQUIRED
    """
    signals.validate()
    gate_thresh = config.gate_threshold
    conf_thresh = config.confidence_threshold

    # Find which signals fail if any
    below = [k for k, v in signals.as_dict().items() if v < gate_thresh]
    all_above = len(below) == 0

    if not all_above:
        # Determine specific reason string
        if len(below) == 1:
            reason = f"{below[0]}_below_threshold"
        else:
            reason = f"signals_below_threshold_{'_and_'.join(below)}"

        result = VerificationGateResult(
            signals=signals,
            gate_threshold=gate_thresh,
            all_signals_above_threshold=False,
            llm_called=False,
            final_status=VerificationStatus.LLM_NOT_REQUIRED,
            reason=reason,
            llm_result=None,
        )
        logger.info("Verification gate: %s", json.dumps(result.to_log_entry(image_id)))
        return result

    # Gate activated: all signals >= gate_threshold
    if verifier is None:
        # If no verifier supplied and gate triggers, mark as human review required
        result = VerificationGateResult(
            signals=signals,
            gate_threshold=gate_thresh,
            all_signals_above_threshold=True,
            llm_called=False,
            final_status=VerificationStatus.HUMAN_REVIEW_REQUIRED,
            reason="all_signals_above_threshold_but_no_verifier_configured",
            llm_result=None,
        )
        logger.warning("Verification gate triggered with no verifier: %s", json.dumps(result.to_log_entry(image_id)))
        return result

    # Prepare ML context without training/eval metrics
    ml_context: dict[str, Any] = {
        "image_probability": round(signals.image_probability, 4),
        "numerical_probability": round(signals.numerical_probability, 4),
        "fusion_probability": round(signals.fusion_probability, 4),
    }
    if environmental_context:
        ml_context["environmental_context"] = environmental_context

    try:
        llm_result = verifier.verify_image(image_path, ml_context)
    except (LLMVerificationError, Exception) as exc:
        logger.error("LLM secondary verification failed for image %s: %s", image_path, exc)
        result = VerificationGateResult(
            signals=signals,
            gate_threshold=gate_thresh,
            all_signals_above_threshold=True,
            llm_called=True,
            final_status=VerificationStatus.LLM_VERIFICATION_FAILED,
            reason=f"verifier_error: {exc}",
            llm_result=None,
        )
        logger.info("Verification gate (failed): %s", json.dumps(result.to_log_entry(image_id)))
        return result

    # Evaluate decision policy
    assessment = llm_result.visual_assessment
    confidence = llm_result.confidence

    if assessment == "cyclone_consistent" and confidence >= conf_thresh and not llm_result.needs_human_review:
        final_status = VerificationStatus.VERIFIED_CYCLONE
        reason = "all_signals_above_threshold_and_llm_confirmed"
    elif assessment == "non_cyclone_consistent":
        final_status = VerificationStatus.SECONDARY_REVIEW_CONFLICT
        reason = "all_signals_above_threshold_but_llm_identified_non_cyclone"
    elif assessment == "uncertain" or llm_result.needs_human_review:
        final_status = VerificationStatus.HUMAN_REVIEW_REQUIRED
        reason = "all_signals_above_threshold_and_llm_uncertain"
    elif assessment == "cyclone_consistent" and confidence < conf_thresh:
        final_status = VerificationStatus.HUMAN_REVIEW_REQUIRED
        reason = f"llm_confidence_{confidence:.2f}_below_threshold_{conf_thresh:.2f}"
    else:
        final_status = VerificationStatus.HUMAN_REVIEW_REQUIRED
        reason = f"unhandled_assessment_{assessment}"

    gate_result = VerificationGateResult(
        signals=signals,
        gate_threshold=gate_thresh,
        all_signals_above_threshold=True,
        llm_called=True,
        final_status=final_status,
        reason=reason,
        llm_result=llm_result,
    )
    logger.info("Verification gate: %s", json.dumps(gate_result.to_log_entry(image_id)))
    return gate_result
