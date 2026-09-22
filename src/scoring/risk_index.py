"""CycloSense Development/Risk Index (0–100, not calibrated probability)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.exceptions import ValidationError


@dataclass(frozen=True)
class ScoringWeights:
    visual_weight: float
    environmental_weight: float
    convection_weight: float
    moisture_weight: float
    organization_weight: float
    persistence_weight: float

    @classmethod
    def from_config(cls, cfg: dict[str, float]) -> ScoringWeights:
        return cls(
            visual_weight=float(cfg["visual_weight"]),
            environmental_weight=float(cfg["environmental_weight"]),
            convection_weight=float(cfg["convection_weight"]),
            moisture_weight=float(cfg["moisture_weight"]),
            organization_weight=float(cfg["organization_weight"]),
            persistence_weight=float(cfg["persistence_weight"]),
        )


@dataclass(frozen=True)
class ComponentScores:
    visual: float
    environmental: float
    convection: float
    moisture: float
    organization: float
    persistence: float

    def as_dict(self) -> dict[str, float]:
        return {
            "visual": self.visual,
            "environmental": self.environmental,
            "convection": self.convection,
            "moisture": self.moisture,
            "organization": self.organization,
            "persistence": self.persistence,
        }


@dataclass(frozen=True)
class RiskIndexResult:
    classification: str
    confidence_pct: float
    development_risk_index: float
    components: ComponentScores
    data_source_tag: str

    def as_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "confidence_pct": self.confidence_pct,
            "development_risk_index": self.development_risk_index,
            "components": self.components.as_dict(),
            "data_source_tag": self.data_source_tag,
            "note": "Development/Risk Index — not a calibrated probability.",
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def compute_risk_index(
    class_probabilities: np.ndarray,
    env_features: dict[str, float],
    weights: ScoringWeights,
    data_source_tag: str = "synthetic_mvp",
) -> RiskIndexResult:
    """Compute transparent 0–100 development/risk index from model outputs + env features."""
    if class_probabilities.shape[-1] != 2:
        raise ValidationError("Expected binary class probabilities.")

    if np.isnan(class_probabilities).any():
        raise ValidationError("NaN detected in class probabilities.")

    probs = class_probabilities.astype(float)
    probs = probs / probs.sum() if probs.sum() > 0 else probs
    cyclone_prob = float(probs[1])
    predicted_class = int(cyclone_prob >= 0.5)
    confidence_pct = float(max(cyclone_prob, 1.0 - cyclone_prob) * 100.0)

    visual = _clamp(cyclone_prob)
    environmental = _clamp(
        (
            env_features.get("cloud_organization_index", 0.0)
            + env_features.get("rotation_index", 0.0)
            + env_features.get("persistence_index", 0.0)
        )
        / 3.0
    )
    convection = _clamp(env_features.get("convection_index", 0.0))
    moisture = _clamp(env_features.get("relative_humidity_pct", 0.0) / 100.0)
    organization = _clamp(env_features.get("cloud_organization_index", 0.0))
    persistence = _clamp(env_features.get("persistence_index", 0.0))

    components = ComponentScores(
        visual=visual,
        environmental=environmental,
        convection=convection,
        moisture=moisture,
        organization=organization,
        persistence=persistence,
    )

    weighted = (
        weights.visual_weight * visual
        + weights.environmental_weight * environmental
        + weights.convection_weight * convection
        + weights.moisture_weight * moisture
        + weights.organization_weight * organization
        + weights.persistence_weight * persistence
    )
    weight_sum = (
        weights.visual_weight
        + weights.environmental_weight
        + weights.convection_weight
        + weights.moisture_weight
        + weights.organization_weight
        + weights.persistence_weight
    )
    normalized = weighted / weight_sum if weight_sum else 0.0
    risk_index = _clamp(normalized, 0.0, 1.0) * 100.0

    return RiskIndexResult(
        classification="cyclone" if predicted_class == 1 else "non_cyclone",
        confidence_pct=confidence_pct,
        development_risk_index=risk_index,
        components=components,
        data_source_tag=data_source_tag,
    )
