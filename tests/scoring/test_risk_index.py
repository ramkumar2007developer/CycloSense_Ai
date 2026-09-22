"""Scoring and risk index tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.exceptions import ValidationError
from src.scoring.risk_index import ScoringWeights, compute_risk_index


@pytest.fixture
def default_weights() -> ScoringWeights:
    return ScoringWeights(0.4, 0.3, 0.1, 0.1, 0.05, 0.05)


def test_risk_index_in_range(default_weights: ScoringWeights) -> None:
    result = compute_risk_index(
        np.array([0.2, 0.8]),
        {
            "cloud_organization_index": 0.5,
            "rotation_index": 0.3,
            "persistence_index": 0.2,
            "convection_index": 0.4,
            "relative_humidity_pct": 70,
        },
        default_weights,
    )
    assert 0.0 <= result.development_risk_index <= 100.0
    assert result.classification == "cyclone"
    assert result.confidence_pct == pytest.approx(80.0)


def test_exact_zero_and_hundred(default_weights: ScoringWeights) -> None:
    low = compute_risk_index(
        np.array([1.0, 0.0]),
        {
            "relative_humidity_pct": 0,
            "cloud_organization_index": 0,
            "convection_index": 0,
            "rotation_index": 0,
            "persistence_index": 0,
        },
        default_weights,
    )
    high = compute_risk_index(
        np.array([0.0, 1.0]),
        {
            "relative_humidity_pct": 100,
            "cloud_organization_index": 1,
            "convection_index": 1,
            "rotation_index": 1,
            "persistence_index": 1,
        },
        default_weights,
    )
    assert low.development_risk_index == pytest.approx(0.0)
    assert high.development_risk_index == pytest.approx(100.0)
    assert low.classification == "non_cyclone"
    assert high.classification == "cyclone"


def test_nan_probability_rejected(default_weights: ScoringWeights) -> None:
    with pytest.raises(ValidationError, match="NaN"):
        compute_risk_index(np.array([np.nan, 0.5]), {}, default_weights)


def test_invalid_shape_rejected(default_weights: ScoringWeights) -> None:
    with pytest.raises(ValidationError, match="Expected binary class probabilities"):
        compute_risk_index(np.array([0.1, 0.2, 0.7]), {}, default_weights)


def test_monotonicity_visual_and_environmental(default_weights: ScoringWeights) -> None:
    """Increasing cyclone probability or favorable env features must strictly not decrease risk index."""
    base_env = {
        "cloud_organization_index": 0.3,
        "rotation_index": 0.2,
        "persistence_index": 0.2,
        "convection_index": 0.3,
        "relative_humidity_pct": 50,
    }
    r1 = compute_risk_index(np.array([0.8, 0.2]), base_env, default_weights)
    r2 = compute_risk_index(np.array([0.5, 0.5]), base_env, default_weights)
    r3 = compute_risk_index(np.array([0.2, 0.8]), base_env, default_weights)
    assert r1.development_risk_index < r2.development_risk_index < r3.development_risk_index

    # Env increase
    high_env = dict(base_env, relative_humidity_pct=95, convection_index=0.9)
    r4 = compute_risk_index(np.array([0.2, 0.8]), high_env, default_weights)
    assert r4.development_risk_index > r3.development_risk_index


def test_component_scores_bounded(default_weights: ScoringWeights) -> None:
    result = compute_risk_index(
        np.array([0.3, 0.7]),
        {
            "cloud_organization_index": 1.5,  # Unclamped in input, must be clamped internally
            "rotation_index": -0.2,
            "persistence_index": 0.5,
            "convection_index": 0.8,
            "relative_humidity_pct": 120,
        },
        default_weights,
    )
    for name, val in result.components.as_dict().items():
        assert 0.0 <= val <= 1.0, f"Component {name} out of [0, 1] range: {val}"


def test_as_dict_metadata(default_weights: ScoringWeights) -> None:
    result = compute_risk_index(np.array([0.4, 0.6]), {}, default_weights, data_source_tag="synthetic_mvp")
    d = result.as_dict()
    assert d["data_source_tag"] == "synthetic_mvp"
    assert "not a calibrated probability" in str(d["note"])
    assert "components" in d
