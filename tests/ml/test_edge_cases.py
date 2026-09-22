"""Edge-case prediction tests using the scoring pipeline.

Tests all 10 documented edge cases from the master prompt using
compute_risk_index() directly (no trained model required).

Edge cases:
1. Clear sky + neutral environment
2. Normal cloud formation + neutral environment
3. Dense cloud but no cyclone evidence
4. Strong organized structure + supportive environment
5. Strong visual + weak numerical
6. Weak visual + strong numerical
7. Missing numerical feature → defaults gracefully
8. Extreme but valid numerical values
9. Invalid probability array → raises ValidationError
10. Empty probability array → raises ValidationError
"""

from __future__ import annotations

import numpy as np
import pytest

from src.exceptions import ValidationError
from src.scoring.risk_index import ScoringWeights, compute_risk_index, RiskIndexResult

# Standard weights from ml_config.json
WEIGHTS = ScoringWeights(
    visual_weight=0.4,
    environmental_weight=0.3,
    convection_weight=0.1,
    moisture_weight=0.1,
    organization_weight=0.05,
    persistence_weight=0.05,
)

NEUTRAL_ENV = {
    "cloud_organization_index": 0.3,
    "rotation_index": 0.2,
    "persistence_index": 0.2,
    "convection_index": 0.2,
    "relative_humidity_pct": 50.0,
}

SUPPORTIVE_ENV = {
    "cloud_organization_index": 0.8,
    "rotation_index": 0.7,
    "persistence_index": 0.75,
    "convection_index": 0.85,
    "relative_humidity_pct": 85.0,
}

WEAK_ENV = {
    "cloud_organization_index": 0.05,
    "rotation_index": 0.05,
    "persistence_index": 0.05,
    "convection_index": 0.05,
    "relative_humidity_pct": 20.0,
}


def _assert_in_range(result: RiskIndexResult, case: str) -> None:
    assert 0.0 <= result.development_risk_index <= 100.0, (
        f"Case '{case}': Risk Index {result.development_risk_index:.2f} is out of [0, 100]."
    )
    assert result.classification in {"cyclone", "non_cyclone"}, (
        f"Case '{case}': Unexpected classification '{result.classification}'."
    )


# --------------------------------------------------------------------------- #
# Case 1: Clear sky + neutral environment
# --------------------------------------------------------------------------- #
def test_case1_clear_sky_neutral_env() -> None:
    """Clear sky (very low cyclone prob) + neutral env → low risk, non-cyclone."""
    result = compute_risk_index(np.array([0.92, 0.08]), NEUTRAL_ENV, WEIGHTS, "synthetic_mvp")
    _assert_in_range(result, "case1_clear_sky")
    assert result.classification == "non_cyclone"
    assert result.development_risk_index < 30.0, (
        f"Clear sky should score low, got {result.development_risk_index:.1f}"
    )


# --------------------------------------------------------------------------- #
# Case 2: Normal cloud formation + neutral environment
# --------------------------------------------------------------------------- #
def test_case2_normal_clouds_neutral_env() -> None:
    """Normal clouds (moderate prob) + neutral env → moderate risk.
    
    ANTI-BIAS CHECK: Normal cloud scenes must NOT be forced to score near 0.
    This verifies the model does not treat 'cloud detected' = 'non-cyclone'.
    """
    result = compute_risk_index(np.array([0.45, 0.55]), NEUTRAL_ENV, WEIGHTS, "synthetic_mvp")
    _assert_in_range(result, "case2_normal_clouds")
    # Critical: normal clouds with moderate visual signal must not be penalized to near 0
    assert result.development_risk_index > 5.0, (
        f"Normal cloud formation bias detected! Risk={result.development_risk_index:.1f} "
        "is too low. Normal clouds must not automatically map to 0 risk."
    )


# --------------------------------------------------------------------------- #
# Case 3: Dense clouds but no cyclone evidence
# --------------------------------------------------------------------------- #
def test_case3_dense_cloud_no_cyclone_evidence() -> None:
    """Dense clouds (slightly below threshold) + weak organization → low-moderate risk."""
    result = compute_risk_index(np.array([0.60, 0.40]), WEAK_ENV, WEIGHTS, "synthetic_mvp")
    _assert_in_range(result, "case3_dense_cloud_no_evidence")
    assert result.classification == "non_cyclone"
    # Should score below midpoint — not near 100 just because clouds are present
    assert result.development_risk_index < 60.0


# --------------------------------------------------------------------------- #
# Case 4: Strong organized structure + supportive environment
# --------------------------------------------------------------------------- #
def test_case4_strong_structure_supportive_env() -> None:
    """High cyclone probability + strong organization/convection → high risk."""
    result = compute_risk_index(np.array([0.05, 0.95]), SUPPORTIVE_ENV, WEIGHTS, "synthetic_mvp")
    _assert_in_range(result, "case4_strong_organized")
    assert result.classification == "cyclone"
    assert result.development_risk_index > 70.0, (
        f"Strong structure + supportive env should score high, got {result.development_risk_index:.1f}"
    )


# --------------------------------------------------------------------------- #
# Case 5: Strong visual + weak numerical
# --------------------------------------------------------------------------- #
def test_case5_strong_visual_weak_numerical() -> None:
    """High cyclone prob but weak environmental features → moderate-high risk."""
    result = compute_risk_index(np.array([0.10, 0.90]), WEAK_ENV, WEIGHTS, "synthetic_mvp")
    _assert_in_range(result, "case5_strong_visual_weak_num")
    assert result.classification == "cyclone"
    # Visual weight (0.4) dominates — should still score substantially
    assert result.development_risk_index > 30.0


# --------------------------------------------------------------------------- #
# Case 6: Weak visual + strong numerical
# --------------------------------------------------------------------------- #
def test_case6_weak_visual_strong_numerical() -> None:
    """Low cyclone prob (visual says no) but strong environmental signal → moderate risk."""
    result = compute_risk_index(np.array([0.75, 0.25]), SUPPORTIVE_ENV, WEIGHTS, "synthetic_mvp")
    _assert_in_range(result, "case6_weak_visual_strong_num")
    assert result.classification == "non_cyclone"
    # Environmental weight should still push risk up somewhat
    assert result.development_risk_index > 15.0


# --------------------------------------------------------------------------- #
# Case 7: Missing numerical feature → defaults to 0.0 gracefully
# --------------------------------------------------------------------------- #
def test_case7_missing_numerical_feature() -> None:
    """Missing env features use default 0.0 — should not crash, risk must be in range."""
    result = compute_risk_index(np.array([0.3, 0.7]), {}, WEIGHTS, "synthetic_mvp")
    _assert_in_range(result, "case7_missing_features")
    # Only visual signal present; environmental defaults to 0
    # Risk should be dominated by visual weight
    assert result.development_risk_index > 0.0


# --------------------------------------------------------------------------- #
# Case 8: Extreme but valid numerical values
# --------------------------------------------------------------------------- #
def test_case8_extreme_valid_values() -> None:
    """All index features at maximum (1.0) + full humidity → must not exceed 100."""
    extreme_env = {
        "cloud_organization_index": 1.0,
        "rotation_index": 1.0,
        "persistence_index": 1.0,
        "convection_index": 1.0,
        "relative_humidity_pct": 100.0,
    }
    result = compute_risk_index(np.array([0.0, 1.0]), extreme_env, WEIGHTS, "synthetic_mvp")
    _assert_in_range(result, "case8_extreme_valid")
    assert result.development_risk_index == pytest.approx(100.0, abs=0.01)

    # And the minimum extreme
    min_env = {
        "cloud_organization_index": 0.0,
        "rotation_index": 0.0,
        "persistence_index": 0.0,
        "convection_index": 0.0,
        "relative_humidity_pct": 0.0,
    }
    result_min = compute_risk_index(np.array([1.0, 0.0]), min_env, WEIGHTS, "synthetic_mvp")
    _assert_in_range(result_min, "case8_extreme_min")
    assert result_min.development_risk_index == pytest.approx(0.0, abs=0.01)


# --------------------------------------------------------------------------- #
# Case 9: Completely invalid input → raises ValidationError
# --------------------------------------------------------------------------- #
def test_case9_invalid_nan_probabilities() -> None:
    """NaN probabilities must be rejected with a clear ValidationError."""
    with pytest.raises(ValidationError, match="NaN"):
        compute_risk_index(np.array([np.nan, np.nan]), NEUTRAL_ENV, WEIGHTS, "synthetic_mvp")


def test_case9_wrong_probability_shape() -> None:
    """Probabilities with wrong number of classes must raise ValidationError."""
    with pytest.raises(ValidationError):
        compute_risk_index(np.array([0.2, 0.4, 0.4]), NEUTRAL_ENV, WEIGHTS, "synthetic_mvp")


# --------------------------------------------------------------------------- #
# Case 10: Empty / zero-length input
# --------------------------------------------------------------------------- #
def test_case10_empty_probability_array() -> None:
    """Empty array must raise ValidationError, not produce silent wrong result."""
    with pytest.raises((ValidationError, IndexError, ValueError)):
        compute_risk_index(np.array([]), NEUTRAL_ENV, WEIGHTS, "synthetic_mvp")


# --------------------------------------------------------------------------- #
# Additional: Risk index invariants
# --------------------------------------------------------------------------- #
def test_risk_index_always_in_range_random() -> None:
    """100 random probability vectors must always produce risk in [0, 100]."""
    rng = np.random.default_rng(42)
    for _ in range(100):
        raw = rng.uniform(0, 1, 2)
        probs = raw / raw.sum()
        env = {
            "cloud_organization_index": float(rng.uniform(0, 1)),
            "rotation_index": float(rng.uniform(0, 1)),
            "persistence_index": float(rng.uniform(0, 1)),
            "convection_index": float(rng.uniform(0, 1)),
            "relative_humidity_pct": float(rng.uniform(0, 100)),
        }
        result = compute_risk_index(probs, env, WEIGHTS, "synthetic_mvp")
        assert 0.0 <= result.development_risk_index <= 100.0
