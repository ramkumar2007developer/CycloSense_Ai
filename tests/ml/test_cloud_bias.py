"""Cloud-bias anti-test — normal clouds must not auto-map to strong non-cyclone."""

import numpy as np

from src.scoring.risk_index import ScoringWeights, compute_risk_index


def test_neutral_clouds_do_not_force_zero_risk() -> None:
    """Normal cloudy neutral env should not automatically get minimum risk from scoring alone."""
    weights = ScoringWeights(0.4, 0.3, 0.1, 0.1, 0.05, 0.05)
    # Moderate visual signal but neutral environmental indices (not anti-cyclone penalty)
    result = compute_risk_index(
        np.array([0.45, 0.55]),
        {
            "cloud_organization_index": 0.35,
            "rotation_index": 0.2,
            "persistence_index": 0.25,
            "convection_index": 0.3,
            "relative_humidity_pct": 65,
        },
        weights,
        data_source_tag="synthetic_mvp",
    )
    assert result.development_risk_index > 5.0
    assert result.classification in {"cyclone", "non_cyclone"}
