"""Unit and integration tests for the LLM Meteorological Explanation Generator."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from src.config.ml_config import LLMExplanationConfig, load_ml_config
from src.inference.llm_explainer import (
    LLMExplanationError,
    LLMExplanationResult,
    MockLLMExplainer,
    build_explainer,
)
from src.inference.predictor import CycloSensePredictor


SAMPLE_CYCLONE_IMAGE = Path("sample_data/images/insat_cyclone_101.jpg")
SAMPLE_BENIGN_IMAGE = Path("sample_data/images/benign_cloud_cluster.jpg")


# ── Case A Tests (Signals >= 50% / High Risk Verification) ───────────────────

def test_case_a_verified_cyclone_explanation() -> None:
    explainer = MockLLMExplainer()
    payload = {
        "all_signals_above_50": True,
        "image_probability": 0.85,
        "numerical_probability": 0.90,
        "fusion_probability": 0.88,
        "development_risk_index": 82.5,
        "classification": "cyclone",
        "final_status": "VERIFIED_CYCLONE",
    }
    env = {
        "sea_surface_temperature_c": 30.5,
        "vertical_wind_shear_ms": 6.0,
        "relative_humidity_pct": 85.0,
        "rotation_index": 0.82,
        "cloud_organization_index": 0.88,
    }

    res = explainer.generate_explanation(payload, env)

    assert res.case_type == "high_risk_verification"
    assert "confirm active cyclogenesis" in res.summary
    assert "82.5" in res.summary
    assert any("sea surface temperature" in d.lower() for d in res.primary_drivers)
    assert any("wind shear" in d.lower() for d in res.primary_drivers)
    assert "warning advisory" in res.operational_guidance.lower()
    assert explainer.call_count == 1


def test_case_a_secondary_conflict_explanation() -> None:
    explainer = MockLLMExplainer()
    payload = {
        "all_signals_above_50": True,
        "image_probability": 0.75,
        "numerical_probability": 0.80,
        "fusion_probability": 0.72,
        "development_risk_index": 68.0,
        "classification": "cyclone",
        "final_status": "SECONDARY_REVIEW_CONFLICT",
    }
    env = {
        "sea_surface_temperature_c": 28.0,
        "vertical_wind_shear_ms": 15.0,
        "relative_humidity_pct": 75.0,
    }

    res = explainer.generate_explanation(payload, env)

    assert res.case_type == "high_risk_verification"
    assert "secondary review conflict" in res.summary.lower()
    assert "benign" in res.meteorological_rationale.lower()
    assert "do not issue cyclone alert" in res.operational_guidance.lower()


def test_case_a_human_review_required_explanation() -> None:
    explainer = MockLLMExplainer()
    payload = {
        "all_signals_above_50": True,
        "image_probability": 0.60,
        "numerical_probability": 0.65,
        "fusion_probability": 0.58,
        "development_risk_index": 62.0,
        "classification": "cyclone",
        "final_status": "HUMAN_REVIEW_REQUIRED",
    }
    env = {"sea_surface_temperature_c": 28.5}

    res = explainer.generate_explanation(payload, env)

    assert res.case_type == "high_risk_verification"
    assert "manual meteorological review" in res.summary.lower()
    assert "analyst triage" in res.meteorological_rationale.lower()
    assert "duty meteorologist" in res.operational_guidance.lower()


# ── Case B Tests (Signals < 50% / Suppressed or Low Risk) ────────────────────

def test_case_b_suppressed_by_high_shear() -> None:
    explainer = MockLLMExplainer()
    payload = {
        "all_signals_above_50": False,
        "image_probability": 0.62,
        "numerical_probability": 0.70,
        "fusion_probability": 0.42,  # < 0.50
        "development_risk_index": 38.0,
        "classification": "non_cyclone",
        "final_status": "LLM_NOT_REQUIRED",
    }
    env = {
        "sea_surface_temperature_c": 29.0,
        "vertical_wind_shear_ms": 32.0,  # Hostile shear
        "relative_humidity_pct": 78.0,
        "rotation_index": 0.20,
    }

    res = explainer.generate_explanation(payload, env)

    assert res.case_type == "suppressed_or_low_risk"
    assert "suppressed or absent" in res.summary.lower()
    assert "not required" in res.summary.lower()
    assert any("shear" in f.lower() for f in res.inhibiting_factors)
    assert any("vorticity" in f.lower() or "rotation" in f.lower() for f in res.inhibiting_factors)
    assert "no cyclone alert required" in res.operational_guidance.lower()


def test_case_b_fair_weather_clear_ocean() -> None:
    explainer = MockLLMExplainer()
    payload = {
        "all_signals_above_50": False,
        "image_probability": 0.15,
        "numerical_probability": 0.10,
        "fusion_probability": 0.08,
        "development_risk_index": 12.0,
        "classification": "non_cyclone",
        "final_status": "LLM_NOT_REQUIRED",
    }
    env = {
        "sea_surface_temperature_c": 24.0,  # Cool water
        "vertical_wind_shear_ms": 14.0,
        "relative_humidity_pct": 35.0,  # Dry air
        "cloud_organization_index": 0.05,
    }

    res = explainer.generate_explanation(payload, env)

    assert res.case_type == "suppressed_or_low_risk"
    assert any("dry air" in f.lower() for f in res.inhibiting_factors)
    assert any("sst" in f.lower() or "ocean heat" in f.lower() for f in res.inhibiting_factors)


# ── Error & Fallback Tests ───────────────────────────────────────────────────

def test_explainer_error_handling() -> None:
    explainer = MockLLMExplainer(error="timeout")
    payload = {"all_signals_above_50": False}

    with pytest.raises(LLMExplanationError) as exc_info:
        explainer.generate_explanation(payload)
    assert "simulated timeout" in str(exc_info.value)


# ── End-to-End Predictor Integration Tests ───────────────────────────────────

def test_predictor_predict_with_explanation_case_a() -> None:
    predictor = CycloSensePredictor.from_project()
    sample_csv = pd.read_csv("sample_data/sample_environmental.csv")
    env_row = sample_csv.iloc[0]

    result = predictor.predict_with_explanation(
        image_path=SAMPLE_CYCLONE_IMAGE,
        env_row=env_row,
    )

    assert "llm_explanation" in result
    explanation = result["llm_explanation"]
    assert "summary" in explanation
    assert "meteorological_rationale" in explanation
    assert "primary_drivers" in explanation
    assert "inhibiting_factors" in explanation
    assert "operational_guidance" in explanation
    assert explanation["case_type"] in {"high_risk_verification", "suppressed_or_low_risk"}


def test_predictor_predict_with_explanation_case_b() -> None:
    predictor = CycloSensePredictor.from_project()
    sample_csv = pd.read_csv("sample_data/sample_environmental.csv")
    env_row = sample_csv.iloc[0]

    # benign cloud cluster
    result = predictor.predict_with_explanation(
        image_path=SAMPLE_BENIGN_IMAGE,
        env_row=env_row,
    )

    assert "llm_explanation" in result
    explanation = result["llm_explanation"]
    assert len(explanation["summary"]) > 0
    assert len(explanation["operational_guidance"]) > 0


def test_predictor_predict_explain_flag() -> None:
    predictor = CycloSensePredictor.from_project()
    sample_csv = pd.read_csv("sample_data/sample_environmental.csv")
    env_row = sample_csv.iloc[0]

    # Without explain=True
    res_normal = predictor.predict(SAMPLE_CYCLONE_IMAGE, env_row, explain=False)
    assert "llm_explanation" not in res_normal

    # With explain=True
    res_explained = predictor.predict(SAMPLE_CYCLONE_IMAGE, env_row, explain=True)
    assert "llm_explanation" in res_explained
