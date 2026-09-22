"""Comprehensive test suite for the Secondary LLM Verification Gate.

Covers all 11 required scenarios from specification:
- Test 1: All signals below 50%
- Test 2: One signal below 50%
- Test 3: Exactly 50% (boundary condition: >= 0.50)
- Test 4: All above 50%
- Test 5: Cyclone-like image + all signals >= 50% -> VERIFIED_CYCLONE
- Test 6: Normal scattered cloud image + all signals >= 50% -> SECONDARY_REVIEW_CONFLICT
- Test 7: LLM uncertain -> HUMAN_REVIEW_REQUIRED
- Test 8: LLM failure -> LLM_VERIFICATION_FAILED (no crash, preserves ML outputs)
- Test 9: Missing numerical data -> handled per existing architecture (no fabrication)
- Test 10: Corrupted/invalid image -> validation error, no LLM call
- Test 11: Normal cloud false-positive scenario with sample_data/images/benign_cloud_cluster.jpg
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.ml_config import LLMVerificationConfig, load_ml_config
from src.exceptions import DatasetValidationError, ValidationError
from src.inference.llm_verifier import (
    LLMVerificationError,
    LLMVerificationResult,
    MockLLMVerifier,
)
from src.inference.predictor import CycloSensePredictor
from src.inference.verification_gate import (
    GateSignals,
    VerificationGateResult,
    VerificationStatus,
    evaluate_verification_gate,
)

SAMPLE_IMAGE = Path("sample_data/images/insat_cyclone_101.jpg")
BENIGN_IMAGE = Path("sample_data/images/benign_cloud_cluster.jpg")


@pytest.fixture
def default_config() -> LLMVerificationConfig:
    return LLMVerificationConfig(
        enabled=True,
        provider="mock",
        model_name="llama-3.2-11b-vision-preview",
        gate_threshold=0.50,
        confidence_threshold=0.70,
        timeout_seconds=5.0,
        max_retries=1,
    )


# ── TEST 1: All signals below 50% ───────────────────────────────────────────
def test_gate_all_signals_below_threshold(default_config: LLMVerificationConfig) -> None:
    signals = GateSignals(image_probability=0.40, numerical_probability=0.45, fusion_probability=0.48)
    mock = MockLLMVerifier(response="cyclone_consistent", confidence=0.90)

    result = evaluate_verification_gate(
        signals=signals,
        image_path=SAMPLE_IMAGE,
        verifier=mock,
        config=default_config,
    )

    assert result.llm_called is False
    assert mock.call_count == 0
    assert result.all_signals_above_threshold is False
    assert result.final_status == VerificationStatus.LLM_NOT_REQUIRED
    assert result.llm_result is None


# ── TEST 2: One signal below 50% ────────────────────────────────────────────
def test_gate_one_signal_below_threshold(default_config: LLMVerificationConfig) -> None:
    signals = GateSignals(image_probability=0.70, numerical_probability=0.49, fusion_probability=0.80)
    mock = MockLLMVerifier(response="cyclone_consistent", confidence=0.90)

    result = evaluate_verification_gate(
        signals=signals,
        image_path=SAMPLE_IMAGE,
        verifier=mock,
        config=default_config,
    )

    assert result.llm_called is False
    assert mock.call_count == 0
    assert result.all_signals_above_threshold is False
    assert result.final_status == VerificationStatus.LLM_NOT_REQUIRED
    assert "numerical_probability_below_threshold" in result.reason


# ── TEST 3: Exactly 50% boundary check ──────────────────────────────────────
def test_gate_exactly_at_threshold(default_config: LLMVerificationConfig) -> None:
    signals = GateSignals(image_probability=0.50, numerical_probability=0.50, fusion_probability=0.50)
    mock = MockLLMVerifier(response="cyclone_consistent", confidence=0.85)

    result = evaluate_verification_gate(
        signals=signals,
        image_path=SAMPLE_IMAGE,
        verifier=mock,
        config=default_config,
    )

    # Must be >= 0.50, so gate must activate
    assert result.all_signals_above_threshold is True
    assert result.llm_called is True
    assert mock.call_count == 1
    assert result.final_status == VerificationStatus.VERIFIED_CYCLONE


# ── TEST 4: All above 50% ───────────────────────────────────────────────────
def test_gate_all_above_threshold(default_config: LLMVerificationConfig) -> None:
    signals = GateSignals(image_probability=0.75, numerical_probability=0.80, fusion_probability=0.72)
    mock = MockLLMVerifier(response="cyclone_consistent", confidence=0.88)

    result = evaluate_verification_gate(
        signals=signals,
        image_path=SAMPLE_IMAGE,
        verifier=mock,
        config=default_config,
    )

    assert result.all_signals_above_threshold is True
    assert result.llm_called is True
    assert mock.call_count == 1
    assert result.final_status == VerificationStatus.VERIFIED_CYCLONE


# ── TEST 5: Cyclone-like image + all signals >= 50% ─────────────────────────
def test_gate_verified_cyclone_scenario(default_config: LLMVerificationConfig) -> None:
    signals = GateSignals(image_probability=0.90, numerical_probability=0.85, fusion_probability=0.88)
    mock = MockLLMVerifier(
        response="cyclone_consistent",
        confidence=0.92,
        reason="Distinct central eye with well-defined curved convective rainbands.",
    )

    result = evaluate_verification_gate(
        signals=signals,
        image_path=SAMPLE_IMAGE,
        verifier=mock,
        config=default_config,
    )

    assert result.llm_called is True
    assert mock.call_count == 1
    assert result.final_status == VerificationStatus.VERIFIED_CYCLONE
    assert result.llm_result is not None
    assert result.llm_result.visual_assessment == "cyclone_consistent"
    assert result.llm_result.confidence == 0.92


# ── TEST 6: Normal scattered cloud image + all signals >= 50% ───────────────
def test_gate_secondary_review_conflict(default_config: LLMVerificationConfig) -> None:
    signals = GateSignals(image_probability=0.78, numerical_probability=0.72, fusion_probability=0.74)
    mock = MockLLMVerifier(
        response="non_cyclone_consistent",
        confidence=0.89,
        reason="Visual features indicate disorganized shallow cumulus without cyclonic rotation or central dense overcast.",
    )

    result = evaluate_verification_gate(
        signals=signals,
        image_path=BENIGN_IMAGE,
        verifier=mock,
        config=default_config,
    )

    assert result.llm_called is True
    assert mock.call_count == 1
    # Must NOT force VERIFIED_CYCLONE; must return SECONDARY_REVIEW_CONFLICT
    assert result.final_status == VerificationStatus.SECONDARY_REVIEW_CONFLICT
    assert result.llm_result is not None
    assert result.llm_result.visual_assessment == "non_cyclone_consistent"


# ── TEST 7: LLM uncertain ───────────────────────────────────────────────────
def test_gate_uncertain_or_low_confidence(default_config: LLMVerificationConfig) -> None:
    signals = GateSignals(image_probability=0.65, numerical_probability=0.60, fusion_probability=0.62)

    # Sub-case A: assessment is uncertain
    mock_uncertain = MockLLMVerifier(response="uncertain", confidence=0.50, needs_human_review=True)
    res_a = evaluate_verification_gate(signals=signals, image_path=SAMPLE_IMAGE, verifier=mock_uncertain, config=default_config)
    assert res_a.final_status == VerificationStatus.HUMAN_REVIEW_REQUIRED

    # Sub-case B: cyclone_consistent but confidence < confidence_threshold (0.60 < 0.70)
    mock_low_conf = MockLLMVerifier(response="cyclone_consistent", confidence=0.60)
    res_b = evaluate_verification_gate(signals=signals, image_path=SAMPLE_IMAGE, verifier=mock_low_conf, config=default_config)
    assert res_b.final_status == VerificationStatus.HUMAN_REVIEW_REQUIRED


# ── TEST 8: LLM failure (timeout / API error / malformed JSON) ───────────────
@pytest.mark.parametrize("error_type", ["timeout", "api_error", "malformed_json"])
def test_gate_llm_failure_does_not_crash(error_type: str, default_config: LLMVerificationConfig) -> None:
    signals = GateSignals(image_probability=0.82, numerical_probability=0.79, fusion_probability=0.80)
    mock = MockLLMVerifier(error=error_type)

    # System must NOT raise exception; returns LLM_VERIFICATION_FAILED
    result = evaluate_verification_gate(
        signals=signals,
        image_path=SAMPLE_IMAGE,
        verifier=mock,
        config=default_config,
    )

    assert result.llm_called is True
    assert result.final_status == VerificationStatus.LLM_VERIFICATION_FAILED
    assert "verifier_error" in result.reason
    assert result.signals.image_probability == 0.82
    assert result.signals.numerical_probability == 0.79
    assert result.signals.fusion_probability == 0.80


# ── TEST 9: Missing numerical data ──────────────────────────────────────────
def test_missing_numerical_data_raises_validation_error() -> None:
    """The existing numerical pipeline must raise validation error; no dummy signals fabricated."""
    cfg = load_ml_config()
    predictor = CycloSensePredictor.from_project()

    incomplete_env = pd.Series({
        "temperature_c": 28.0,
        # missing relative_humidity, wind_speed, etc.
    })

    with pytest.raises(DatasetValidationError):
        predictor.predict_with_verification(SAMPLE_IMAGE, incomplete_env)


# ── TEST 10: Corrupted / invalid image ───────────────────────────────────────
def test_corrupted_image_raises_validation_error(tmp_path: Path) -> None:
    """Invalid image file raises DatasetValidationError without calling the LLM."""
    cfg = load_ml_config()
    predictor = CycloSensePredictor.from_project()

    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE")

    sample_csv = pd.read_csv("sample_data/sample_environmental.csv")
    valid_env = sample_csv.iloc[0]

    mock = MockLLMVerifier()
    with pytest.raises(DatasetValidationError):
        predictor.predict_with_verification(corrupt_file, valid_env, verifier=mock)

    # Verifier must NOT have been called
    assert mock.call_count == 0


# ── TEST 11: Normal cloud false-positive scenario with benign image ─────────
def test_normal_cloud_benign_cluster_evaluation() -> None:
    """Evaluate benign cloud cluster with high ML activation followed by LLM rejection."""
    cfg = load_ml_config()
    predictor = CycloSensePredictor.from_project()

    sample_csv = pd.read_csv("sample_data/sample_environmental.csv")
    env_row = sample_csv.iloc[0]

    # Inspect signals
    signals = predictor.extract_signals(BENIGN_IMAGE, env_row)
    assert 0.0 <= signals.image_probability <= 1.0
    assert 0.0 <= signals.numerical_probability <= 1.0
    assert 0.0 <= signals.fusion_probability <= 1.0

    # Test with mock verifier identifying benign clouds
    mock = MockLLMVerifier(
        response="non_cyclone_consistent",
        confidence=0.88,
        reason="Disorganized cloud cluster without vorticity or feeder bands.",
    )

    result = predictor.predict_with_verification(
        image_path=BENIGN_IMAGE,
        env_row=env_row,
        verifier=mock,
        image_id="BENIGN-001",
    )

    assert "final_status" in result
    if result["all_signals_above_50"]:
        assert result["llm_called"] is True
        assert result["final_status"] == VerificationStatus.SECONDARY_REVIEW_CONFLICT.value
    else:
        assert result["llm_called"] is False
        assert result["final_status"] == VerificationStatus.LLM_NOT_REQUIRED.value


# ── Structured Logging & Context Verification ────────────────────────────────
def test_structured_log_entry_no_secrets(default_config: LLMVerificationConfig) -> None:
    signals = GateSignals(image_probability=0.72, numerical_probability=0.81, fusion_probability=0.76)
    mock = MockLLMVerifier(response="cyclone_consistent", confidence=0.84)

    gate_result = evaluate_verification_gate(
        signals=signals,
        image_path=SAMPLE_IMAGE,
        verifier=mock,
        config=default_config,
        environmental_context={"sea_surface_temperature_c": 29.5},
        image_id="TEST-IMG-42",
    )

    log_entry = gate_result.to_log_entry(image_id="TEST-IMG-42")
    assert log_entry["image_id"] == "TEST-IMG-42"
    assert log_entry["image_probability"] == 0.72
    assert log_entry["numerical_probability"] == 0.81
    assert log_entry["fusion_probability"] == 0.76
    assert log_entry["all_signals_above_50"] is True
    assert log_entry["llm_called"] is True
    assert log_entry["llm_assessment"] == "cyclone_consistent"
    assert log_entry["llm_confidence"] == 0.84
    assert log_entry["final_status"] == "VERIFIED_CYCLONE"

    # Verify context passed to mock does not contain eval metrics (F1, AUC, precision, recall)
    assert len(mock.calls) == 1
    ctx = mock.calls[0]["ml_context"]
    assert "f1" not in ctx
    assert "auc" not in ctx
    assert "precision" not in ctx
    assert "recall" not in ctx
    assert "environmental_context" in ctx
