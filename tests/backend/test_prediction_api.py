"""Comprehensive tests for the /predict endpoint and backend inference service."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.dependencies import get_prediction_service, set_prediction_service
from backend.app.main import app
from backend.app.services.prediction_service import PredictionService
from src.inference.llm_explainer import MockLLMExplainer
from src.inference.llm_verifier import MockLLMVerifier
from src.inference.predictor import CycloSensePredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CYCLONE_IMAGE = PROJECT_ROOT / "sample_data" / "images" / "insat_cyclone_101.jpg"
CLEAR_IMAGE = PROJECT_ROOT / "sample_data" / "images" / "clear_sky_ocean.jpg"

TC01_NUMERICAL: dict[str, str] = {
    "temperature_c": "28.5",
    "sea_surface_temperature_c": "30.5",
    "relative_humidity_pct": "88.0",
    "water_vapour_gkg": "22.0",
    "surface_pressure_hpa": "975.0",
    "wind_speed_ms": "42.0",
    "wind_direction_deg": "120.0",
    "vertical_wind_shear_ms": "4.5",
    "cloud_organization_index": "0.92",
    "convection_index": "0.88",
    "rotation_index": "0.85",
    "persistence_index": "0.90",
}

TC02_NUMERICAL: dict[str, str] = {
    "temperature_c": "25.0",
    "sea_surface_temperature_c": "26.0",
    "relative_humidity_pct": "35.0",
    "water_vapour_gkg": "10.0",
    "surface_pressure_hpa": "1016.0",
    "wind_speed_ms": "6.0",
    "wind_direction_deg": "45.0",
    "vertical_wind_shear_ms": "12.0",
    "cloud_organization_index": "0.05",
    "convection_index": "0.08",
    "rotation_index": "0.02",
    "persistence_index": "0.05",
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def predictor() -> CycloSensePredictor:
    """Shared predictor instance for module-scoped tests."""
    return CycloSensePredictor.from_project()


# ── TEST 1: Valid Prediction Request — TC-01 High-Risk Case ─────────────────
def test_predict_tc01_high_risk(client: TestClient) -> None:
    """Test valid prediction on TC-01: high risk, gate triggered, verified cyclone."""
    assert CYCLONE_IMAGE.is_file(), f"Sample image missing at {CYCLONE_IMAGE}"

    with CYCLONE_IMAGE.open("rb") as f:
        files = {"image": ("insat_cyclone_101.jpg", f, "image/jpeg")}
        data = {**TC01_NUMERICAL, "case_id": "TC-01"}
        response = client.post("/predict", files=files, data=data)

    assert response.status_code == 200
    res = response.json()

    assert res["case_id"] == "TC-01"
    # ML prediction checks
    pred = res["prediction"]
    assert pred["classification"] == "cyclone"
    assert pred["visual_signal"] >= 0.50
    assert pred["environmental_signal"] >= 0.50
    assert pred["fusion_signal"] >= 0.50
    assert pred["development_risk_index"] > 60.0
    assert pred["data_source_tag"] == "synthetic_mvp"
    assert "Development/Risk Index" in pred["note"]

    # Verification gate checks
    verif = res["verification"]
    assert verif["gate_triggered"] is True
    assert verif["llm_called"] is True
    assert verif["status"] in ("VERIFIED_CYCLONE", "HUMAN_REVIEW_REQUIRED", "SECONDARY_REVIEW_CONFLICT")
    assert verif["diagnostic"] != ""

    # Explanation checks
    expl = res["explanation"]
    assert expl["type"] == "high_risk_verification"
    assert len(expl["summary"]) > 0
    assert len(expl["atmospheric_rationale"]) > 0
    assert len(expl["primary_drivers"]) > 0
    assert len(expl["operational_guidance"]) > 0

    # Timing check
    assert res["timing_ms"]["total_ms"] > 0


# ── TEST 2: Valid Prediction Request — TC-02 Low-Risk Case ──────────────────
def test_predict_tc02_low_risk(client: TestClient) -> None:
    """Test valid prediction on TC-02: low risk with suppressed gate and low risk explanation.

    NOTE: Per ADR-007, the Image CNN was trained on cyclone-only images and acts as an
    embedding feature extractor — it may still assign high visual probability to any input.
    However, the environmental MLP and fusion signals for TC-02 (calm ocean, high shear,
    low SST, minimal rotation) correctly suppress the gate. We therefore only assert gate
    suppression and explanation type, NOT the raw visual probability.
    """
    assert CLEAR_IMAGE.is_file(), f"Sample image missing at {CLEAR_IMAGE}"

    with CLEAR_IMAGE.open("rb") as f:
        files = {"image": ("clear_sky_ocean.jpg", f, "image/jpeg")}
        data = {**TC02_NUMERICAL, "case_id": "TC-02"}
        response = client.post("/predict", files=files, data=data)

    assert response.status_code == 200
    res = response.json()

    assert res["case_id"] == "TC-02"
    pred = res["prediction"]
    # The risk index must be low for benign conditions regardless of CNN bias
    assert pred["development_risk_index"] < 60.0

    # Gate must be suppressed (env and fusion signals < 50% for TC-02)
    verif = res["verification"]
    assert verif["gate_triggered"] is False
    assert verif["llm_called"] is False
    assert verif["status"] == "LLM_NOT_REQUIRED"

    # Explanation must be suppressed/low risk
    expl = res["explanation"]
    assert expl["type"] == "suppressed_or_low_risk"
    assert "suppressed" in expl["summary"].lower() or "absent" in expl["summary"].lower()
    assert len(expl["inhibiting_barriers"]) > 0


# ── TEST 3: Missing Image File ──────────────────────────────────────────────
def test_predict_missing_image(client: TestClient) -> None:
    """Test request without an image file returns 422 Unprocessable Entity."""
    data = {**TC01_NUMERICAL, "case_id": "TC-01"}
    response = client.post("/predict", data=data)
    assert response.status_code == 422


# ── TEST 4: Invalid Image Extension ─────────────────────────────────────────
def test_predict_invalid_image_extension(client: TestClient) -> None:
    """Test request with unsupported file extension (.txt) returns 400 Bad Request."""
    dummy_bytes = io.BytesIO(b"fake text content")
    files = {"image": ("test_file.txt", dummy_bytes, "text/plain")}
    data = {**TC01_NUMERICAL, "case_id": "TC-01"}
    response = client.post("/predict", files=files, data=data)
    assert response.status_code == 400
    assert "Unsupported image extension" in response.json()["detail"]


# ── TEST 5: Corrupted Image File ────────────────────────────────────────────
def test_predict_corrupted_image(client: TestClient) -> None:
    """Test corrupted image content returns 400 Bad Request."""
    corrupted_bytes = io.BytesIO(b"not an image file at all")
    files = {"image": ("corrupted.jpg", corrupted_bytes, "image/jpeg")}
    data = {**TC01_NUMERICAL, "case_id": "TC-01"}
    response = client.post("/predict", files=files, data=data)
    assert response.status_code == 400
    assert "valid or readable image" in response.json()["detail"]


# ── TEST 6: Empty Image File ────────────────────────────────────────────────
def test_predict_empty_image(client: TestClient) -> None:
    """Test empty image (0 bytes) returns 400 Bad Request."""
    empty_bytes = io.BytesIO(b"")
    files = {"image": ("empty.jpg", empty_bytes, "image/jpeg")}
    data = {**TC01_NUMERICAL, "case_id": "TC-01"}
    response = client.post("/predict", files=files, data=data)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


# ── TEST 7: Missing Required Numerical Field ────────────────────────────────
def test_predict_missing_numerical_field(client: TestClient) -> None:
    """Test request missing a required numerical feature returns 422 Unprocessable Entity."""
    data = dict(TC01_NUMERICAL)
    del data["sea_surface_temperature_c"]  # Missing feature

    with CYCLONE_IMAGE.open("rb") as f:
        files = {"image": ("insat_cyclone_101.jpg", f, "image/jpeg")}
        response = client.post("/predict", files=files, data=data)

    assert response.status_code == 422


# ── TEST 8: Invalid Numerical Value Type ────────────────────────────────────
def test_predict_invalid_numerical_value_type(client: TestClient) -> None:
    """Test non-numeric string for a float field returns 422."""
    data = dict(TC01_NUMERICAL)
    data["sea_surface_temperature_c"] = "invalid_string"

    with CYCLONE_IMAGE.open("rb") as f:
        files = {"image": ("insat_cyclone_101.jpg", f, "image/jpeg")}
        response = client.post("/predict", files=files, data=data)

    assert response.status_code == 422


# ── TEST 9: Out-of-Range Numerical Value ───────────────────────────────────
def test_predict_out_of_range_numerical_value(client: TestClient) -> None:
    """Test out-of-range numerical feature (SST = 999.0 °C) returns 422.

    NOTE: FastAPI raises 422 via its default Pydantic validation handler on the
    Form dependency. The HTTPException from as_form() also produces 422.
    """
    data = dict(TC01_NUMERICAL)
    data["sea_surface_temperature_c"] = "999.0"  # Exceeds max 45.0

    with CYCLONE_IMAGE.open("rb") as f:
        files = {"image": ("insat_cyclone_101.jpg", f, "image/jpeg")}
        response = client.post("/predict", files=files, data=data)

    assert response.status_code in (422, 400)


# ── TEST 10: Verification Gate Conflict (SECONDARY_REVIEW_CONFLICT) ─────────
def test_predict_secondary_review_conflict(client: TestClient, predictor: CycloSensePredictor) -> None:
    """Test scenario where LLM verifier identifies non-cyclone despite high ML signals."""
    mock_verifier = MockLLMVerifier(
        response="non_cyclone_consistent",
        confidence=0.88,
        reason="Disorganized benign convective cluster",
    )

    class _ConflictService(PredictionService):
        def predict(self, image_path, numerical_features, case_id="custom", **_kwargs):  # type: ignore[override]
            return super().predict(
                image_path=image_path,
                numerical_features=numerical_features,
                case_id=case_id,
                verifier=mock_verifier,
            )

    set_prediction_service(_ConflictService(predictor=predictor))

    try:
        with CYCLONE_IMAGE.open("rb") as f:
            files = {"image": ("insat_cyclone_101.jpg", f, "image/jpeg")}
            data = {**TC01_NUMERICAL, "case_id": "TC-01"}
            response = client.post("/predict", files=files, data=data)

        assert response.status_code == 200
        res = response.json()
        assert res["verification"]["gate_triggered"] is True
        assert res["verification"]["llm_called"] is True
        assert res["verification"]["status"] == "SECONDARY_REVIEW_CONFLICT"
    finally:
        set_prediction_service(None)  # Reset to default singleton


# ── TEST 11: Verification Gate Uncertain (HUMAN_REVIEW_REQUIRED) ────────────
def test_predict_human_review_required(client: TestClient, predictor: CycloSensePredictor) -> None:
    """Test scenario where LLM verifier is uncertain and flags human review."""
    mock_verifier = MockLLMVerifier(
        response="uncertain",
        confidence=0.50,
        reason="Structural ambiguity between spiral band and shear line",
        needs_human_review=True,
    )

    class _UncertainService(PredictionService):
        def predict(self, image_path, numerical_features, case_id="custom", **_kwargs):  # type: ignore[override]
            return super().predict(
                image_path=image_path,
                numerical_features=numerical_features,
                case_id=case_id,
                verifier=mock_verifier,
            )

    set_prediction_service(_UncertainService(predictor=predictor))

    try:
        with CYCLONE_IMAGE.open("rb") as f:
            files = {"image": ("insat_cyclone_101.jpg", f, "image/jpeg")}
            data = {**TC01_NUMERICAL, "case_id": "TC-01"}
            response = client.post("/predict", files=files, data=data)

        assert response.status_code == 200
        res = response.json()
        assert res["verification"]["gate_triggered"] is True
        assert res["verification"]["llm_called"] is True
        assert res["verification"]["status"] == "HUMAN_REVIEW_REQUIRED"
    finally:
        set_prediction_service(None)


# ── TEST 12: LLM Verification Failure (Graceful Fallback) ───────────────────
def test_predict_llm_failure_fallback(client: TestClient, predictor: CycloSensePredictor) -> None:
    """Test that LLM verifier timeout gracefully records LLM_VERIFICATION_FAILED without crashing."""
    mock_verifier = MockLLMVerifier(error="timeout")

    class _FailService(PredictionService):
        def predict(self, image_path, numerical_features, case_id="custom", **_kwargs):  # type: ignore[override]
            return super().predict(
                image_path=image_path,
                numerical_features=numerical_features,
                case_id=case_id,
                verifier=mock_verifier,
            )

    set_prediction_service(_FailService(predictor=predictor))

    try:
        with CYCLONE_IMAGE.open("rb") as f:
            files = {"image": ("insat_cyclone_101.jpg", f, "image/jpeg")}
            data = {**TC01_NUMERICAL, "case_id": "TC-01"}
            response = client.post("/predict", files=files, data=data)

        assert response.status_code == 200
        res = response.json()
        assert res["verification"]["gate_triggered"] is True
        assert res["verification"]["llm_called"] is True
        assert res["verification"]["status"] == "LLM_VERIFICATION_FAILED"
        # ML predictions must still be preserved (no crash)
        assert res["prediction"]["classification"] == "cyclone"
    finally:
        set_prediction_service(None)


# ── TEST 13: Response Schema Keys Conformance ───────────────────────────────
def test_response_schema_exact_keys(client: TestClient) -> None:
    """Validate that API response contains all expected fields in exact schema layout."""
    with CYCLONE_IMAGE.open("rb") as f:
        files = {"image": ("insat_cyclone_101.jpg", f, "image/jpeg")}
        data = {**TC01_NUMERICAL, "case_id": "TC-01"}
        response = client.post("/predict", files=files, data=data)

    assert response.status_code == 200
    res = response.json()

    # Top-level keys
    assert set(res.keys()) == {"case_id", "prediction", "verification", "explanation", "timing_ms"}

    # Prediction keys
    assert set(res["prediction"].keys()) == {
        "classification",
        "visual_signal",
        "environmental_signal",
        "fusion_signal",
        "development_risk_index",
        "confidence_pct",
        "components",
        "data_source_tag",
        "note",
    }

    # Verification keys
    assert set(res["verification"].keys()) == {
        "gate_triggered",
        "llm_called",
        "status",
        "diagnostic",
        "assessment",
        "confidence",
        "reason",
        "needs_human_review",
    }

    # Explanation keys
    assert set(res["explanation"].keys()) == {
        "type",
        "summary",
        "atmospheric_rationale",
        "primary_drivers",
        "inhibiting_barriers",
        "operational_guidance",
    }


# ── TEST 14: CORS Preflight ─────────────────────────────────────────────────
def test_cors_preflight(client: TestClient) -> None:
    """Test CORS preflight OPTIONS request returns approved headers."""
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


# ── TEST 15: Isolated Image Prediction ──────────────────────────────────────
def test_predict_image_isolated(client: TestClient) -> None:
    """Test /predict/image isolated endpoint."""
    with CYCLONE_IMAGE.open("rb") as f:
        files = {"image": ("insat_cyclone_101.jpg", f, "image/jpeg")}
        data = {"case_id": "TEST-IMAGE"}
        response = client.post("/predict/image", files=files, data=data)

    assert response.status_code == 200
    res = response.json()
    assert res["case_id"] == "TEST-IMAGE"
    assert "visual_signal" in res
    assert res["timing_ms"] > 0


# ── TEST 16: Isolated Numerical Prediction ──────────────────────────────────
def test_predict_numerical_isolated(client: TestClient) -> None:
    """Test /predict/numerical isolated endpoint."""
    data = {**TC01_NUMERICAL, "case_id": "TEST-NUM"}
    response = client.post("/predict/numerical", data=data)

    assert response.status_code == 200
    res = response.json()
    assert res["case_id"] == "TEST-NUM"
    assert "environmental_signal" in res
    assert res["timing_ms"] > 0


# ── TEST 17: Isolated Fusion Prediction ─────────────────────────────────────
def test_predict_fusion_isolated(client: TestClient) -> None:
    """Test /predict/fusion isolated endpoint."""
    with CYCLONE_IMAGE.open("rb") as f:
        files = {"image": ("insat_cyclone_101.jpg", f, "image/jpeg")}
        data = {**TC01_NUMERICAL, "case_id": "TEST-FUSION"}
        response = client.post("/predict/fusion", files=files, data=data)

    assert response.status_code == 200
    res = response.json()
    assert res["case_id"] == "TEST-FUSION"
    assert "fusion_signal" in res
    assert res["timing_ms"] > 0


# ── TEST 18: Isolated Risk Index Calculation ────────────────────────────────
def test_predict_risk_index_isolated(client: TestClient) -> None:
    """Test /scoring/risk-index isolated endpoint."""
    with CYCLONE_IMAGE.open("rb") as f:
        files = {"image": ("insat_cyclone_101.jpg", f, "image/jpeg")}
        data = {**TC01_NUMERICAL, "case_id": "TEST-RISK"}
        response = client.post("/scoring/risk-index", files=files, data=data)

    assert response.status_code == 200
    res = response.json()
    assert res["case_id"] == "TEST-RISK"
    assert "development_risk_index" in res
    assert "components" in res
    assert res["timing_ms"] > 0

