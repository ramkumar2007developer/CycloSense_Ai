"""Tests for the /health endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """Test that GET /health returns 200 and expected service status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "CycloSense AI"
    assert data["version"] == "0.1.0"
