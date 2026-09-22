"""Integration tests for full ML pipeline."""

import copy
from pathlib import Path

import pytest

from src.config.ml_config import MLConfig, load_ml_config
from src.data.audit import audit_datasets
from src.data.manifest import PAIRING_TAG, build_synthetic_mvp_manifest
from src.training.run_ml_pipeline import run_ml_pipeline


@pytest.fixture(scope="module")
def external_available() -> bool:
    root = Path.home() / "Downloads" / "dataset-main" / "dataset-main"
    env = Path.home() / "Downloads" / "cyclosense_mvp_environmental_dataset.csv"
    return root.is_dir() and env.is_file()


def test_audit_runs(external_available) -> None:
    if not external_available:
        pytest.skip("Local datasets not available")
    cfg = load_ml_config()
    report = audit_datasets(cfg)
    assert report.image_count == 136
    assert report.numerical_row_count == 1000
    assert report.source_tag == "synthetic_mvp"


def test_manifest_pairing_tag(external_available) -> None:
    if not external_available:
        pytest.skip("Local datasets not available")
    manifest = build_synthetic_mvp_manifest(load_ml_config())
    assert (manifest["pairing_tag"] == PAIRING_TAG).all()
    assert (manifest["data_source_tag"] == "synthetic_mvp").all()


@pytest.mark.slow
def test_full_pipeline_runs(external_available, tmp_path_factory) -> None:
    if not external_available:
        pytest.skip("Local datasets not available")
    base_cfg = load_ml_config()
    cfg_raw = copy.deepcopy(base_cfg.raw)
    cfg_raw["training"]["epochs"] = 1
    test_cfg = MLConfig(
        project_root=base_cfg.project_root,
        external_data_root=base_cfg.external_data_root,
        raw=cfg_raw,
    )

    out = tmp_path_factory.mktemp("artifacts")
    artifacts = run_ml_pipeline(test_cfg, output_dir=out)

    assert artifacts.manifest_path.is_file()
    assert artifacts.image_model_path.is_file()
    assert artifacts.numerical_model_path.is_file()
    assert artifacts.fusion_model_path.is_file()
    assert artifacts.numerical_scaler_path.is_file()
    assert artifacts.metrics_path.is_file()
    assert artifacts.history_path.is_file()
    assert 0.0 <= artifacts.fusion_test_metrics.accuracy <= 1.0
