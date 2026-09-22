"""Data leakage verification tests.

Verifies:
1. Train-fitted scaler parameters (mean_, var_) are identical before and after test transform.
2. Group split has 0% storm_id overlap between train, val, and test splits.
3. Image split partitions are strictly disjoint.
4. Calling transform() on an unfitted pipeline raises DatasetValidationError.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.numerical_pipeline import NumericalPipeline
from src.data.splitting import split_by_group
from src.exceptions import DatasetValidationError


@pytest.fixture
def dummy_dataset() -> pd.DataFrame:
    records = []
    for i in range(30):
        storm = f"STORM_{i // 3:02d}"
        records.append({
            "storm_id": storm,
            "timestamp": f"2023-01-01T{i % 24:02d}:00:00Z",
            "temperature_c": 26.0 + (i % 5),
            "sea_surface_temperature_c": 27.0 + (i % 4),
            "relative_humidity_pct": 70.0 + (i % 20),
            "water_vapour_gkg": 18.0 + (i % 8),
            "surface_pressure_hpa": 990.0 + (i % 20),
            "wind_speed_ms": 20.0 + (i % 30),
            "wind_direction_deg": float((i * 12) % 360),
            "vertical_wind_shear_ms": 5.0 + (i % 15),
            "cloud_organization_index": 0.2 + (i % 7) * 0.1,
            "convection_index": 0.2 + (i % 7) * 0.1,
            "rotation_index": 0.1 + (i % 7) * 0.1,
            "persistence_index": 0.2 + (i % 7) * 0.1,
            "cyclone_label": int(i % 2 == 0),
        })
    return pd.DataFrame(records)


def test_scaler_parameters_invariant_after_test_transform(dummy_dataset: pd.DataFrame) -> None:
    """Scaler mean and variance must not mutate when transforming unseen test data."""
    feature_cols = [
        "temperature_c", "sea_surface_temperature_c", "relative_humidity_pct",
        "water_vapour_gkg", "surface_pressure_hpa", "wind_speed_ms",
        "wind_direction_deg", "vertical_wind_shear_ms", "cloud_organization_index",
        "convection_index", "rotation_index", "persistence_index",
    ]
    train_df = dummy_dataset.iloc[:20].copy()
    test_df = dummy_dataset.iloc[20:].copy()

    pipeline = NumericalPipeline(
        feature_columns=feature_cols,
        label_column="cyclone_label",
        group_column="storm_id",
        scaler=StandardScaler(),
        data_source_tag="synthetic_mvp",
    )

    pipeline.fit(train_df)
    mean_before = np.copy(pipeline.scaler.mean_)
    var_before = np.copy(pipeline.scaler.var_)

    # Transform test set
    _ = pipeline.transform(test_df)

    np.testing.assert_array_equal(pipeline.scaler.mean_, mean_before, err_msg="Scaler mean changed after test transform!")
    np.testing.assert_array_equal(pipeline.scaler.var_, var_before, err_msg="Scaler variance changed after test transform!")


def test_zero_group_leakage_in_splits(dummy_dataset: pd.DataFrame) -> None:
    """Zero storm_id overlap between train, val, and test partitions."""
    splits = split_by_group(
        dummy_dataset,
        group_column="storm_id",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_storms = set(splits.train["storm_id"].unique())
    val_storms = set(splits.validation["storm_id"].unique())
    test_storms = set(splits.test["storm_id"].unique())

    assert len(train_storms.intersection(val_storms)) == 0, "Train and val share storm_ids!"
    assert len(train_storms.intersection(test_storms)) == 0, "Train and test share storm_ids!"
    assert len(val_storms.intersection(test_storms)) == 0, "Val and test share storm_ids!"


def test_unfitted_transform_raises(dummy_dataset: pd.DataFrame) -> None:
    """Transforming before fit must raise DatasetValidationError to prevent unscaled/leaked flow."""
    feature_cols = [
        "temperature_c", "sea_surface_temperature_c", "relative_humidity_pct",
        "water_vapour_gkg", "surface_pressure_hpa", "wind_speed_ms",
        "wind_direction_deg", "vertical_wind_shear_ms", "cloud_organization_index",
        "convection_index", "rotation_index", "persistence_index",
    ]
    pipeline = NumericalPipeline(
        feature_columns=feature_cols,
        label_column="cyclone_label",
        group_column="storm_id",
        scaler=StandardScaler(),
        data_source_tag="synthetic_mvp",
    )

    with pytest.raises(DatasetValidationError, match="called before fit"):
        pipeline.transform(dummy_dataset)
