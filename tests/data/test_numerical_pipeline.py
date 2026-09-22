"""Numerical pipeline and leakage tests."""

import numpy as np
import pandas as pd
import pytest

from src.data.numerical_pipeline import NumericalPipeline
from src.exceptions import DatasetValidationError


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "storm_id": ["A", "A", "B", "B", "C", "C"],
            "temperature_c": [27.0, 28.0, 26.0, 27.5, 25.0, 26.0],
            "sea_surface_temperature_c": [28.0, 28.5, 27.0, 27.2, 26.5, 26.8],
            "relative_humidity_pct": [70, 72, 80, 78, 65, 67],
            "water_vapour_gkg": [20, 21, 19, 20, 18, 19],
            "surface_pressure_hpa": [990, 992, 995, 994, 998, 997],
            "wind_speed_ms": [40, 42, 35, 36, 30, 31],
            "wind_direction_deg": [90, 100, 180, 170, 270, 260],
            "vertical_wind_shear_ms": [8, 9, 7, 7.5, 6, 6.5],
            "cloud_organization_index": [0.5, 0.6, 0.4, 0.45, 0.3, 0.35],
            "convection_index": [0.4, 0.5, 0.3, 0.35, 0.2, 0.25],
            "rotation_index": [0.2, 0.25, 0.15, 0.18, 0.1, 0.12],
            "persistence_index": [0.3, 0.32, 0.25, 0.27, 0.2, 0.22],
            "cyclone_label": [1, 1, 0, 0, 1, 0],
        }
    )


@pytest.fixture
def pipeline() -> NumericalPipeline:
    cols = [
        "temperature_c", "sea_surface_temperature_c", "relative_humidity_pct",
        "water_vapour_gkg", "surface_pressure_hpa", "wind_speed_ms",
        "wind_direction_deg", "vertical_wind_shear_ms", "cloud_organization_index",
        "convection_index", "rotation_index", "persistence_index",
    ]
    return NumericalPipeline(cols, "cyclone_label", "storm_id", __import__("sklearn.preprocessing").preprocessing.StandardScaler(), "synthetic_mvp")


def test_train_only_scaler_fit(sample_df, pipeline) -> None:
    train = sample_df.iloc[:4]
    test = sample_df.iloc[4:]
    pipeline.fit(train)
    x_test = pipeline.transform(test)
    assert x_test.shape == (2, 12)


def test_nan_rejected(sample_df, pipeline) -> None:
    bad = sample_df.copy()
    bad.loc[0, "temperature_c"] = np.nan
    with pytest.raises(DatasetValidationError, match="NaN"):
        pipeline.transform(bad)


def test_out_of_range_rejected(sample_df, pipeline) -> None:
    bad = sample_df.copy()
    bad.loc[0, "relative_humidity_pct"] = 150
    with pytest.raises(DatasetValidationError, match="Out-of-range"):
        pipeline.transform(bad)
