"""Numerical feature pipeline with train-only scaler fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config.ml_config import MLConfig
from src.data.audit import _resolve_env_csv
from src.exceptions import DatasetValidationError, ValidationError

FEATURE_RANGES = {
    "temperature_c": (-100.0, 60.0),
    "sea_surface_temperature_c": (-5.0, 45.0),
    "relative_humidity_pct": (0.0, 100.0),
    "water_vapour_gkg": (0.0, 100.0),
    "surface_pressure_hpa": (800.0, 1100.0),
    "wind_speed_ms": (0.0, 120.0),
    "wind_direction_deg": (0.0, 360.0),
    "vertical_wind_shear_ms": (0.0, 50.0),
    "cloud_organization_index": (0.0, 1.0),
    "convection_index": (0.0, 1.0),
    "rotation_index": (0.0, 1.0),
    "persistence_index": (0.0, 1.0),
}


@dataclass
class NumericalPipeline:
    """Numerical feature pipeline: validates, scales, and transforms features.

    IMPORTANT: Call fit() or fit_transform() on training data ONLY.
    Call transform() on validation and test data using the train-fitted scaler.
    Never fit on the full dataset before splitting (data leakage).

    data_source_tag: Always "synthetic_mvp" for this project.
    """

    feature_columns: list[str]
    label_column: str
    group_column: str
    scaler: StandardScaler
    data_source_tag: str

    def _is_fitted(self) -> bool:
        """Return True if the scaler has been fitted on training data."""
        return hasattr(self.scaler, "mean_") and self.scaler.mean_ is not None

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit scaler on training data. Must be called before transform()."""
        x = self._validate_features(train_df)
        self.scaler.fit(x)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features using the train-fitted scaler.

        Raises:
            DatasetValidationError: If df fails validation or if scaler has not been fitted yet.
        """
        x = self._validate_features(df)
        if not self._is_fitted():
            raise DatasetValidationError(
                "NumericalPipeline.transform() called before fit(). "
                "Call fit(train_df) on training data first to prevent data leakage."
            )
        return self.scaler.transform(x)

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        """Fit on training data then transform it. Convenience for training split."""
        self.fit(train_df)
        return self.scaler.transform(self._validate_features(train_df))

    def _validate_features(self, df: pd.DataFrame) -> np.ndarray:
        """Validate dataframe features and return float32 numpy array.

        Checks:
        - All feature columns present
        - No NaN values
        - No infinity values
        - Values within known physical/synthetic_mvp ranges
        """
        missing_cols = [c for c in self.feature_columns if c not in df.columns]
        if missing_cols:
            raise DatasetValidationError(f"Missing numerical columns: {missing_cols}")

        matrix = df[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        if matrix.isna().any().any():
            bad = matrix.isna().sum()
            cols = bad[bad > 0].to_dict()
            raise DatasetValidationError(f"NaN/invalid numerical values in columns: {cols}")

        if np.isinf(matrix.to_numpy()).any():
            raise DatasetValidationError("Infinity values found in numerical features.")

        for col in self.feature_columns:
            if col not in FEATURE_RANGES:
                continue
            low, high = FEATURE_RANGES[col]
            values = matrix[col]
            if ((values < low) | (values > high)).any():
                raise DatasetValidationError(
                    f"Out-of-range values in {col}; expected [{low}, {high}] "
                    f"for synthetic_mvp validation."
                )
        return matrix.to_numpy(dtype=np.float32)


def load_environmental_dataframe(config: MLConfig) -> pd.DataFrame:
    """Load the synthetic_mvp environmental CSV.

    NOTE: data_source_tag is always 'synthetic_mvp' — not real measurements.
    """
    csv_path = _resolve_env_csv(config)
    df = pd.read_csv(csv_path)
    df.attrs["data_source_tag"] = config.data_source_tag
    return df


def build_numerical_pipeline(config: MLConfig) -> NumericalPipeline:
    """Build NumericalPipeline from config. Scaler is unfitted at this point."""
    num_cfg = config.raw["numerical"]
    return NumericalPipeline(
        feature_columns=list(num_cfg["feature_columns"]),
        label_column=num_cfg["label_column"],
        group_column=num_cfg["group_column"],
        scaler=StandardScaler(),
        data_source_tag=config.data_source_tag,
    )


def labels_from_frame(df: pd.DataFrame, label_column: str) -> np.ndarray:
    """Extract integer labels from a dataframe column."""
    if label_column not in df.columns:
        raise ValidationError(f"Label column {label_column!r} not found.")
    return df[label_column].astype(int).to_numpy()
