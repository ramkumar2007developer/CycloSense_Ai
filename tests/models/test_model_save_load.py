"""Model save, load, and prediction consistency tests.

Verifies that:
- Models can be saved as state_dicts and reloaded
- Loaded models produce identical predictions (deterministic)
- Scaler can be saved and reloaded with joblib
- Loading wrong architecture raises a clear error
"""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.models.fusion import FusionModel
from src.models.image_cnn import ImageBaselineCNN
from src.models.numerical_mlp import NumericalMLP

NUM_FEATURES = 12
NUM_CLASSES = 2
IMAGE_SIZE = 224


def _rand_image(batch: int = 1) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(batch, 3, IMAGE_SIZE, IMAGE_SIZE)


def _rand_features(batch: int = 1) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(batch, NUM_FEATURES)


# --------------------------------------------------------------------------- #
# ImageBaselineCNN save / load / consistency
# --------------------------------------------------------------------------- #

class TestImageModelSaveLoad:
    def test_save_and_load_state_dict(self, tmp_path) -> None:
        """Save model weights, load into fresh model, verify outputs match."""
        original = ImageBaselineCNN(num_classes=NUM_CLASSES, pretrained=False)
        original.eval()
        path = tmp_path / "image_model.pt"
        torch.save(original.state_dict(), path)
        assert path.is_file(), "Model file was not created."

        loaded = ImageBaselineCNN(num_classes=NUM_CLASSES, pretrained=False)
        loaded.load_state_dict(torch.load(path, map_location="cpu"))
        loaded.eval()

        x = _rand_image(4)
        with torch.no_grad():
            out_original = original(x)
            out_loaded = loaded(x)

        assert torch.allclose(out_original, out_loaded, atol=1e-6), (
            "Loaded ImageBaselineCNN produces different predictions than the saved model."
        )

    def test_loaded_model_deterministic(self, tmp_path) -> None:
        """Same input twice after load produces identical output."""
        model = ImageBaselineCNN(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        path = tmp_path / "img.pt"
        torch.save(model.state_dict(), path)

        loaded = ImageBaselineCNN(num_classes=NUM_CLASSES, pretrained=False)
        loaded.load_state_dict(torch.load(path, map_location="cpu"))
        loaded.eval()

        x = _rand_image(2)
        with torch.no_grad():
            out1 = loaded(x)
            out2 = loaded(x)
        assert torch.allclose(out1, out2), "Non-deterministic output from loaded model."

    def test_wrong_architecture_raises(self, tmp_path) -> None:
        """Loading weights from wrong architecture raises RuntimeError."""
        small = NumericalMLP(input_dim=NUM_FEATURES)
        path = tmp_path / "wrong.pt"
        torch.save(small.state_dict(), path)

        img_model = ImageBaselineCNN(num_classes=NUM_CLASSES, pretrained=False)
        with pytest.raises((RuntimeError, Exception)):
            img_model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)


# --------------------------------------------------------------------------- #
# NumericalMLP save / load / consistency
# --------------------------------------------------------------------------- #

class TestNumericalModelSaveLoad:
    def test_save_and_load(self, tmp_path) -> None:
        original = NumericalMLP(input_dim=NUM_FEATURES)
        original.eval()
        path = tmp_path / "num_model.pt"
        torch.save(original.state_dict(), path)

        loaded = NumericalMLP(input_dim=NUM_FEATURES)
        loaded.load_state_dict(torch.load(path, map_location="cpu"))
        loaded.eval()

        x = _rand_features(8)
        with torch.no_grad():
            assert torch.allclose(original(x), loaded(x), atol=1e-6)

    def test_prediction_consistent_after_reload(self, tmp_path) -> None:
        """Two separate loads of same weights produce same predictions."""
        model = NumericalMLP(input_dim=NUM_FEATURES)
        model.eval()
        path = tmp_path / "num2.pt"
        torch.save(model.state_dict(), path)

        m1 = NumericalMLP(input_dim=NUM_FEATURES)
        m2 = NumericalMLP(input_dim=NUM_FEATURES)
        m1.load_state_dict(torch.load(path, map_location="cpu"))
        m2.load_state_dict(torch.load(path, map_location="cpu"))
        m1.eval(); m2.eval()

        x = _rand_features(4)
        with torch.no_grad():
            assert torch.allclose(m1(x), m2(x), atol=1e-6)


# --------------------------------------------------------------------------- #
# FusionModel save / load / consistency
# --------------------------------------------------------------------------- #

class TestFusionModelSaveLoad:
    def test_save_and_load(self, tmp_path) -> None:
        original = FusionModel(num_features=NUM_FEATURES)
        original.eval()
        path = tmp_path / "fusion.pt"
        torch.save(original.state_dict(), path)

        loaded = FusionModel(num_features=NUM_FEATURES)
        loaded.load_state_dict(torch.load(path, map_location="cpu"))
        loaded.eval()

        img = _rand_image(4)
        feat = _rand_features(4)
        with torch.no_grad():
            assert torch.allclose(original(img, feat), loaded(img, feat), atol=1e-6)

    def test_fusion_deterministic_after_reload(self, tmp_path) -> None:
        model = FusionModel(num_features=NUM_FEATURES)
        model.eval()
        path = tmp_path / "f2.pt"
        torch.save(model.state_dict(), path)

        loaded = FusionModel(num_features=NUM_FEATURES)
        loaded.load_state_dict(torch.load(path, map_location="cpu"))
        loaded.eval()

        img = _rand_image(2)
        feat = _rand_features(2)
        with torch.no_grad():
            o1 = loaded(img, feat)
            o2 = loaded(img, feat)
        assert torch.allclose(o1, o2)


# --------------------------------------------------------------------------- #
# Numerical scaler save / load
# --------------------------------------------------------------------------- #

class TestScalerSaveLoad:
    def test_scaler_save_load_produces_same_transform(self, tmp_path) -> None:
        """Joblib-saved scaler reproduces exact same scaled values after reload."""
        scaler = StandardScaler()
        rng = np.random.default_rng(42)
        x_train = rng.uniform(0, 1, (50, NUM_FEATURES)).astype(np.float32)
        x_test = rng.uniform(0, 1, (10, NUM_FEATURES)).astype(np.float32)
        scaler.fit(x_train)
        expected = scaler.transform(x_test)

        path = tmp_path / "scaler.joblib"
        joblib.dump(scaler, path)

        loaded_scaler = joblib.load(path)
        result = loaded_scaler.transform(x_test)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_unfitted_scaler_raises_on_transform(self) -> None:
        """A freshly created (unfitted) NumericalPipeline must raise before transform."""
        from src.data.numerical_pipeline import NumericalPipeline
        from src.exceptions import DatasetValidationError
        import pandas as pd

        cols = [
            "temperature_c", "sea_surface_temperature_c", "relative_humidity_pct",
            "water_vapour_gkg", "surface_pressure_hpa", "wind_speed_ms",
            "wind_direction_deg", "vertical_wind_shear_ms", "cloud_organization_index",
            "convection_index", "rotation_index", "persistence_index",
        ]
        df = pd.DataFrame([{
            "temperature_c": 28.0,
            "sea_surface_temperature_c": 29.0,
            "relative_humidity_pct": 80.0,
            "water_vapour_gkg": 20.0,
            "surface_pressure_hpa": 1005.0,
            "wind_speed_ms": 25.0,
            "wind_direction_deg": 180.0,
            "vertical_wind_shear_ms": 10.0,
            "cloud_organization_index": 0.6,
            "convection_index": 0.5,
            "rotation_index": 0.4,
            "persistence_index": 0.5,
        }])
        pipeline = NumericalPipeline(cols, "cyclone_label", "storm_id", StandardScaler(), "synthetic_mvp")

        with pytest.raises(DatasetValidationError, match="fit()"):
            pipeline.transform(df)
