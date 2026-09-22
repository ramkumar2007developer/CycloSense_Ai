"""Model robustness tests.

Tests that small, irrelevant perturbations do not catastrophically flip predictions.
This is NOT testing that the model is insensitive — it is testing that it is
not wildly brittle to minor variations.

Robustness principle:
- A well-behaved model should not flip class when input changes by epsilon.
- Robustness here means "prediction does not completely collapse for minor
  irrelevant perturbations" — not perfect invariance.

These tests use untrained (random weight) models. For a real trained model,
thresholds would differ. The goal here is to catch catastrophic architectural
failures (e.g., division by zero, exploding activations).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image, ImageEnhance

from src.data.image_pipeline import build_image_transforms, preprocess_image
from src.models.fusion import FusionModel
from src.models.image_cnn import ImageBaselineCNN
from src.models.numerical_mlp import NumericalMLP

NUM_FEATURES = 12
IMAGE_SIZE = 224
TRANSFORM = build_image_transforms(
    IMAGE_SIZE,
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
    augment=False,
)


def _base_image_tensor() -> torch.Tensor:
    """Create a repeatable base image (synthetic blue-grey gradient)."""
    torch.manual_seed(42)
    return torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)


def _base_features() -> torch.Tensor:
    """Create repeatable base numerical features."""
    torch.manual_seed(42)
    return torch.rand(1, NUM_FEATURES)


@pytest.fixture(scope="module")
def image_model() -> ImageBaselineCNN:
    torch.manual_seed(99)
    model = ImageBaselineCNN(num_classes=2, pretrained=False)
    model.eval()
    return model


@pytest.fixture(scope="module")
def num_model() -> NumericalMLP:
    torch.manual_seed(99)
    model = NumericalMLP(input_dim=NUM_FEATURES)
    model.eval()
    return model


@pytest.fixture(scope="module")
def fusion_model() -> FusionModel:
    torch.manual_seed(99)
    model = FusionModel(num_features=NUM_FEATURES)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Image robustness tests
# --------------------------------------------------------------------------- #

class TestImageRobustness:
    def test_brightness_change_no_nan(self, image_model: ImageBaselineCNN) -> None:
        """Slightly brightened image must not produce NaN logits."""
        x = _base_image_tensor()
        x_bright = torch.clamp(x * 1.1, 0, 1)
        with torch.no_grad():
            out_orig = image_model(x)
            out_bright = image_model(x_bright)
        assert not torch.isnan(out_orig).any()
        assert not torch.isnan(out_bright).any()

    def test_mild_gaussian_noise_no_nan(self, image_model: ImageBaselineCNN) -> None:
        """Adding N(0, 0.05) Gaussian noise must not produce NaN logits."""
        x = _base_image_tensor()
        noise = torch.randn_like(x) * 0.05
        x_noisy = torch.clamp(x + noise, 0, 1)
        with torch.no_grad():
            out = image_model(x_noisy)
        assert not torch.isnan(out).any()

    def test_brightness_change_does_not_completely_flip_class(
        self, image_model: ImageBaselineCNN, tmp_path
    ) -> None:
        """Mild brightness change should not flip classification on all 10 test images."""
        flip_count = 0
        for i in range(10):
            img = Image.new("RGB", (64, 64), color=(i * 20, i * 15, i * 10))
            path = tmp_path / f"test_{i}.jpg"
            img.save(path)

            t = TRANSFORM
            x = preprocess_image(path, t).unsqueeze(0)

            # Brighter version
            bright = Image.open(path).convert("RGB")
            bright = ImageEnhance.Brightness(bright).enhance(1.2)
            path_b = tmp_path / f"bright_{i}.jpg"
            bright.save(path_b)
            x_b = preprocess_image(path_b, t).unsqueeze(0)

            with torch.no_grad():
                cls_orig = int(torch.softmax(image_model(x), dim=1)[0, 1] >= 0.5)
                cls_bright = int(torch.softmax(image_model(x_b), dim=1)[0, 1] >= 0.5)
            if cls_orig != cls_bright:
                flip_count += 1

        # Allow flips — just ensure it's not ALL 10 images
        # (untrained model: some flips expected due to randomness, not brightness sensitivity)
        # With a trained model this threshold would be 0-1. With random weights, ≤8 is acceptable.
        assert flip_count <= 8, (
            f"Brightness change flipped classification on {flip_count}/10 images. "
            "Possible architectural instability."
        )

    def test_slight_resize_no_crash(self, tmp_path) -> None:
        """Image resized slightly before pipeline should process fine."""
        img = Image.new("RGB", (200, 200), color=(100, 150, 200))
        path = tmp_path / "resize_test.jpg"
        img.save(path)
        tensor = preprocess_image(path, TRANSFORM)
        assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert not torch.isnan(tensor).any()


# --------------------------------------------------------------------------- #
# Numerical robustness tests
# --------------------------------------------------------------------------- #

class TestNumericalRobustness:
    def test_small_perturbation_no_nan(self, num_model: NumericalMLP) -> None:
        """Adding ε=0.001 perturbation to features must not produce NaN."""
        x = _base_features()
        x_perturbed = x + 0.001 * torch.randn_like(x)
        with torch.no_grad():
            out = num_model(x_perturbed)
        assert not torch.isnan(out).any()

    def test_zero_features_no_nan(self, num_model: NumericalMLP) -> None:
        """All-zero features must not produce NaN."""
        with torch.no_grad():
            out = num_model(torch.zeros(1, NUM_FEATURES))
        assert not torch.isnan(out).any()

    def test_extreme_valid_features_no_nan(self, num_model: NumericalMLP) -> None:
        """Scaled features representing extreme (but valid after normalization) range."""
        # Simulate StandardScaler output for extreme raw values → large scaled values
        x = torch.full((1, NUM_FEATURES), 5.0)  # 5 std deviations from mean
        with torch.no_grad():
            out = num_model(x)
        assert not torch.isnan(out).any()

    def test_irrelevant_feature_variation_no_class_collapse(
        self, num_model: NumericalMLP
    ) -> None:
        """Varying a single feature across its full range should not all map to same class."""
        results = []
        for val in np.linspace(-3, 3, 20):
            x = torch.zeros(1, NUM_FEATURES)
            x[0, 0] = float(val)  # Vary only feature 0
            with torch.no_grad():
                prob = float(torch.softmax(num_model(x), dim=1)[0, 1])
            results.append(prob)

        # Not all predictions should be identical — there should be some variation
        unique_predictions = len(set(round(r, 3) for r in results))
        assert unique_predictions > 1, (
            "Model completely ignores feature variation — possible dead network."
        )


# --------------------------------------------------------------------------- #
# Fusion robustness tests
# --------------------------------------------------------------------------- #

class TestFusionRobustness:
    def test_combined_perturbation_no_nan(self, fusion_model: FusionModel) -> None:
        """Mild noise on both image and features should not produce NaN."""
        x = _base_image_tensor()
        f = _base_features()
        x_noisy = torch.clamp(x + torch.randn_like(x) * 0.02, 0, 1)
        f_noisy = f + torch.randn_like(f) * 0.01
        with torch.no_grad():
            out = fusion_model(x_noisy, f_noisy)
        assert not torch.isnan(out).any()

    def test_risk_index_no_nan_on_perturbed_probs(self) -> None:
        """Perturbed probabilities near boundary must not produce NaN in risk score."""
        from src.scoring.risk_index import ScoringWeights, compute_risk_index
        weights = ScoringWeights(0.4, 0.3, 0.1, 0.1, 0.05, 0.05)
        env = {"cloud_organization_index": 0.5, "convection_index": 0.5,
               "relative_humidity_pct": 50, "rotation_index": 0.3, "persistence_index": 0.3}
        for p in [0.499, 0.500, 0.501, 0.001, 0.999]:
            probs = np.array([1 - p, p])
            result = compute_risk_index(probs, env, weights, "synthetic_mvp")
            assert not np.isnan(result.development_risk_index)
            assert 0 <= result.development_risk_index <= 100
