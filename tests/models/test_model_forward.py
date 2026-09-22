"""Model forward pass tests — shape, dtype, batch size correctness.

Tests ImageBaselineCNN, NumericalMLP, and FusionModel forward passes.
No training required. Pure architecture validation.
"""

from __future__ import annotations

import pytest
import torch

from src.models.fusion import FusionModel
from src.models.image_cnn import ImageBaselineCNN
from src.models.numerical_mlp import NumericalMLP

NUM_FEATURES = 12  # matches ml_config.json feature_columns count
NUM_CLASSES = 2
IMAGE_SIZE = 224


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def image_model() -> ImageBaselineCNN:
    model = ImageBaselineCNN(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()
    return model


@pytest.fixture(scope="module")
def num_model() -> NumericalMLP:
    model = NumericalMLP(input_dim=NUM_FEATURES, hidden_dim=64, num_classes=NUM_CLASSES)
    model.eval()
    return model


@pytest.fixture(scope="module")
def fusion_model() -> FusionModel:
    model = FusionModel(num_features=NUM_FEATURES, num_classes=NUM_CLASSES, fusion_dim=128)
    model.eval()
    return model


def _rand_image(batch: int = 1) -> torch.Tensor:
    return torch.rand(batch, 3, IMAGE_SIZE, IMAGE_SIZE)


def _rand_features(batch: int = 1) -> torch.Tensor:
    return torch.rand(batch, NUM_FEATURES)


# --------------------------------------------------------------------------- #
# ImageBaselineCNN
# --------------------------------------------------------------------------- #

class TestImageBaselineCNN:
    def test_output_shape_batch1(self, image_model: ImageBaselineCNN) -> None:
        """Batch of 1 image → (1, 2) logits."""
        with torch.no_grad():
            out = image_model(_rand_image(1))
        assert out.shape == (1, NUM_CLASSES), f"Expected (1, {NUM_CLASSES}), got {tuple(out.shape)}"

    def test_output_shape_batch16(self, image_model: ImageBaselineCNN) -> None:
        """Batch of 16 images → (16, 2) logits."""
        with torch.no_grad():
            out = image_model(_rand_image(16))
        assert out.shape == (16, NUM_CLASSES)

    def test_output_shape_batch64(self, image_model: ImageBaselineCNN) -> None:
        """Batch of 64 images → (64, 2) logits."""
        with torch.no_grad():
            out = image_model(_rand_image(64))
        assert out.shape == (64, NUM_CLASSES)

    def test_embedding_shape(self, image_model: ImageBaselineCNN) -> None:
        """Embedding pooled output must be (batch, 576) for MobileNetV3-Small."""
        with torch.no_grad():
            emb = image_model.embedding(_rand_image(4))
        assert emb.shape == (4, 576), (
            f"Expected embedding shape (4, 576), got {tuple(emb.shape)}. "
            "FusionModel depends on this exact dimension."
        )

    def test_output_dtype_float32(self, image_model: ImageBaselineCNN) -> None:
        with torch.no_grad():
            out = image_model(_rand_image(2))
        assert out.dtype == torch.float32

    def test_no_nan_in_output(self, image_model: ImageBaselineCNN) -> None:
        with torch.no_grad():
            out = image_model(_rand_image(4))
        assert not torch.isnan(out).any(), "NaN detected in ImageBaselineCNN output."

    def test_softmax_sums_to_one(self, image_model: ImageBaselineCNN) -> None:
        """Softmax of logits must sum to 1.0 per sample."""
        with torch.no_grad():
            logits = image_model(_rand_image(8))
            probs = torch.softmax(logits, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# --------------------------------------------------------------------------- #
# NumericalMLP
# --------------------------------------------------------------------------- #

class TestNumericalMLP:
    def test_output_shape_batch1(self, num_model: NumericalMLP) -> None:
        with torch.no_grad():
            out = num_model(_rand_features(1))
        assert out.shape == (1, NUM_CLASSES)

    def test_output_shape_batch16(self, num_model: NumericalMLP) -> None:
        with torch.no_grad():
            out = num_model(_rand_features(16))
        assert out.shape == (16, NUM_CLASSES)

    def test_embedding_shape(self, num_model: NumericalMLP) -> None:
        """Encoder output must be (batch, 64)."""
        with torch.no_grad():
            emb = num_model.embedding(_rand_features(8))
        assert emb.shape == (8, 64)

    def test_no_nan_in_output(self, num_model: NumericalMLP) -> None:
        with torch.no_grad():
            out = num_model(_rand_features(4))
        assert not torch.isnan(out).any()

    def test_all_zeros_input(self, num_model: NumericalMLP) -> None:
        """All-zeros input should not crash or produce NaN."""
        with torch.no_grad():
            out = num_model(torch.zeros(4, NUM_FEATURES))
        assert out.shape == (4, NUM_CLASSES)
        assert not torch.isnan(out).any()

    def test_extreme_positive_input(self, num_model: NumericalMLP) -> None:
        """Large positive values must not produce NaN (dropout off in eval)."""
        with torch.no_grad():
            out = num_model(torch.full((2, NUM_FEATURES), 100.0))
        assert not torch.isnan(out).any()


# --------------------------------------------------------------------------- #
# FusionModel
# --------------------------------------------------------------------------- #

class TestFusionModel:
    def test_output_shape_batch1(self, fusion_model: FusionModel) -> None:
        with torch.no_grad():
            out = fusion_model(_rand_image(1), _rand_features(1))
        assert out.shape == (1, NUM_CLASSES)

    def test_output_shape_batch16(self, fusion_model: FusionModel) -> None:
        with torch.no_grad():
            out = fusion_model(_rand_image(16), _rand_features(16))
        assert out.shape == (16, NUM_CLASSES)

    def test_no_nan_in_output(self, fusion_model: FusionModel) -> None:
        with torch.no_grad():
            out = fusion_model(_rand_image(4), _rand_features(4))
        assert not torch.isnan(out).any()

    def test_image_features_batch_must_match(self, fusion_model: FusionModel) -> None:
        """Mismatched batch sizes must raise a RuntimeError (not silently corrupt)."""
        with pytest.raises((RuntimeError, ValueError)):
            with torch.no_grad():
                fusion_model(_rand_image(4), _rand_features(8))
