"""Image pipeline edge-case tests."""

from pathlib import Path

import pytest
from PIL import Image

from src.data.image_pipeline import build_image_transforms, load_image_rgb, preprocess_image
from src.exceptions import DatasetValidationError


@pytest.fixture
def transform():
    return build_image_transforms(64, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], augment=False)


def test_normal_rgb_image(tmp_path: Path, transform) -> None:
    path = tmp_path / "rgb.jpg"
    Image.new("RGB", (128, 128), color=(10, 20, 30)).save(path)
    tensor = preprocess_image(path, transform)
    assert tensor.shape == (3, 64, 64)


def test_grayscale_converted(tmp_path: Path, transform) -> None:
    path = tmp_path / "gray.jpg"
    Image.new("L", (50, 50), color=100).save(path)
    tensor = preprocess_image(path, transform)
    assert tensor.shape[0] == 3


def test_small_image(tmp_path: Path, transform) -> None:
    path = tmp_path / "small.jpg"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(path)
    tensor = preprocess_image(path, transform)
    assert tensor.shape == (3, 64, 64)


def test_large_image(tmp_path: Path, transform) -> None:
    path = tmp_path / "large.jpg"
    Image.new("RGB", (1024, 768), color=(1, 2, 3)).save(path)
    tensor = preprocess_image(path, transform)
    assert tensor.shape == (3, 64, 64)


def test_corrupted_image(tmp_path: Path) -> None:
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"not-an-image")
    with pytest.raises(DatasetValidationError, match="Corrupted"):
        load_image_rgb(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="Missing image"):
        load_image_rgb(tmp_path / "missing.jpg")


def test_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("not an image", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Corrupted"):
        load_image_rgb(path)
