"""Deterministic image preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.exceptions import DatasetValidationError

_SUPPORTED_MODES = {"RGB", "L", "RGBA"}


def load_image_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise DatasetValidationError(f"Missing image file: {path}")
    try:
        image = Image.open(path)
        image.load()
    except Exception as exc:
        raise DatasetValidationError(f"Corrupted or unreadable image: {path}") from exc

    if image.mode not in _SUPPORTED_MODES:
        raise DatasetValidationError(
            f"Unsupported image mode {image.mode!r} for {path.name}"
        )
    return image.convert("RGB")


def build_image_transforms(
    size: int,
    mean: list[float],
    std: list[float],
    augment: bool = False,
) -> transforms.Compose:
    ops: list[transforms.Compose | transforms.RandomHorizontalFlip] = []
    if augment:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        )
    ops.extend(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(ops)


def preprocess_image(
    path: Path,
    transform: transforms.Compose,
) -> torch.Tensor:
    image = load_image_rgb(path)
    tensor = transform(image)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise DatasetValidationError(
            f"Expected CHW RGB tensor shape (3, H, W), got {tuple(tensor.shape)}"
        )
    return tensor
