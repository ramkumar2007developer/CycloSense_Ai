"""Overfitting sanity check tests.

Verifies that model architectures have genuine learning capacity:
On a tiny fixed batch of 4 samples, training for 25 epochs must reduce loss by > 50%.
This guarantees that backpropagation and gradient flow are functional.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import pytest

from src.models.numerical_mlp import NumericalMLP
from src.models.fusion import FusionModel


def test_numerical_mlp_overfits_tiny_batch() -> None:
    """NumericalMLP should rapidly memorize 4 samples."""
    torch.manual_seed(42)
    model = NumericalMLP(input_dim=12, hidden_dim=32, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 12)
    y = torch.tensor([1, 0, 1, 0], dtype=torch.long)

    initial_loss = None
    final_loss = None

    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert initial_loss is not None and final_loss is not None
    assert final_loss < initial_loss * 0.4, (
        f"Loss did not decrease sufficiently: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


def test_fusion_model_overfits_tiny_batch() -> None:
    """FusionModel should rapidly memorize 4 multimodal samples."""
    torch.manual_seed(42)
    model = FusionModel(
        num_features=12,
        num_classes=2,
        fusion_dim=32,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    img = torch.randn(4, 3, 224, 224)
    num = torch.randn(4, 12)
    y = torch.tensor([1, 0, 1, 0], dtype=torch.long)

    initial_loss = None
    final_loss = None

    model.train()
    for epoch in range(25):
        optimizer.zero_grad()
        logits = model(img, num)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert initial_loss is not None and final_loss is not None
    assert final_loss < initial_loss * 0.5, (
        f"Fusion loss did not decrease sufficiently: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
