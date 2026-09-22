"""Numerical-only baseline MLP."""

from __future__ import annotations

import torch
import torch.nn as nn


class NumericalMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
