"""Multimodal fusion model."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.image_cnn import ImageBaselineCNN
from src.models.numerical_mlp import NumericalMLP


class FusionModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        fusion_dim: int = 128,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageBaselineCNN(num_classes=num_classes, pretrained=True)
        self.num_encoder = NumericalMLP(input_dim=num_features, hidden_dim=64, num_classes=num_classes)
        self.image_proj = nn.Linear(576, fusion_dim)
        self.num_proj = nn.Linear(64, fusion_dim)
        self.classifier = nn.Linear(fusion_dim * 2, num_classes)

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        img_emb = self.image_proj(self.image_encoder.embedding(image))
        num_emb = self.num_proj(self.num_encoder.embedding(features))
        fused = torch.cat([img_emb, num_emb], dim=1)
        return self.classifier(fused)
