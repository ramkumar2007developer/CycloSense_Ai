"""Image-only baseline CNN."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ImageBaselineCNN(nn.Module):
    """Lightweight MobileNetV3-Small classifier."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.features(x)
        pooled = self.model.avgpool(features)
        return torch.flatten(pooled, 1)
