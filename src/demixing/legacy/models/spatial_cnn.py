from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class SpatialCNNConfig:
    in_channels: int
    base_channels: int = 32
    num_classes: int = 3


class SpatialGroupClassifier(nn.Module):
    def __init__(self, config: SpatialCNNConfig) -> None:
        super().__init__()
        c = config.base_channels
        self.backbone = nn.Sequential(
            nn.Conv2d(config.in_channels, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.GELU(),
            nn.Conv2d(c * 2, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 2, c),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(c, config.num_classes),
        )

    def forward(self, image: Tensor) -> Tensor:
        feat = self.backbone(image)
        pooled = self.pool(feat)
        return self.head(pooled)
