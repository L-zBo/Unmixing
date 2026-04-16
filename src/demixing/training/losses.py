from __future__ import annotations

import torch
from torch import Tensor, nn


def spectral_angle_loss(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    x_norm = torch.norm(x, dim=-1).clamp_min(eps)
    y_norm = torch.norm(y, dim=-1).clamp_min(eps)
    cosine = torch.sum(x * y, dim=-1) / (x_norm * y_norm)
    cosine = cosine.clamp(-1 + eps, 1 - eps)
    return torch.mean(torch.acos(cosine))


def anchor_penalty(endmembers: Tensor, anchors: Tensor, weight: float = 1.0) -> Tensor:
    if anchors.numel() == 0:
        return endmembers.new_tensor(0.0)
    main = endmembers[: anchors.shape[0]]
    return weight * torch.mean((main - anchors) ** 2)


def smoothness_penalty(endmembers: Tensor, weight: float = 1e-4) -> Tensor:
    diff = endmembers[:, 1:] - endmembers[:, :-1]
    return weight * torch.mean(torch.abs(diff))


def ordinal_label_loss(label_logits: Tensor, labels: Tensor | None, weight: float = 1.0) -> Tensor:
    if labels is None:
        return label_logits.new_tensor(0.0)
    valid_mask = labels >= 0
    if not torch.any(valid_mask):
        return label_logits.new_tensor(0.0)
    criterion = nn.CrossEntropyLoss()
    return weight * criterion(label_logits[valid_mask], labels[valid_mask])
