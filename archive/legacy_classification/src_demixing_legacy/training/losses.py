from __future__ import annotations

import torch
import torch.nn.functional as F
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


def concentration_interval_loss(
    microplastic_score: Tensor,
    labels: Tensor | None,
    weight: float = 1.0,
) -> Tensor:
    if labels is None:
        return microplastic_score.new_tensor(0.0)
    valid_mask = labels >= 0
    if not torch.any(valid_mask):
        return microplastic_score.new_tensor(0.0)
    scores = microplastic_score.view(-1)[valid_mask]
    label_values = labels[valid_mask]

    lower_bounds = torch.zeros_like(scores)
    upper_bounds = torch.ones_like(scores)
    lower_bounds = torch.where(label_values == 0, torch.full_like(scores, 0.1), lower_bounds)
    upper_bounds = torch.where(label_values == 0, torch.full_like(scores, 0.4), upper_bounds)
    lower_bounds = torch.where(label_values == 1, torch.full_like(scores, 0.4), lower_bounds)
    upper_bounds = torch.where(label_values == 1, torch.full_like(scores, 0.7), upper_bounds)
    lower_bounds = torch.where(label_values == 2, torch.full_like(scores, 0.8), lower_bounds)
    upper_bounds = torch.where(label_values == 2, torch.full_like(scores, 1.0), upper_bounds)

    lower_violation = F.relu(lower_bounds - scores)
    upper_violation = F.relu(scores - upper_bounds)
    return weight * torch.mean(lower_violation.pow(2) + upper_violation.pow(2))


def family_forbidden_abundance_loss(
    abundances: Tensor,
    allowed_main_mask: Tensor | None,
    n_main_endmembers: int,
    weight: float = 1.0,
) -> Tensor:
    if allowed_main_mask is None:
        return abundances.new_tensor(0.0)
    main_abundances = abundances[:, :n_main_endmembers]
    forbidden_mask = 1.0 - allowed_main_mask[:, :n_main_endmembers]
    return weight * torch.mean((main_abundances * forbidden_mask).pow(2))


def endmember_separation_loss(endmembers: Tensor, n_main_endmembers: int, weight: float = 1.0) -> Tensor:
    main = endmembers[:n_main_endmembers]
    if main.shape[0] <= 1:
        return endmembers.new_tensor(0.0)
    normalized = F.normalize(main, dim=1)
    similarity = normalized @ normalized.T
    eye = torch.eye(similarity.shape[0], device=similarity.device, dtype=similarity.dtype)
    off_diag = similarity - eye
    return weight * torch.mean(off_diag.pow(2))
