from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from demixing.legacy.training.losses import (
    anchor_penalty,
    concentration_interval_loss,
    endmember_separation_loss,
    family_forbidden_abundance_loss,
    ordinal_label_loss,
    smoothness_penalty,
    spectral_angle_loss,
)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 16
    recon_weight: float = 1.0
    sad_weight: float = 0.2
    anchor_weight: float = 0.1
    smooth_weight: float = 1e-4
    label_weight: float = 0.2
    interval_weight: float = 0.3
    forbidden_weight: float = 0.3
    separation_weight: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    config: TrainConfig,
    run_dir: Path,
    val_dataloader: DataLoader | None = None,
) -> list[dict[str, float]]:
    model.to(config.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    mse = nn.MSELoss(reduction="none")
    history: list[dict[str, float]] = []
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            x = batch["x"].to(config.device)
            weights = batch["weight"].to(config.device).float().view(-1, 1)
            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(config.device)
            microplastic_mask = batch.get("microplastic_mask")
            if microplastic_mask is not None:
                microplastic_mask = microplastic_mask.to(config.device)
            allowed_main_mask = batch.get("allowed_main_mask")
            if allowed_main_mask is not None:
                allowed_main_mask = allowed_main_mask.to(config.device)

            outputs = model(x, microplastic_mask=microplastic_mask, allowed_main_mask=allowed_main_mask)
            recon = outputs["reconstruction"]
            endmembers = outputs["endmembers"]

            recon_loss = (mse(recon, x).mean(dim=1, keepdim=True) * weights).mean()
            sad_loss = spectral_angle_loss(recon, x)
            anc_loss = anchor_penalty(endmembers, getattr(model, "endmember_anchors", torch.empty(0, device=config.device)), config.anchor_weight)
            smt_loss = smoothness_penalty(endmembers, config.smooth_weight)
            lbl_loss = ordinal_label_loss(outputs["label_logits"], labels, config.label_weight)
            int_loss = concentration_interval_loss(outputs["microplastic_score"], labels, config.interval_weight)
            fam_loss = family_forbidden_abundance_loss(outputs["abundances"], allowed_main_mask, model.config.n_main_endmembers, config.forbidden_weight)
            sep_loss = endmember_separation_loss(endmembers, model.config.n_main_endmembers, config.separation_weight)

            loss = (
                config.recon_weight * recon_loss
                + config.sad_weight * sad_loss
                + anc_loss
                + smt_loss
                + lbl_loss
                + int_loss
                + fam_loss
                + sep_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        epoch_record = {"epoch": epoch + 1, "loss": epoch_loss / max(len(dataloader), 1)}
        if val_dataloader is not None:
            epoch_record["val_loss"] = evaluate_model(model, val_dataloader, config)
        history.append(epoch_record)
    return history


@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader, config: TrainConfig) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="none")
    total = 0.0
    count = 0
    for batch in dataloader:
        x = batch["x"].to(config.device)
        weights = batch["weight"].to(config.device).float().view(-1, 1)
        labels = batch.get("label")
        if labels is not None:
            labels = labels.to(config.device)
        microplastic_mask = batch.get("microplastic_mask")
        if microplastic_mask is not None:
            microplastic_mask = microplastic_mask.to(config.device)
        allowed_main_mask = batch.get("allowed_main_mask")
        if allowed_main_mask is not None:
            allowed_main_mask = allowed_main_mask.to(config.device)

        outputs = model(x, microplastic_mask=microplastic_mask, allowed_main_mask=allowed_main_mask)
        recon = outputs["reconstruction"]
        endmembers = outputs["endmembers"]

        recon_loss = (mse(recon, x).mean(dim=1, keepdim=True) * weights).mean()
        sad_loss = spectral_angle_loss(recon, x)
        anc_loss = anchor_penalty(endmembers, getattr(model, "endmember_anchors", torch.empty(0, device=config.device)), config.anchor_weight)
        smt_loss = smoothness_penalty(endmembers, config.smooth_weight)
        lbl_loss = ordinal_label_loss(outputs["label_logits"], labels, config.label_weight)
        int_loss = concentration_interval_loss(outputs["microplastic_score"], labels, config.interval_weight)
        fam_loss = family_forbidden_abundance_loss(outputs["abundances"], allowed_main_mask, model.config.n_main_endmembers, config.forbidden_weight)
        sep_loss = endmember_separation_loss(endmembers, model.config.n_main_endmembers, config.separation_weight)
        loss = (
            config.recon_weight * recon_loss
            + config.sad_weight * sad_loss
            + anc_loss
            + smt_loss
            + lbl_loss
            + int_loss
            + fam_loss
            + sep_loss
        )
        total += float(loss.item())
        count += 1
    model.train()
    return total / max(count, 1)
