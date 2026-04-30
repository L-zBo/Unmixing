from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class SpatialTrainConfig:
    lr: float = 1e-3
    epochs: int = 40
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_spatial_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: SpatialTrainConfig,
    run_dir: Path,
) -> list[dict[str, float]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            x = batch["image"].to(config.device)
            y = batch["label"].to(config.device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())

        val_metrics = evaluate_spatial_model(model, val_loader, config.device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / max(len(train_loader), 1),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
    return history


@torch.no_grad()
def evaluate_spatial_model(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        total_loss += float(loss.item())
        total += int(y.numel())
        correct += int((pred == y).sum().item())
    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def predict_spatial_groups(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    rows = []
    for batch in loader:
        x = batch["image"].to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        for idx in range(len(pred)):
            rows.append(
                {
                    "group_id": batch["group_id"][idx],
                    "family": batch["family"][idx],
                    "label": int(batch["label"][idx].item()),
                    "pred_label": int(pred[idx]),
                    "prob_low": float(probs[idx, 0]),
                    "prob_medium": float(probs[idx, 1]),
                    "prob_high": float(probs[idx, 2]),
                }
            )
    return rows


def collate_spatial_batch(batch):
    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)
    images = []
    labels = []
    group_ids = []
    families = []
    for item in batch:
        image = item["image"]
        pad_h = max_h - image.shape[1]
        pad_w = max_w - image.shape[2]
        padded = F.pad(image, (0, pad_w, 0, pad_h))
        images.append(padded)
        labels.append(item["label"])
        group_ids.append(item["group_id"])
        families.append(item["family"])
    return {
        "image": torch.stack(images, dim=0),
        "label": torch.stack(labels, dim=0),
        "group_id": group_ids,
        "family": families,
    }
