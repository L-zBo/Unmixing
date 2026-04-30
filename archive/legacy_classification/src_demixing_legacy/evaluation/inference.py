from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


def score_to_label(score: float) -> int:
    if score < 0.4:
        return 0
    if score < 0.75:
        return 1
    return 2


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    prediction_mode: str = "score",
) -> tuple[pd.DataFrame, dict[str, list[list[float]]]]:
    model.eval()
    model.to(device)

    rows: list[dict[str, object]] = []
    traces = {
        "x": [],
        "reconstruction": [],
        "axis": [],
    }
    for batch in dataloader:
        x = batch["x"].to(device)
        mask = batch["microplastic_mask"].to(device)
        allowed_main_mask = batch.get("allowed_main_mask")
        if allowed_main_mask is not None:
            allowed_main_mask = allowed_main_mask.to(device)
        outputs = model(x, microplastic_mask=mask, allowed_main_mask=allowed_main_mask)

        abundances = outputs["abundances"].detach().cpu().numpy()
        microplastic_score = outputs["microplastic_score"].detach().cpu().numpy().reshape(-1)
        label_logits = outputs["label_logits"].detach().cpu().numpy()
        pred_label_head = label_logits.argmax(axis=1)
        pred_label_score = [score_to_label(float(score)) for score in microplastic_score]

        traces["x"].extend(batch["x"].numpy().tolist())
        traces["reconstruction"].extend(outputs["reconstruction"].detach().cpu().numpy().tolist())
        traces["axis"].extend(batch["axis"].numpy().tolist())

        for idx in range(len(abundances)):
            pred_label = pred_label_score[idx] if prediction_mode == "score" else int(pred_label_head[idx])
            row = {
                "relative_path": batch["relative_path"][idx],
                "sample_group_id": batch["sample_group_id"][idx],
                "quality_tier": batch["quality_tier"][idx],
                "family": batch["family"][idx],
                "source_kind": batch["source_kind"][idx],
                "label": int(batch["label"][idx].item()),
                "pred_label": int(pred_label),
                "pred_label_head": int(pred_label_head[idx]),
                "pred_label_score": int(pred_label_score[idx]),
                "weight": float(batch["weight"][idx]),
                "weak_label_available": int(batch["weak_label_available"][idx].item()),
                "microplastic_score": float(microplastic_score[idx]),
            }
            for j, value in enumerate(abundances[idx]):
                row[f"abundance_{j + 1}"] = float(value)
            rows.append(row)

    return pd.DataFrame(rows), traces


def save_predictions(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
