from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def run_inference(model: torch.nn.Module, dataloader: DataLoader, device: str) -> tuple[pd.DataFrame, dict[str, list[list[float]]]]:
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
        outputs = model(x, microplastic_mask=mask)

        abundances = outputs["abundances"].detach().cpu().numpy()
        microplastic_score = outputs["microplastic_score"].detach().cpu().numpy().reshape(-1)
        label_logits = outputs["label_logits"].detach().cpu().numpy()
        pred_label = label_logits.argmax(axis=1)

        traces["x"].extend(batch["x"].numpy().tolist())
        traces["reconstruction"].extend(outputs["reconstruction"].detach().cpu().numpy().tolist())
        traces["axis"].extend(batch["axis"].numpy().tolist())

        for idx in range(len(abundances)):
            row = {
                "relative_path": batch["relative_path"][idx],
                "quality_tier": batch["quality_tier"][idx],
                "family": batch["family"][idx],
                "source_kind": batch["source_kind"][idx],
                "label": int(batch["label"][idx].item()),
                "pred_label": int(pred_label[idx]),
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
