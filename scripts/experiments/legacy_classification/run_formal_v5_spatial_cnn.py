from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demixing.legacy.data.group_dataset import GroupPCAProjector, SpatialGroupDataset
from demixing.data.manifest import build_sample_manifest
from demixing.legacy.data.splits import assign_group_split
from demixing.legacy.evaluation.inference import save_predictions
from demixing.legacy.models.spatial_cnn import SpatialCNNConfig, SpatialGroupClassifier
from demixing.legacy.training.spatial_trainer import (
    SpatialTrainConfig,
    collate_spatial_batch,
    evaluate_spatial_model,
    predict_spatial_groups,
    train_spatial_model,
)
from demixing.legacy.visualization.plots import plot_confusion_matrix, plot_family_accuracy, plot_loss_curve, save_experiment_summary


def main() -> None:
    data_root = ROOT / "outputs/preprocessing/dataset_preprocessed_v1"
    quality_manifest_csv = data_root / "_reports/quality_manifest.csv"
    sample_manifest_csv = data_root / "_reports/sample_manifest.csv"
    build_sample_manifest(quality_manifest_csv, sample_manifest_csv)

    experiment_root = ROOT / "outputs/experiments/formal_v5_spatial_cnn"
    figure_dir = experiment_root / "figures"
    report_dir = experiment_root / "reports"
    training_dir = experiment_root / "training"

    manifest_df = pd.read_csv(sample_manifest_csv, encoding="utf-8-sig")
    filtered = manifest_df[
        (manifest_df["source_kind"] == "raw")
        & (manifest_df["weak_label_available"] == 1)
        & (manifest_df["quality_tier"].isin(["A", "B"]))
        & (manifest_df["family"].isin(["pp_starch", "pe_starch", "pp_pe_starch"]))
    ].copy()
    filtered["split"] = filtered["sample_group_id"].map(assign_group_split)

    train_df = filtered[filtered["split"] == "train"].copy()
    val_df = filtered[filtered["split"] == "val"].copy()
    test_df = filtered[filtered["split"] == "test"].copy()

    projector = GroupPCAProjector.fit(train_df, data_root, n_components=8, use_normalized=True)
    train_dataset = SpatialGroupDataset(train_df, data_root, projector, use_normalized=True)
    val_dataset = SpatialGroupDataset(val_df, data_root, projector, use_normalized=True)
    test_dataset = SpatialGroupDataset(test_df, data_root, projector, use_normalized=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_spatial_batch)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_spatial_batch)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_spatial_batch)

    model = SpatialGroupClassifier(SpatialCNNConfig(in_channels=9, base_channels=32, num_classes=3))
    history = train_spatial_model(
        model,
        train_loader,
        val_loader,
        SpatialTrainConfig(epochs=40, lr=1e-3),
        training_dir,
    )

    training_dir.mkdir(parents=True, exist_ok=True)
    (training_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), training_dir / "model.pt")

    plot_loss_curve(
        [{"epoch": item["epoch"], "loss": item["train_loss"], "val_loss": item["val_loss"]} for item in history],
        figure_dir / "training/loss_curve.png",
    )

    test_metrics = evaluate_spatial_model(model, test_loader, "cuda" if torch.cuda.is_available() else "cpu")
    pred_rows = predict_spatial_groups(model, test_loader, "cuda" if torch.cuda.is_available() else "cpu")
    pred_df = pd.DataFrame(pred_rows).sort_values(["family", "group_id"]).reset_index(drop=True)
    save_predictions(pred_df, report_dir / "group_predictions.csv")
    plot_confusion_matrix(pred_df, figure_dir / "group_confusion_matrix.png", title="Spatial CNN group confusion matrix")

    family_accuracy = {
        str(family): float((subset["label"] == subset["pred_label"]).mean())
        for family, subset in pred_df.groupby("family")
    }
    plot_family_accuracy(family_accuracy, figure_dir / "group_family_accuracy.png", title="Spatial CNN family accuracy")

    summary = {
        "experiment": "formal_v5_spatial_cnn",
        "primary_metric": "group_accuracy",
        "group_accuracy": float((pred_df["label"] == pred_df["pred_label"]).mean()) if not pred_df.empty else 0.0,
        "test_loss": test_metrics["loss"],
        "train_groups": len(train_dataset),
        "val_groups": len(val_dataset),
        "test_groups": len(test_dataset),
        "family_accuracy": family_accuracy,
        "note": "formal_v5是第一个真正吃二维空间图的小型CNN版本，输入为每个样本组的PCA光谱特征图加占用mask。",
    }
    save_experiment_summary(summary, report_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
