from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demixing.data.dataset import RamanSpectrumDataset
from demixing.data.manifest import build_sample_manifest
from demixing.data.preprocess import TARGET_AXIS, normalized_value_from_row
from demixing.data.splits import assign_group_split
from demixing.evaluation.baselines import run_anchor_nnls_baseline
from demixing.evaluation.inference import run_inference, save_predictions
from demixing.models.unified_unmixing import UnifiedRamanUnmixingNet, UnifiedUnmixingConfig
from demixing.training.trainer import TrainConfig, train_model
from demixing.visualization.plots import (
    plot_accuracy_comparison,
    plot_average_abundance,
    plot_endmembers,
    plot_family_grouped_scores,
    plot_loss_curve,
    plot_microplastic_score_boxplot,
    plot_model_vs_baseline_scores,
    plot_reconstruction_examples,
    save_experiment_summary,
)


def load_anchor_tensor(data_root: Path, manifest_csv: Path) -> torch.Tensor | None:
    with manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    family_order = {"pure_pp": 0, "pure_pe": 1, "pure_starch": 2}
    pure_rows = [row for row in manifest_rows if row.get("family") in family_order]
    pure_rows = sorted(pure_rows, key=lambda row: family_order[row["family"]])

    anchors = []
    for row in pure_rows:
        path = data_root / row["relative_path"]
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            spectrum_rows = list(csv.DictReader(handle))
        anchors.append([normalized_value_from_row(item) for item in spectrum_rows])
    return torch.tensor(anchors, dtype=torch.float32) if anchors else None


def select_group_ids(manifest_df: pd.DataFrame) -> dict[str, set[str]]:
    labeled_raw = manifest_df[
        (manifest_df["source_kind"] == "raw")
        & (manifest_df["weak_label_available"] == 1)
        & (manifest_df["family"].isin(["pp_starch", "pe_starch", "pp_pe_starch"]))
        & (manifest_df["quality_tier"].isin(["A", "B"]))
    ].copy()
    group_ids = sorted(labeled_raw["sample_group_id"].unique())
    groups = {"train": set(), "val": set(), "test": set()}
    for group_id in group_ids:
        groups[assign_group_split(group_id)].add(group_id)
    return groups


def dataset_counts(dataset: RamanSpectrumDataset) -> dict[str, int]:
    label_counts = pd.Series([int(sample["label"]) for sample in dataset.samples]).value_counts().sort_index()
    return {str(key): int(value) for key, value in label_counts.items()}


def accuracy(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float((df["label"] == df["pred_label"]).mean())


def family_accuracy(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for family, subset in df.groupby("family"):
        out[str(family)] = accuracy(subset)
    return out


def main() -> None:
    data_root = ROOT / "outputs/preprocessing/dataset_preprocessed_v1"
    quality_manifest_csv = data_root / "_reports/quality_manifest.csv"
    sample_manifest_csv = data_root / "_reports/sample_manifest.csv"
    build_sample_manifest(quality_manifest_csv, sample_manifest_csv)

    experiment_root = ROOT / "outputs/experiments/formal_v2"
    training_dir = experiment_root / "training"
    figure_dir = experiment_root / "figures"
    report_dir = experiment_root / "reports"

    manifest_df = pd.read_csv(sample_manifest_csv, encoding="utf-8-sig")
    group_sets = select_group_ids(manifest_df)
    allowed_families = {"pp_starch", "pe_starch", "pp_pe_starch"}

    train_dataset = RamanSpectrumDataset(
        data_root=data_root,
        manifest_csv=sample_manifest_csv,
        use_normalized=True,
        min_quality_tier="B",
        allowed_source_kinds={"raw"},
        allowed_families=allowed_families,
        require_weak_label=True,
        allowed_group_ids=group_sets["train"],
    )
    val_dataset = RamanSpectrumDataset(
        data_root=data_root,
        manifest_csv=sample_manifest_csv,
        use_normalized=True,
        min_quality_tier="B",
        allowed_source_kinds={"raw"},
        allowed_families=allowed_families,
        require_weak_label=True,
        allowed_group_ids=group_sets["val"],
    )
    test_raw_dataset = RamanSpectrumDataset(
        data_root=data_root,
        manifest_csv=sample_manifest_csv,
        use_normalized=True,
        min_quality_tier="B",
        allowed_source_kinds={"raw"},
        allowed_families=allowed_families,
        require_weak_label=True,
        allowed_group_ids=group_sets["test"],
    )
    test_average_dataset = RamanSpectrumDataset(
        data_root=data_root,
        manifest_csv=sample_manifest_csv,
        use_normalized=True,
        min_quality_tier="C",
        allowed_source_kinds={"average"},
        allowed_families=allowed_families,
        require_weak_label=True,
        allowed_group_ids=group_sets["test"],
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_raw_loader = DataLoader(test_raw_dataset, batch_size=128, shuffle=False)
    test_average_loader = DataLoader(test_average_dataset, batch_size=128, shuffle=False)

    anchor_tensor = load_anchor_tensor(data_root, sample_manifest_csv)
    model = UnifiedRamanUnmixingNet(
        UnifiedUnmixingConfig(mode="fixed", n_residual_endmembers=0, hidden_dim=192, latent_dim=96),
        endmember_anchors=anchor_tensor,
    )
    history = train_model(
        model,
        train_loader,
        TrainConfig(
            epochs=24,
            batch_size=32,
            recon_weight=1.0,
            sad_weight=0.4,
            anchor_weight=0.0,
            smooth_weight=8e-4,
            label_weight=0.0,
            interval_weight=4.0,
            forbidden_weight=1.0,
            separation_weight=0.0,
        ),
        training_dir,
        val_dataloader=val_loader,
    )

    training_dir.mkdir(parents=True, exist_ok=True)
    (training_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), training_dir / "model.pt")

    test_raw_predictions, raw_traces = run_inference(
        model,
        test_raw_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        prediction_mode="score",
    )
    test_average_predictions, average_traces = run_inference(
        model,
        test_average_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        prediction_mode="score",
    )

    save_predictions(test_raw_predictions, report_dir / "test_raw_predictions_model.csv")
    save_predictions(test_average_predictions, report_dir / "test_average_predictions_model.csv")

    anchors_np = anchor_tensor.numpy() if anchor_tensor is not None else None
    baseline_raw = run_anchor_nnls_baseline(test_raw_predictions, raw_traces, anchors_np if anchors_np is not None else np.eye(3))
    baseline_average = run_anchor_nnls_baseline(test_average_predictions, average_traces, anchors_np if anchors_np is not None else np.eye(3))
    save_predictions(baseline_raw.predictions, report_dir / "test_raw_predictions_baseline.csv")
    save_predictions(baseline_average.predictions, report_dir / "test_average_predictions_baseline.csv")

    plot_loss_curve(history, figure_dir / "training/loss_curve.png")
    endmembers_np = model.get_endmember_matrix().detach().cpu().numpy()
    plot_endmembers(TARGET_AXIS, endmembers_np, anchors_np, figure_dir / "endmembers/learned_vs_anchor.png")
    plot_reconstruction_examples(TARGET_AXIS, average_traces, test_average_predictions, figure_dir / "reconstruction/average_test_examples.png")
    plot_average_abundance(test_average_predictions, figure_dir / "abundance/average_test_stacked.png")
    plot_microplastic_score_boxplot(test_raw_predictions, figure_dir / "concentration_levels/model_test_raw_boxplot.png")
    plot_model_vs_baseline_scores(test_raw_predictions, baseline_raw.predictions, figure_dir / "comparison/model_vs_baseline_scores.png")
    plot_family_grouped_scores(test_average_predictions, figure_dir / "comparison/family_grouped_scores.png")
    plot_accuracy_comparison(test_raw_predictions, baseline_raw.predictions, figure_dir / "comparison/family_accuracy_comparison.png")

    summary = {
        "experiment": "formal_v2",
        "mode": "fixed+weak_interval",
        "train_raw_samples": len(train_dataset),
        "val_raw_samples": len(val_dataset),
        "test_raw_samples": len(test_raw_dataset),
        "test_average_samples": len(test_average_dataset),
        "epochs": len(history),
        "loss_last": history[-1]["loss"] if history else None,
        "val_loss_last": history[-1].get("val_loss") if history else None,
        "train_label_distribution": dataset_counts(train_dataset),
        "val_label_distribution": dataset_counts(val_dataset),
        "test_label_distribution": dataset_counts(test_raw_dataset),
        "model_test_accuracy": accuracy(test_raw_predictions),
        "baseline_test_accuracy": accuracy(baseline_raw.predictions),
        "model_family_accuracy": family_accuracy(test_raw_predictions),
        "baseline_family_accuracy": family_accuracy(baseline_raw.predictions),
    }
    save_experiment_summary(summary, report_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
