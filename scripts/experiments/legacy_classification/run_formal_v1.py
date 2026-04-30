from __future__ import annotations

import csv
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

from demixing.legacy.data.dataset import RamanSpectrumDataset
from demixing.data.preprocess import TARGET_AXIS
from demixing.legacy.evaluation.inference import run_inference, save_predictions
from demixing.legacy.models.unified_unmixing import UnifiedRamanUnmixingNet, UnifiedUnmixingConfig
from demixing.legacy.training.trainer import TrainConfig, train_model
from demixing.legacy.visualization.plots import (
    plot_average_abundance,
    plot_endmembers,
    plot_loss_curve,
    plot_microplastic_score_boxplot,
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
        anchors.append([float(item["Intensity_corrected"]) for item in spectrum_rows])
    return torch.tensor(anchors, dtype=torch.float32) if anchors else None


def main() -> None:
    data_root = ROOT / "outputs/preprocessing/dataset_preprocessed_v1"
    manifest_csv = data_root / "_reports/sample_manifest.csv"
    experiment_root = ROOT / "outputs/experiments/formal_v1"
    training_dir = experiment_root / "training"
    figure_dir = experiment_root / "figures"
    report_dir = experiment_root / "reports"

    allowed_families = {"pp_starch", "pe_starch", "pp_pe_starch"}
    train_dataset = RamanSpectrumDataset(
        data_root=data_root,
        manifest_csv=manifest_csv,
        use_normalized=False,
        min_quality_tier="B",
        allowed_source_kinds={"raw", "average"},
        allowed_families=allowed_families,
        require_weak_label=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    eval_dataset = RamanSpectrumDataset(
        data_root=data_root,
        manifest_csv=manifest_csv,
        use_normalized=False,
        min_quality_tier="C",
        allowed_source_kinds={"average"},
        allowed_families=allowed_families,
        require_weak_label=True,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

    anchor_tensor = load_anchor_tensor(data_root, manifest_csv)
    model = UnifiedRamanUnmixingNet(
        UnifiedUnmixingConfig(mode="weak", n_residual_endmembers=0),
        endmember_anchors=anchor_tensor,
    )
    history = train_model(
        model,
        train_loader,
        TrainConfig(
            epochs=12,
            batch_size=32,
            sad_weight=0.4,
            label_weight=0.5,
            anchor_weight=0.5,
            smooth_weight=5e-4,
        ),
        training_dir,
    )

    training_dir.mkdir(parents=True, exist_ok=True)
    (training_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), training_dir / "model.pt")

    prediction_df, traces = run_inference(model, eval_loader, device="cuda" if torch.cuda.is_available() else "cpu")
    save_predictions(prediction_df, report_dir / "average_predictions.csv")

    plot_loss_curve(history, figure_dir / "training/loss_curve.png")
    anchors_np = anchor_tensor.numpy() if anchor_tensor is not None else None
    endmembers_np = model.get_endmember_matrix().detach().cpu().numpy()
    plot_endmembers(TARGET_AXIS, endmembers_np, anchors_np, figure_dir / "endmembers/learned_vs_anchor.png")
    plot_reconstruction_examples(TARGET_AXIS, traces, prediction_df, figure_dir / "reconstruction/average_examples.png")
    plot_average_abundance(prediction_df, figure_dir / "abundance/average_stacked.png")

    all_eval_dataset = RamanSpectrumDataset(
        data_root=data_root,
        manifest_csv=manifest_csv,
        use_normalized=False,
        min_quality_tier="B",
        allowed_source_kinds={"raw", "average"},
        allowed_families=allowed_families,
        require_weak_label=True,
    )
    all_eval_loader = DataLoader(all_eval_dataset, batch_size=128, shuffle=False)
    all_prediction_df, _ = run_inference(model, all_eval_loader, device="cuda" if torch.cuda.is_available() else "cpu")
    save_predictions(all_prediction_df, report_dir / "all_labeled_predictions.csv")
    plot_microplastic_score_boxplot(all_prediction_df, figure_dir / "concentration_levels/microplastic_score_boxplot.png")

    summary = {
        "experiment": "formal_v1",
        "mode": "weak",
        "train_samples": len(train_dataset),
        "eval_average_samples": len(eval_dataset),
        "all_eval_samples": len(all_eval_dataset),
        "epochs": len(history),
        "loss_last": history[-1]["loss"] if history else None,
        "label_distribution_train": {
            str(key): int(value)
            for key, value in pd.Series([int(sample["label"]) for sample in train_dataset.samples]).value_counts().sort_index().items()
        },
        "quality_distribution_train": {
            str(key): int(value)
            for key, value in pd.Series([str(sample["quality_tier"]) for sample in train_dataset.samples]).value_counts().sort_index().items()
        },
    }
    save_experiment_summary(summary, report_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
