from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demixing.data.manifest import build_sample_manifest
from demixing.data.splits import assign_group_split
from demixing.evaluation.classical_models import run_family_specific_svc
from demixing.evaluation.inference import save_predictions
from demixing.visualization.plots import plot_confusion_matrix, save_experiment_summary


def main() -> None:
    data_root = ROOT / "outputs/preprocessing/dataset_preprocessed_v1"
    quality_manifest_csv = data_root / "_reports/quality_manifest.csv"
    sample_manifest_csv = data_root / "_reports/sample_manifest.csv"
    build_sample_manifest(quality_manifest_csv, sample_manifest_csv)

    experiment_root = ROOT / "outputs/experiments/formal_v3_family_svc"
    figure_dir = experiment_root / "figures"
    report_dir = experiment_root / "reports"

    manifest_df = pd.read_csv(sample_manifest_csv, encoding="utf-8-sig")
    filtered = manifest_df[
        (manifest_df["source_kind"] == "raw")
        & (manifest_df["weak_label_available"] == 1)
        & (manifest_df["quality_tier"].isin(["A", "B"]))
        & (manifest_df["family"].isin(["pp_starch", "pe_starch", "pp_pe_starch"]))
    ].copy()
    filtered["split"] = filtered["sample_group_id"].map(assign_group_split)

    result = run_family_specific_svc(filtered, data_root, split_col="split", feature_mode="both")
    save_predictions(result.predictions, report_dir / "test_predictions.csv")
    plot_confusion_matrix(result.predictions, figure_dir / "confusion_matrix.png", title="Family-specific SVC confusion matrix")

    summary = {
        "experiment": "formal_v3_family_svc",
        "model": "family_specific_svc",
        "feature_mode": "both",
        "train_samples": int((filtered["split"] == "train").sum()),
        "test_samples": int((filtered["split"] == "test").sum()),
        "overall_accuracy": result.overall_accuracy,
        "family_accuracy": result.family_accuracy,
        "label_distribution_test": {
            str(k): int(v)
            for k, v in filtered[filtered["split"] == "test"]["concentration_label"].value_counts().sort_index().items()
        },
    }
    save_experiment_summary(summary, report_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
