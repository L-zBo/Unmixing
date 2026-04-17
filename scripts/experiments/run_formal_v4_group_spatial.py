from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demixing.data.manifest import build_sample_manifest
from demixing.data.splits import assign_group_split
from demixing.evaluation.classical_models import load_spectrum_features
from demixing.evaluation.inference import save_predictions
from demixing.visualization.plots import (
    plot_confusion_matrix,
    plot_family_accuracy,
    plot_family_grouped_scores,
    save_experiment_summary,
)


def build_group_model(family: str) -> tuple[Pipeline, str]:
    if family == "pp_starch":
        return (
            Pipeline(
                steps=[
                    ("scale", StandardScaler(with_mean=False)),
                    ("pca", PCA(n_components=12)),
                    ("clf", LinearSVC(C=2.0, max_iter=5000)),
                ]
            ),
            "norm+deriv",
        )
    if family == "pe_starch":
        return (
            Pipeline(
                steps=[
                    ("scale", StandardScaler(with_mean=False)),
                    ("pca", PCA(n_components=10)),
                    ("clf", SVC(C=2.0, kernel="rbf", gamma="scale")),
                ]
            ),
            "both",
        )
    return (
        Pipeline(
            steps=[
                ("scale", StandardScaler(with_mean=False)),
                ("pca", PCA(n_components=16)),
                ("clf", SVC(C=3.0, kernel="rbf", gamma="scale")),
            ]
        ),
        "fingerprint+ch",
    )


def aggregate_group_features(df: pd.DataFrame, data_root: Path, feature_mode: str) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    grouped = df.groupby(["family", "sample_group_id"], sort=True)
    features = []
    labels = []
    keys: list[tuple[str, str]] = []
    for (family, group_id), group in grouped:
        spectra = np.stack([load_spectrum_features(data_root, rel, feature_mode) for rel in group["relative_path"]])
        features.append(spectra.mean(axis=0))
        labels.append(int(group["concentration_label"].iloc[0]))
        keys.append((family, group_id))
    return np.stack(features), np.asarray(labels, dtype=int), keys


def main() -> None:
    data_root = ROOT / "outputs/preprocessing/dataset_preprocessed_v1"
    quality_manifest_csv = data_root / "_reports/quality_manifest.csv"
    sample_manifest_csv = data_root / "_reports/sample_manifest.csv"
    build_sample_manifest(quality_manifest_csv, sample_manifest_csv)

    experiment_root = ROOT / "outputs/experiments/formal_v4_group_spatial"
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

    rows = []
    family_accuracy = {}
    for family in ["pp_starch", "pe_starch", "pp_pe_starch"]:
        family_df = filtered[filtered["family"] == family].copy()
        trainval_df = family_df[family_df["split"].isin(["train", "val"])].copy()
        test_df = family_df[family_df["split"] == "test"].copy()
        model, feature_mode = build_group_model(family)

        X_train, y_train, _ = aggregate_group_features(trainval_df, data_root, feature_mode)
        X_test, y_test, keys = aggregate_group_features(test_df, data_root, feature_mode)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        family_accuracy[family] = float(np.mean(pred == y_test))

        for (row_family, group_id), label, pred_label in zip(keys, y_test, pred):
            n_pixels = int((test_df["sample_group_id"] == group_id).sum())
            rows.append(
                {
                    "family": row_family,
                    "sample_group_id": group_id,
                    "label": int(label),
                    "pred_label": int(pred_label),
                    "n_spectra": n_pixels,
                    "feature_mode": feature_mode,
                }
            )

    group_predictions = pd.DataFrame(rows).sort_values(["family", "sample_group_id"]).reset_index(drop=True)
    save_predictions(group_predictions, report_dir / "group_predictions.csv")
    plot_confusion_matrix(group_predictions, figure_dir / "group_confusion_matrix.png", title="Group-level confusion matrix")
    plot_family_accuracy(family_accuracy, figure_dir / "group_family_accuracy.png", title="Group-level family accuracy")

    family_group_summary = group_predictions.copy()
    family_group_summary["source_kind"] = "group"
    family_group_summary["microplastic_score"] = family_group_summary["pred_label"]
    plot_family_grouped_scores(family_group_summary, figure_dir / "group_family_score_summary.png")

    expanded_true = []
    expanded_pred = []
    for _, row in group_predictions.iterrows():
        expanded_true.extend([int(row["label"])] * int(row["n_spectra"]))
        expanded_pred.extend([int(row["pred_label"])] * int(row["n_spectra"]))

    group_accuracy = float(np.mean(group_predictions["label"] == group_predictions["pred_label"]))
    expanded_accuracy = float(np.mean(np.asarray(expanded_true) == np.asarray(expanded_pred)))

    summary = {
        "experiment": "formal_v4_group_spatial",
        "primary_metric": "group_accuracy",
        "group_accuracy": group_accuracy,
        "group_vote_expanded_accuracy": expanded_accuracy,
        "group_family_accuracy": family_accuracy,
        "n_test_groups": int(len(group_predictions)),
        "n_test_pixels": int(group_predictions["n_spectra"].sum()),
        "note": "formal_v4把样本组/空间图作为主评估单位，直接在组均值特征上训练家族专属分类器。",
    }
    save_experiment_summary(summary, report_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
