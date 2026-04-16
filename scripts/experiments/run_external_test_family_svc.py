from __future__ import annotations

import sys
from pathlib import Path

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
from demixing.evaluation.classical_models import load_spectrum_features
from demixing.evaluation.inference import save_predictions
from demixing.visualization.plots import plot_prediction_map, save_experiment_summary


def build_family_model(family: str) -> Pipeline:
    if family == "pp_starch":
        return Pipeline(
            steps=[
                ("scale", StandardScaler(with_mean=False)),
                ("pca", PCA(n_components=96)),
                ("clf", LinearSVC(C=2.0, max_iter=5000)),
            ]
        )
    if family == "pe_starch":
        return Pipeline(
            steps=[
                ("scale", StandardScaler(with_mean=False)),
                ("pca", PCA(n_components=48)),
                ("clf", SVC(C=2.0, kernel="rbf", gamma="scale")),
            ]
        )
    return Pipeline(
        steps=[
            ("scale", StandardScaler(with_mean=False)),
            ("pca", PCA(n_components=64)),
            ("clf", SVC(C=3.0, kernel="rbf", gamma="scale")),
        ]
    )


def feature_mode_for_family(family: str) -> str:
    if family == "pp_starch":
        return "norm+deriv"
    if family == "pe_starch":
        return "both"
    return "both"


def main() -> None:
    train_root = ROOT / "outputs/preprocessing/dataset_preprocessed_v1"
    test_root = ROOT / "outputs/preprocessing/test_preprocessed_v1"
    experiment_root = ROOT / "outputs/experiments/external_test_family_svc_v1"
    figure_dir = experiment_root / "figures"
    report_dir = experiment_root / "reports"

    train_manifest = train_root / "_reports/sample_manifest.csv"
    test_quality_manifest = test_root / "_reports/quality_manifest.csv"
    test_manifest = test_root / "_reports/sample_manifest.csv"
    build_sample_manifest(test_quality_manifest, test_manifest)

    train_df = pd.read_csv(train_manifest, encoding="utf-8-sig")
    train_df = train_df[
        (train_df["source_kind"] == "raw")
        & (train_df["weak_label_available"] == 1)
        & (train_df["quality_tier"].isin(["A", "B"]))
        & (train_df["family"].isin(["pp_starch", "pe_starch", "pp_pe_starch"]))
    ].copy()

    test_df = pd.read_csv(test_manifest, encoding="utf-8-sig")
    test_df = test_df[
        (test_df["source_kind"] == "raw")
        & (test_df["family"].isin(["pp_starch", "pe_starch", "pp_pe_starch"]))
    ].copy()

    rows: list[dict[str, object]] = []
    family_counts: dict[str, dict[str, int]] = {}
    for family in sorted(test_df["family"].unique()):
        train_family = train_df[train_df["family"] == family].copy()
        test_family = test_df[test_df["family"] == family].copy()
        mode = feature_mode_for_family(family)
        model = build_family_model(family)

        X_train = pd.Series(train_family["relative_path"]).map(lambda rel: load_spectrum_features(train_root, rel, mode)).to_list()
        X_test = pd.Series(test_family["relative_path"]).map(lambda rel: load_spectrum_features(test_root, rel, mode)).to_list()
        model.fit(X_train, train_family["concentration_label"].to_numpy(dtype=int))
        pred = model.predict(X_test)

        label_names = {0: "low", 1: "medium", 2: "high"}
        family_counts[family] = {
            label_names[int(k)]: int(v)
            for k, v in pd.Series(pred).value_counts().sort_index().items()
        }
        for (_, meta_row), pred_label in zip(test_family.iterrows(), pred):
            rows.append(
                {
                    "relative_path": meta_row["relative_path"],
                    "sample_group_id": meta_row["sample_group_id"],
                    "family": meta_row["family"],
                    "source_kind": meta_row["source_kind"],
                    "pred_label": int(pred_label),
                    "quality_tier": meta_row["quality_tier"],
                    "feature_mode": mode,
                }
            )

    predictions = pd.DataFrame(rows).sort_values(["family", "relative_path"]).reset_index(drop=True)
    save_predictions(predictions, report_dir / "external_test_predictions.csv")

    for family, subset in predictions.groupby("family"):
        plot_prediction_map(subset, figure_dir / f"{family}_prediction_map.png", title=f"{family} external test prediction map")

    summary = {
        "experiment": "external_test_family_svc_v1",
        "families": family_counts,
        "num_predictions": int(len(predictions)),
    }
    save_experiment_summary(summary, report_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
