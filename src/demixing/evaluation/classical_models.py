from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class FamilySpecificSVCResult:
    predictions: pd.DataFrame
    family_accuracy: dict[str, float]
    overall_accuracy: float
    group_vote_predictions: pd.DataFrame | None = None
    group_accuracy: float | None = None
    group_vote_expanded_accuracy: float | None = None


def load_spectrum_features(data_root: Path, relative_path: str, feature_mode: str = "both") -> np.ndarray:
    with (data_root / relative_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    corrected = np.asarray([float(row["Intensity_corrected"]) for row in rows], dtype=np.float32)
    normalized = np.asarray([float(row["Intensity_norm_max"]) for row in rows], dtype=np.float32)
    if feature_mode == "corrected":
        return corrected
    if feature_mode == "normalized":
        return normalized
    return np.concatenate([corrected, normalized], dtype=np.float32)


def build_family_specific_svc() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler(with_mean=False)),
            ("pca", PCA(n_components=64)),
            ("svc", SVC(C=3.0, kernel="rbf", gamma="scale")),
        ]
    )


def run_family_specific_svc(
    manifest_df: pd.DataFrame,
    data_root: Path,
    split_col: str = "split",
    feature_mode: str = "both",
) -> FamilySpecificSVCResult:
    train_df = manifest_df[manifest_df[split_col] == "train"].copy()
    test_df = manifest_df[manifest_df[split_col] == "test"].copy()

    rows: list[dict[str, object]] = []
    family_accuracy: dict[str, float] = {}

    for family in sorted(train_df["family"].unique()):
        family_train = train_df[train_df["family"] == family].copy()
        family_test = test_df[test_df["family"] == family].copy()
        if family_train.empty or family_test.empty:
            continue

        X_train = np.stack([load_spectrum_features(data_root, rel, feature_mode) for rel in family_train["relative_path"]])
        y_train = family_train["concentration_label"].to_numpy(dtype=int)
        X_test = np.stack([load_spectrum_features(data_root, rel, feature_mode) for rel in family_test["relative_path"]])
        y_test = family_test["concentration_label"].to_numpy(dtype=int)

        clf = build_family_specific_svc()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        family_accuracy[family] = float(np.mean(pred == y_test))
        for (_, meta_row), pred_label in zip(family_test.iterrows(), pred):
            rows.append(
                {
                    "relative_path": meta_row["relative_path"],
                    "sample_group_id": meta_row["sample_group_id"],
                    "family": meta_row["family"],
                    "source_kind": meta_row["source_kind"],
                    "label": int(meta_row["concentration_label"]),
                    "pred_label": int(pred_label),
                    "quality_tier": meta_row["quality_tier"],
                    "feature_mode": feature_mode,
                }
            )

    predictions = pd.DataFrame(rows).sort_values(["family", "relative_path"]).reset_index(drop=True)
    overall_accuracy = float(np.mean(predictions["label"] == predictions["pred_label"])) if not predictions.empty else 0.0

    group_vote_predictions = None
    group_accuracy = None
    group_vote_expanded_accuracy = None
    if not predictions.empty:
        group_vote_predictions = (
            predictions.groupby(["family", "sample_group_id"], as_index=False)
            .agg(
                label=("label", "first"),
                pred_label=("pred_label", lambda s: int(pd.Series(s).mode().iloc[0])),
                n_spectra=("pred_label", "size"),
            )
            .sort_values(["family", "sample_group_id"])
            .reset_index(drop=True)
        )
        group_accuracy = float(np.mean(group_vote_predictions["label"] == group_vote_predictions["pred_label"]))
        expanded_true: list[int] = []
        expanded_pred: list[int] = []
        for _, row in group_vote_predictions.iterrows():
            expanded_true.extend([int(row["label"])] * int(row["n_spectra"]))
            expanded_pred.extend([int(row["pred_label"])] * int(row["n_spectra"]))
        group_vote_expanded_accuracy = float(np.mean(np.asarray(expanded_true) == np.asarray(expanded_pred)))
    return FamilySpecificSVCResult(
        predictions=predictions,
        family_accuracy=family_accuracy,
        overall_accuracy=overall_accuracy,
        group_vote_predictions=group_vote_predictions,
        group_accuracy=group_accuracy,
        group_vote_expanded_accuracy=group_vote_expanded_accuracy,
    )
