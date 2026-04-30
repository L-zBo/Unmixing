from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

from demixing.data.preprocess import normalized_value_from_row


def _extract_xy(relative_path: str) -> tuple[int, int]:
    name = Path(relative_path).name
    mx = re.search(r"-X(\d+)-", name)
    my = re.search(r"-Y(\d+)", name)
    if mx is None or my is None:
        raise ValueError(f"Cannot parse X/Y from {relative_path}")
    return int(mx.group(1)), int(my.group(1))


def load_pixel_spectrum(data_root: Path, relative_path: str, use_normalized: bool = True) -> np.ndarray:
    with (data_root / relative_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if use_normalized:
        return np.asarray([normalized_value_from_row(row) for row in rows], dtype=np.float32)
    return np.asarray([float(row["Intensity_corrected"]) for row in rows], dtype=np.float32)


@dataclass
class GroupPCAProjector:
    pca: PCA

    @classmethod
    def fit(
        cls,
        manifest_df: pd.DataFrame,
        data_root: Path,
        n_components: int = 8,
        use_normalized: bool = True,
    ) -> "GroupPCAProjector":
        spectra = []
        for rel in manifest_df["relative_path"]:
            spectra.append(load_pixel_spectrum(data_root, rel, use_normalized=use_normalized))
        X = np.stack(spectra)
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X)
        return cls(pca=pca)

    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        return self.pca.transform(spectrum[None, :])[0].astype(np.float32)


class SpatialGroupDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        data_root: Path,
        projector: GroupPCAProjector,
        use_normalized: bool = True,
    ) -> None:
        self.manifest_df = manifest_df.copy()
        self.data_root = Path(data_root)
        self.projector = projector
        self.use_normalized = use_normalized

        self.groups = []
        for group_id, group in self.manifest_df.groupby("sample_group_id", sort=True):
            family = str(group["family"].iloc[0])
            label = int(group["concentration_label"].iloc[0]) if "concentration_label" in group.columns else -1
            rows = []
            xs = []
            ys = []
            for _, row in group.iterrows():
                x, y = _extract_xy(str(row["relative_path"]))
                xs.append(x)
                ys.append(y)
                rows.append((row, x, y))
            self.groups.append(
                {
                    "group_id": group_id,
                    "family": family,
                    "label": label,
                    "rows": rows,
                    "width": max(xs) + 1,
                    "height": max(ys) + 1,
                }
            )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        group = self.groups[index]
        n_channels = self.projector.pca.n_components_ + 1
        grid = np.zeros((n_channels, group["width"], group["height"]), dtype=np.float32)
        for row, x, y in group["rows"]:
            spectrum = load_pixel_spectrum(self.data_root, str(row["relative_path"]), use_normalized=self.use_normalized)
            projected = self.projector.transform(spectrum)
            grid[:-1, x, y] = projected
            grid[-1, x, y] = 1.0  # occupancy mask
        return {
            "image": torch.from_numpy(grid),
            "label": torch.tensor(group["label"], dtype=torch.long),
            "group_id": str(group["group_id"]),
            "family": str(group["family"]),
        }
