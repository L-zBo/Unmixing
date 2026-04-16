from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RamanSpectrumDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        manifest_csv: Path | None = None,
        use_normalized: bool = False,
        min_quality_tier: str | None = None,
        allowed_source_kinds: set[str] | None = None,
        allowed_families: set[str] | None = None,
        require_weak_label: bool = False,
        allowed_group_ids: set[str] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.use_normalized = use_normalized
        self.samples: list[dict[str, object]] = []
        self.allowed_source_kinds = allowed_source_kinds
        self.allowed_families = allowed_families
        self.require_weak_label = require_weak_label
        self.allowed_group_ids = allowed_group_ids

        if manifest_csv is None:
            for csv_path in sorted(self.data_root.rglob("*.csv")):
                if "_reports" in csv_path.parts:
                    continue
                self.samples.append(
                    {
                        "relative_path": csv_path.relative_to(self.data_root).as_posix(),
                        "quality_tier": "A",
                        "weight": 1.0,
                        "label": -1,
                        "family": "unknown",
                        "source_kind": "unknown",
                        "microplastic_mask": [0.0, 0.0, 0.0],
                        "weak_label_available": 0,
                    }
                )
        else:
            min_rank = {"A": 0, "B": 1, "C": 2}
            threshold = min_rank.get(min_quality_tier or "C", 2)
            with Path(manifest_csv).open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                tier = row["quality_tier"]
                if min_rank[tier] > threshold:
                    continue
                source_kind = row.get("source_kind", "unknown")
                family = row.get("family", "unknown")
                weak_label_available = int(row.get("weak_label_available", 0))
                if self.allowed_source_kinds is not None and source_kind not in self.allowed_source_kinds:
                    continue
                if self.allowed_families is not None and family not in self.allowed_families:
                    continue
                if self.require_weak_label and weak_label_available != 1:
                    continue
                group_id = row.get("sample_group_id", row["relative_path"])
                if self.allowed_group_ids is not None and group_id not in self.allowed_group_ids:
                    continue
                self.samples.append(
                    {
                        "relative_path": row["relative_path"],
                        "quality_tier": tier,
                        "weight": float(row["recommended_weight"]),
                        "label": int(row.get("concentration_label", -1)),
                        "family": family,
                        "source_kind": source_kind,
                        "microplastic_mask": [float(v) for v in row.get("microplastic_mask", "0,0,0").split(",")],
                        "allowed_main_mask": [float(v) for v in row.get("allowed_main_mask", "1,1,1").split(",")],
                        "sample_group_id": group_id,
                        "weak_label_available": weak_label_available,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        sample = self.samples[index]
        path = self.data_root / str(sample["relative_path"])
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        values = np.asarray(
            [
                float(row["Intensity_norm_max"] if self.use_normalized else row["Intensity_corrected"])
                for row in rows
            ],
            dtype=np.float32,
        )
        axis = np.asarray([float(row["RamanShift_cm-1"]) for row in rows], dtype=np.float32)
        return {
            "x": torch.from_numpy(values),
            "axis": torch.from_numpy(axis),
            "quality_tier": str(sample["quality_tier"]),
            "weight": float(sample["weight"]),
            "relative_path": str(sample["relative_path"]),
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
            "family": str(sample["family"]),
            "source_kind": str(sample["source_kind"]),
            "microplastic_mask": torch.tensor(sample["microplastic_mask"], dtype=torch.float32),
            "allowed_main_mask": torch.tensor(sample["allowed_main_mask"], dtype=torch.float32),
            "sample_group_id": str(sample["sample_group_id"]),
            "weak_label_available": torch.tensor(int(sample["weak_label_available"]), dtype=torch.long),
        }
