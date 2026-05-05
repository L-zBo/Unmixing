from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QualityThresholds:
    spike_a: float = 40.0
    spike_b: float = 80.0
    rough_a: float = 0.12
    rough_b: float = 0.20


def assign_quality_tier(spike_score: float, roughness: float, thresholds: QualityThresholds) -> str:
    if spike_score <= thresholds.spike_a and roughness <= thresholds.rough_a:
        return "A"
    if spike_score <= thresholds.spike_b and roughness <= thresholds.rough_b:
        return "B"
    return "C"


def build_quality_manifest(report_csv: Path, output_csv: Path, thresholds: QualityThresholds | None = None) -> dict[str, int]:
    thresholds = thresholds or QualityThresholds()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with report_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    counts = {"A": 0, "B": 0, "C": 0}
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "relative_path",
                "quality_tier",
                "spike_score_after",
                "roughness_after",
                "recommended_weight",
                "converted_from_wavelength",
            ]
        )
        for row in rows:
            spike = float(row["spike_score_after"])
            rough = float(row["roughness_after"])
            tier = assign_quality_tier(spike, rough, thresholds)
            counts[tier] += 1
            recommended_weight = {"A": 1.0, "B": 0.6, "C": 0.2}[tier]
            writer.writerow(
                [
                    row["relative_path"],
                    tier,
                    row["spike_score_after"],
                    row["roughness_after"],
                    recommended_weight,
                    row["converted_from_wavelength"],
                ]
            )
    return counts
