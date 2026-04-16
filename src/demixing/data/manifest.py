from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleMetadata:
    relative_path: str
    family: str
    source_kind: str
    concentration_level: str
    concentration_label: int
    concentration_min_pct: float | None
    concentration_max_pct: float | None
    weak_label_available: bool
    microplastic_mask: tuple[int, int, int]


def infer_family(relative_path: str) -> str:
    if relative_path.startswith("PP+PE+淀粉/"):
        return "pp_pe_starch"
    if relative_path.startswith("PP+玉米淀粉/"):
        return "pp_starch"
    if relative_path.startswith("PE+玉米淀粉/"):
        return "pe_starch"
    if relative_path.startswith("pp纯谱/"):
        return "pure_pp"
    if relative_path.startswith("pe纯谱/"):
        return "pure_pe"
    if relative_path.startswith("淀粉纯谱/"):
        return "pure_starch"
    return "unknown"


def infer_source_kind(relative_path: str) -> str:
    if "/平均光谱" in relative_path:
        return "average"
    if "纯谱/" in relative_path:
        return "pure"
    return "raw"


def infer_concentration(relative_path: str) -> tuple[str, int, float | None, float | None, bool]:
    if "中PP+中PE+淀粉" in relative_path:
        return "medium", 1, 0.4, 0.7, True
    if "低PP+低PE+淀粉" in relative_path or "低浓度" in relative_path or "低PE+淀粉" in relative_path or "低 PP+淀粉" in relative_path:
        return "low", 0, 0.1, 0.4, True
    if "高PP+高PE+淀粉" in relative_path or "高浓度" in relative_path or "高PE+淀粉" in relative_path or "高 PP+淀粉" in relative_path:
        return "high", 2, 0.8, 1.0, True
    return "unlabeled", -1, None, None, False


def infer_microplastic_mask(family: str) -> tuple[int, int, int]:
    if family == "pp_starch":
        return (1, 0, 0)
    if family == "pe_starch":
        return (0, 1, 0)
    if family == "pp_pe_starch":
        return (1, 1, 0)
    if family == "pure_pp":
        return (1, 0, 0)
    if family == "pure_pe":
        return (0, 1, 0)
    if family == "pure_starch":
        return (0, 0, 0)
    return (0, 0, 0)


def infer_metadata(relative_path: str) -> SampleMetadata:
    family = infer_family(relative_path)
    source_kind = infer_source_kind(relative_path)
    level, label, min_pct, max_pct, available = infer_concentration(relative_path)
    mask = infer_microplastic_mask(family)
    if source_kind == "pure":
        level, label, min_pct, max_pct, available = "unlabeled", -1, None, None, False
    return SampleMetadata(
        relative_path=relative_path,
        family=family,
        source_kind=source_kind,
        concentration_level=level,
        concentration_label=label,
        concentration_min_pct=min_pct,
        concentration_max_pct=max_pct,
        weak_label_available=available,
        microplastic_mask=mask,
    )


def build_sample_manifest(quality_manifest_csv: Path, output_csv: Path) -> dict[str, int]:
    with quality_manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    counts = {"low": 0, "medium": 0, "high": 0, "unlabeled": 0}
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "relative_path",
                "quality_tier",
                "recommended_weight",
                "family",
                "source_kind",
                "concentration_level",
                "concentration_label",
                "concentration_min_pct",
                "concentration_max_pct",
                "weak_label_available",
                "microplastic_mask",
                "converted_from_wavelength",
            ]
        )
        for row in rows:
            meta = infer_metadata(row["relative_path"])
            counts[meta.concentration_level] += 1
            writer.writerow(
                [
                    meta.relative_path,
                    row["quality_tier"],
                    row["recommended_weight"],
                    meta.family,
                    meta.source_kind,
                    meta.concentration_level,
                    meta.concentration_label,
                    "" if meta.concentration_min_pct is None else meta.concentration_min_pct,
                    "" if meta.concentration_max_pct is None else meta.concentration_max_pct,
                    int(meta.weak_label_available),
                    ",".join(str(v) for v in meta.microplastic_mask),
                    row["converted_from_wavelength"],
                ]
            )
    return counts
