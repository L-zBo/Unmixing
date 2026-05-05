from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


PP_PE_STARCH = "PP+PE+\u6dc0\u7c89/"
PP_STARCH = "PP+\u7389\u7c73\u6dc0\u7c89/"
PE_STARCH = "PE+\u7389\u7c73\u6dc0\u7c89/"
PP_STARCH_TEST = "PP+\u6dc0\u7c89/"
PE_STARCH_TEST = "PE+\u6dc0\u7c89/"
PURE_PP = "pp\u7eaf\u8c31/"
PURE_PE = "pe\u7eaf\u8c31/"
PURE_STARCH = "\u6dc0\u7c89\u7eaf\u8c31/"

AVG_TOKEN = "/\u5e73\u5747\u5149\u8c31"
PURE_TOKEN = "\u7eaf\u8c31/"

LOW_TOKENS = (
    "\u4f4ePP+\u4f4ePE+\u6dc0\u7c89",
    "\u4f4e\u6d53\u5ea6",
    "\u4f4ePE+\u6dc0\u7c89",
    "\u4f4e PP+\u6dc0\u7c89",
)
MEDIUM_TOKENS = ("\u4e2dPP+\u4e2dPE+\u6dc0\u7c89",)
HIGH_TOKENS = (
    "\u9ad8PP+\u9ad8PE+\u6dc0\u7c89",
    "\u9ad8\u6d53\u5ea6",
    "\u9ad8PE+\u6dc0\u7c89",
    "\u9ad8 PP+\u6dc0\u7c89",
)


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
    allowed_main_mask: tuple[int, int, int]
    sample_group_id: str


def infer_family(relative_path: str) -> str:
    if relative_path.startswith(PP_PE_STARCH):
        return "pp_pe_starch"
    if relative_path.startswith(PP_STARCH) or relative_path.startswith(PP_STARCH_TEST):
        return "pp_starch"
    if relative_path.startswith(PE_STARCH) or relative_path.startswith(PE_STARCH_TEST):
        return "pe_starch"
    if relative_path.startswith(PURE_PP):
        return "pure_pp"
    if relative_path.startswith(PURE_PE):
        return "pure_pe"
    if relative_path.startswith(PURE_STARCH):
        return "pure_starch"
    return "unknown"


def infer_source_kind(relative_path: str) -> str:
    if AVG_TOKEN in relative_path:
        return "average"
    if PURE_TOKEN in relative_path:
        return "pure"
    return "raw"


def _contains_any(relative_path: str, tokens: tuple[str, ...]) -> bool:
    return any(token in relative_path for token in tokens)


def infer_concentration(relative_path: str) -> tuple[str, int, float | None, float | None, bool]:
    if _contains_any(relative_path, MEDIUM_TOKENS):
        return "medium", 1, 0.4, 0.7, True
    if _contains_any(relative_path, LOW_TOKENS):
        return "low", 0, 0.1, 0.4, True
    if _contains_any(relative_path, HIGH_TOKENS):
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


def infer_allowed_main_mask(family: str) -> tuple[int, int, int]:
    if family == "pp_starch":
        return (1, 0, 1)
    if family == "pe_starch":
        return (0, 1, 1)
    if family == "pp_pe_starch":
        return (1, 1, 1)
    if family == "pure_pp":
        return (1, 0, 0)
    if family == "pure_pe":
        return (0, 1, 0)
    if family == "pure_starch":
        return (0, 0, 1)
    return (1, 1, 1)


def infer_sample_group_id(relative_path: str, source_kind: str, family: str, concentration_level: str) -> str:
    parts = relative_path.split("/")
    target_text = ""
    if source_kind == "raw" and len(parts) >= 2:
        target_text = parts[-2]
    elif source_kind == "average":
        target_text = Path(parts[-1]).stem
    if target_text:
        match = re.match(r"(\d+)", target_text)
        if match:
            return f"{family}|{concentration_level}|{int(match.group(1)):03d}"
        match = re.search(r"样本_(\d+)", target_text)
        if match:
            return f"{family}|{concentration_level}|{int(match.group(1)):03d}"
    if source_kind == "pure" and len(parts) >= 2:
        return "/".join(parts[:2])
    return relative_path


def infer_metadata(relative_path: str) -> SampleMetadata:
    family = infer_family(relative_path)
    source_kind = infer_source_kind(relative_path)
    level, label, min_pct, max_pct, available = infer_concentration(relative_path)
    mask = infer_microplastic_mask(family)
    allowed_mask = infer_allowed_main_mask(family)
    if source_kind == "pure":
        level, label, min_pct, max_pct, available = "unlabeled", -1, None, None, False
    sample_group_id = infer_sample_group_id(relative_path, source_kind, family, level)
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
        allowed_main_mask=allowed_mask,
        sample_group_id=sample_group_id,
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
                "allowed_main_mask",
                "sample_group_id",
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
                    ",".join(str(v) for v in meta.allowed_main_mask),
                    meta.sample_group_id,
                    row["converted_from_wavelength"],
                ]
            )
    return counts
