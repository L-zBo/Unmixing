from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demixing.data.endmembers import (
    build_default_endmember_library,
    list_available_starch_sources,
)
from demixing.data.preprocess import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_PROTOCOL_NAME,
    PREPROCESS_PROTOCOLS,
    load_spectrum,
    preprocess_record,
)
from demixing.evaluation.classical_unmixing import (
    align_blind_nmf_to_reference,
    blind_nmf_unmix_spectra,
    unmix_spectra,
)


PE_STARCH_DIR = "PE+\u6dc0\u7c89"
PP_STARCH_DIR = "PP+\u6dc0\u7c89"

GENERALIZATION_PRESETS: list[dict[str, object]] = [
    {"sample_dir": Path("泛化/展艺玉米淀粉/PE+淀粉/1 785mw 2s 1 1 40 40"), "matched_starch": "展艺玉米淀粉", "components": ("PE", "starch")},
    {"sample_dir": Path("泛化/展艺玉米淀粉/PP+淀粉/1 785mw 2s 1 1 40 40"), "matched_starch": "展艺玉米淀粉", "components": ("PP", "starch")},
    {"sample_dir": Path("泛化/新良小麦淀粉/PE+淀粉/1 785mw 2s 2 2 40 40"), "matched_starch": "新良小麦淀粉", "components": ("PE", "starch")},
    {"sample_dir": Path("泛化/新良小麦淀粉/PP+淀粉/1 785mw 2s 2 2 40 40"), "matched_starch": "新良小麦淀粉", "components": ("PP", "starch")},
    {"sample_dir": Path("泛化/甘汁园小麦淀粉/PE+淀粉/1 785mw 2s 2 2 40 40"), "matched_starch": "甘汁园小麦淀粉", "components": ("PE", "starch")},
    {"sample_dir": Path("泛化/甘汁园小麦淀粉/PP+淀粉/1 785mw 2s 2 2 40 40"), "matched_starch": "甘汁园小麦淀粉", "components": ("PP", "starch")},
]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/formal_v12_generalization_batch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate OLS/NNLS/FCLS/NMF on the 泛化 dataset under matched vs baseline starch endmember configs."
    )
    parser.add_argument("--input-root", type=Path, default=ROOT / DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--feature-mode", choices=["normalized", "corrected"], default="normalized")
    parser.add_argument("--protocol", choices=sorted(PREPROCESS_PROTOCOLS), default=DEFAULT_PROTOCOL_NAME)
    parser.add_argument(
        "--starch-config",
        choices=["baseline", "matched", "both"],
        default="both",
        help="baseline=玉米基线端元；matched=每个样本用自家淀粉源；both=两者都跑作对比",
    )
    parser.add_argument("--methods", nargs="+", default=["ols", "nnls", "fcls", "nmf"])
    parser.add_argument("--nmf-components", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def parse_xy(name: str) -> tuple[int | None, int | None]:
    match_x = re.search(r"-X(\d+)-", name)
    match_y = re.search(r"-Y(\d+)-", name)
    if match_x is None or match_y is None:
        return None, None
    return int(match_x.group(1)), int(match_y.group(1))


def load_mapping_spectra(
    input_root: Path,
    sample_dir: Path,
    feature_mode: str,
    protocol_name: str,
    limit: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    resolved_dir = sample_dir if sample_dir.is_absolute() else input_root / sample_dir
    csv_files = sorted(resolved_dir.glob("DATA-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No DATA-*.csv files found in {resolved_dir}")

    rows: list[dict[str, object]] = []
    spectra: list[np.ndarray] = []
    for index, csv_path in enumerate(csv_files):
        record = load_spectrum(csv_path, input_root)
        _, corrected, normalized, _ = preprocess_record(record, protocol_name=protocol_name)
        x_idx, y_idx = parse_xy(csv_path.name)
        rows.append(
            {
                "relative_path": csv_path.relative_to(input_root).as_posix(),
                "x_idx": x_idx,
                "y_idx": y_idx,
            }
        )
        spectra.append(normalized if feature_mode == "normalized" else corrected)
        if limit is not None and (index + 1) >= limit:
            break
    return pd.DataFrame(rows), np.stack(spectra).astype(np.float32)


def summarize(
    sample_dir: Path,
    matched_starch: str,
    starch_config: str,
    starch_source_used: str,
    method: str,
    component_names: tuple[str, ...],
    prediction_df: pd.DataFrame,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_dir": sample_dir.as_posix(),
        "matched_starch_source": matched_starch,
        "starch_config": starch_config,
        "starch_source_used": starch_source_used,
        "method": method,
        "n_spectra": int(len(prediction_df)),
        "mean_residual_l2": float(prediction_df["residual_l2"].mean()),
        "mean_residual_rmse": float(prediction_df["residual_rmse"].mean()),
        "mean_residual_r2": float(prediction_df["residual_r2"].mean()),
        "dominant_component_counts": json.dumps(
            prediction_df["dominant_component"].value_counts().to_dict(), ensure_ascii=False
        ),
    }
    for name in component_names:
        abundance_col = f"abundance_{name}"
        if abundance_col in prediction_df.columns:
            row[f"mean_abundance_{name}"] = float(prediction_df[abundance_col].mean())
    return row


def run_single_combination(
    *,
    input_root: Path,
    sample_dir: Path,
    matched_starch: str,
    components: tuple[str, ...],
    starch_config: str,
    starch_source_used: str,
    methods: list[str],
    feature_mode: str,
    protocol_name: str,
    nmf_components_override: int | None,
    limit: int | None,
) -> list[dict[str, object]]:
    metadata_df, spectra = load_mapping_spectra(
        input_root=input_root,
        sample_dir=sample_dir,
        feature_mode=feature_mode,
        protocol_name=protocol_name,
        limit=limit,
    )
    library = build_default_endmember_library(
        input_root=input_root,
        include_components=components,
        starch_source=starch_source_used,
        feature_mode=feature_mode,
        protocol_name=protocol_name,
    )
    rows: list[dict[str, object]] = []
    for method in methods:
        if method == "nmf":
            n_components = nmf_components_override or len(components)
            nmf_result = blind_nmf_unmix_spectra(spectra, n_components=n_components)
            nmf_result, _ = align_blind_nmf_to_reference(nmf_result, library)
            prediction_df = pd.concat(
                [metadata_df.reset_index(drop=True), nmf_result.to_frame()], axis=1
            )
            prediction_df["dominant_component"] = [
                nmf_result.component_names[index]
                for index in np.argmax(nmf_result.abundances, axis=1)
            ]
            rows.append(
                summarize(
                    sample_dir,
                    matched_starch,
                    starch_config,
                    starch_source_used,
                    method,
                    nmf_result.component_names,
                    prediction_df,
                )
            )
        else:
            result = unmix_spectra(spectra, library=library, method=method)
            prediction_df = pd.concat(
                [metadata_df.reset_index(drop=True), result.to_frame()], axis=1
            )
            prediction_df["dominant_component"] = [
                result.component_names[index]
                for index in np.argmax(result.abundances, axis=1)
            ]
            rows.append(
                summarize(
                    sample_dir,
                    matched_starch,
                    starch_config,
                    starch_source_used,
                    method,
                    result.component_names,
                    prediction_df,
                )
            )
    return rows


def main() -> None:
    args = parse_args()
    available_starch = set(list_available_starch_sources())
    starch_configs: list[str]
    if args.starch_config == "both":
        starch_configs = ["baseline", "matched"]
    else:
        starch_configs = [args.starch_config]

    rows: list[dict[str, object]] = []
    for preset in GENERALIZATION_PRESETS:
        sample_dir = preset["sample_dir"]
        matched_starch = preset["matched_starch"]
        components = preset["components"]
        if matched_starch not in available_starch:
            raise KeyError(f"Starch source {matched_starch!r} is not registered in endmembers module.")
        for starch_config in starch_configs:
            starch_source_used = "baseline" if starch_config == "baseline" else matched_starch
            rows.extend(
                run_single_combination(
                    input_root=args.input_root,
                    sample_dir=sample_dir,
                    matched_starch=matched_starch,
                    components=components,
                    starch_config=starch_config,
                    starch_source_used=starch_source_used,
                    methods=args.methods,
                    feature_mode=args.feature_mode,
                    protocol_name=args.protocol,
                    nmf_components_override=args.nmf_components,
                    limit=args.limit,
                )
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(
        args.output_root / "generalization_batch_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_json = {
        "feature_mode": args.feature_mode,
        "protocol": args.protocol,
        "starch_configs": starch_configs,
        "methods": args.methods,
        "presets": [
            {
                "sample_dir": preset["sample_dir"].as_posix(),
                "matched_starch": preset["matched_starch"],
                "components": list(preset["components"]),
            }
            for preset in GENERALIZATION_PRESETS
        ],
        "interpretation_note": (
            "baseline口径用玉米基线端元解混所有泛化样本，反映端元失配的鲁棒性；"
            "matched口径用每个样本对应的自家淀粉端元，反映端元更新后的解混质量。"
            "两组联合解读：matched残差应明显低于baseline，且组分主导比例应更接近真实家族。"
        ),
        "rows": rows,
    }
    (args.output_root / "generalization_batch_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"n_rows": len(rows), "output_root": args.output_root.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
