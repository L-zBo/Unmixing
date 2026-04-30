from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demixing.data.endmembers import build_default_endmember_library
from demixing.data.preprocess import DEFAULT_INPUT_ROOT, PREPROCESS_PROTOCOLS, load_spectrum, preprocess_record
from demixing.evaluation.classical_unmixing import (
    align_blind_nmf_to_reference,
    blind_nmf_unmix_spectra,
    unmix_spectra,
)
from demixing.visualization.classical_unmixing import plot_method_abundance_bars, plot_method_metric_bars


PE_STARCH_DIR = "PE+\u6dc0\u7c89"
PP_STARCH_DIR = "PP+\u6dc0\u7c89"
PP_PE_STARCH_DIR = "PP+PE+\u6dc0\u7c89"

DEFAULT_SAMPLE_DIR = Path(PE_STARCH_DIR) / "1 785mw 2s 2 2 40 40"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/formal_v10_preprocessing_comparison_real"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare preprocessing protocols for one real Raman map under a fixed unmixing method.")
    parser.add_argument("--input-root", type=Path, default=ROOT / DEFAULT_INPUT_ROOT)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--method", choices=["ols", "nnls", "fcls", "nmf"], default="nnls")
    parser.add_argument("--feature-mode", choices=["normalized", "corrected"], default="normalized")
    parser.add_argument("--protocol", action="append", choices=sorted(PREPROCESS_PROTOCOLS), default=None)
    parser.add_argument("--nmf-components", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def infer_components(sample_dir: Path) -> tuple[str, ...]:
    path_text = sample_dir.as_posix()
    if PP_PE_STARCH_DIR in path_text:
        return ("PE", "PP", "starch")
    if PE_STARCH_DIR in path_text:
        return ("PE", "starch")
    if PP_STARCH_DIR in path_text:
        return ("PP", "starch")
    raise ValueError(f"Unable to infer components from sample_dir={sample_dir.as_posix()!r}")


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
        rows.append({"relative_path": csv_path.relative_to(input_root).as_posix(), "x_idx": x_idx, "y_idx": y_idx})
        spectra.append(normalized if feature_mode == "normalized" else corrected)
        if limit is not None and (index + 1) >= limit:
            break
    return pd.DataFrame(rows), np.stack(spectra).astype(np.float32)


def summarize_protocol(protocol_name: str, method: str, component_names: tuple[str, ...], prediction_df: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {
        "protocol": protocol_name,
        "method": method,
        "n_spectra": int(len(prediction_df)),
        "mean_residual_l2": float(prediction_df["residual_l2"].mean()),
        "mean_residual_rmse": float(prediction_df["residual_rmse"].mean()),
        "mean_residual_r2": float(prediction_df["residual_r2"].mean()),
        "dominant_component_counts": json.dumps(prediction_df["dominant_component"].value_counts().to_dict(), ensure_ascii=False),
    }
    for name in component_names:
        abundance_col = f"abundance_{name}"
        if abundance_col in prediction_df.columns:
            row[f"mean_abundance_{name}"] = float(prediction_df[abundance_col].mean())
    return row


def main() -> None:
    args = parse_args()
    protocols = args.protocol or ["als_l2", "als_max", "none_l2"]
    components = infer_components(args.sample_dir)
    output_dir = args.output_root / args.sample_dir.as_posix().replace("/", "__").replace("\\", "__")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for protocol_name in protocols:
        metadata_df, spectra = load_mapping_spectra(
            input_root=args.input_root,
            sample_dir=args.sample_dir,
            feature_mode=args.feature_mode,
            protocol_name=protocol_name,
            limit=args.limit,
        )
        if args.method == "nmf":
            reference_library = build_default_endmember_library(
                input_root=args.input_root,
                include_components=components,
                feature_mode=args.feature_mode,
                protocol_name=protocol_name,
            )
            nmf_components = args.nmf_components or len(components)
            result = blind_nmf_unmix_spectra(spectra, n_components=nmf_components)
            result, _ = align_blind_nmf_to_reference(result, reference_library)
            prediction_df = pd.concat([metadata_df.reset_index(drop=True), result.to_frame()], axis=1)
            prediction_df["dominant_component"] = [
                result.component_names[index]
                for index in np.argmax(result.abundances, axis=1)
            ]
            component_names = result.component_names
        else:
            library = build_default_endmember_library(
                input_root=args.input_root,
                include_components=components,
                feature_mode=args.feature_mode,
                protocol_name=protocol_name,
            )
            result = unmix_spectra(spectra, library=library, method=args.method)
            prediction_df = pd.concat([metadata_df.reset_index(drop=True), result.to_frame()], axis=1)
            prediction_df["dominant_component"] = [
                result.component_names[index]
                for index in np.argmax(result.abundances, axis=1)
            ]
            component_names = result.component_names

        prediction_df.to_csv(output_dir / f"{args.method}_{protocol_name}_pixels.csv", index=False, encoding="utf-8-sig")
        rows.append(summarize_protocol(protocol_name, args.method, component_names, prediction_df))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / f"{args.method}_preprocessing_comparison_summary.csv", index=False, encoding="utf-8-sig")
    plot_df = summary_df.copy()
    plot_df["method"] = plot_df["protocol"]
    plot_method_metric_bars(
        plot_df,
        metric_cols=["mean_residual_rmse", "mean_residual_r2"],
        output_path=output_dir / f"{args.method}_preprocessing_metric_comparison.png",
        title=f"{args.method.upper()} preprocessing residual comparison",
    )
    plot_method_abundance_bars(
        plot_df,
        component_names=list(components),
        output_path=output_dir / f"{args.method}_preprocessing_abundance_comparison.png",
        title=f"{args.method.upper()} preprocessing abundance comparison",
    )
    summary_json = {
        "sample_dir": args.sample_dir.as_posix(),
        "method": args.method,
        "feature_mode": args.feature_mode,
        "protocols": protocols,
        "components": list(components),
        "rows": rows,
    }
    (output_dir / f"{args.method}_preprocessing_comparison_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_json)


if __name__ == "__main__":
    main()
