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
PP_PE_STARCH_DIR = "PP+PE+\u6dc0\u7c89"

DEFAULT_SAMPLE_DIRS = [
    Path(PE_STARCH_DIR) / "1 785mw 2s 2 2 40 40",
    Path(PP_STARCH_DIR) / "1 785mw 2s 2 2 40 40",
    Path("test") / PP_PE_STARCH_DIR / "1 785mw 2s 1 1 40 40",
]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/formal_v8_batch_method_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-compare OLS/NNLS/FCLS/NMF on multiple real Raman mapping directories.")
    parser.add_argument("--input-root", type=Path, default=ROOT / DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--feature-mode", choices=["normalized", "corrected"], default="normalized")
    parser.add_argument("--starch-source", default="baseline")
    parser.add_argument("--protocol", choices=sorted(PREPROCESS_PROTOCOLS), default=DEFAULT_PROTOCOL_NAME)
    parser.add_argument("--nmf-components", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-dir", type=Path, action="append", default=None)
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


def summarize_prediction_df(
    sample_dir: Path,
    method: str,
    component_names: tuple[str, ...],
    prediction_df: pd.DataFrame,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_dir": sample_dir.as_posix(),
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
        coef_col = f"coef_{name}"
        if coef_col in prediction_df.columns:
            row[f"mean_coef_{name}"] = float(prediction_df[coef_col].mean())
    return row


def main() -> None:
    args = parse_args()
    sample_dirs = args.sample_dir or DEFAULT_SAMPLE_DIRS
    rows: list[dict[str, object]] = []

    for sample_dir in sample_dirs:
        components = infer_components(sample_dir)
        _, spectra = load_mapping_spectra(
            input_root=args.input_root,
            sample_dir=sample_dir,
            feature_mode=args.feature_mode,
            protocol_name=args.protocol,
            limit=args.limit,
        )
        reference_library = build_default_endmember_library(
            input_root=args.input_root,
            include_components=components,
            starch_source=args.starch_source,
            feature_mode=args.feature_mode,
            protocol_name=args.protocol,
        )

        for method in ("ols", "nnls", "fcls"):
            result = unmix_spectra(spectra, library=reference_library, method=method)
            prediction_df = result.to_frame()
            prediction_df["dominant_component"] = [
                result.component_names[index]
                for index in np.argmax(result.abundances, axis=1)
            ]
            rows.append(summarize_prediction_df(sample_dir, method, result.component_names, prediction_df))

        nmf_components = args.nmf_components or len(components)
        nmf_result = blind_nmf_unmix_spectra(spectra, n_components=nmf_components)
        nmf_result, _ = align_blind_nmf_to_reference(nmf_result, reference_library)
        nmf_prediction_df = nmf_result.to_frame()
        nmf_prediction_df["dominant_component"] = [
            nmf_result.component_names[index]
            for index in np.argmax(nmf_result.abundances, axis=1)
        ]
        rows.append(summarize_prediction_df(sample_dir, "nmf", nmf_result.component_names, nmf_prediction_df))

    summary_df = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_root / "batch_method_comparison_summary.csv", index=False, encoding="utf-8-sig")
    summary_json = {
        "feature_mode": args.feature_mode,
        "protocol": args.protocol,
        "starch_source": args.starch_source,
        "samples": [sample_dir.as_posix() for sample_dir in sample_dirs],
        "rows": rows,
    }
    (args.output_root / "batch_method_comparison_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_json)


if __name__ == "__main__":
    main()
