"""Absent-endmember stress test: run NNLS/PRISM on real samples with the full 3-endmember library."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.endmembers import build_default_endmember_library
from preprocessing.preprocess import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_PROTOCOL_NAME,
    PREPROCESS_PROTOCOLS,
    load_spectrum,
    preprocess_record,
)
from unmixing.unmix import prism_unmix_spectra, unmix_spectra


DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/prism_absent_check"

SAMPLES = [
    {"name": "PE_starch_train", "sample_dir": Path("PE+淀粉") / "1 785mw 2s 2 2 40 40", "absent": "PP"},
    {"name": "PP_starch_train", "sample_dir": Path("PP+淀粉") / "1 785mw 2s 2 2 40 40", "absent": "PE"},
]

PRISM_CONFIGS = {
    "PRISM_OLD": {"lambda_l2": 1e-4, "lambda_tv": 0.02, "tv_iters": 2},
    "PRISM_MID": {"lambda_l2": 1e-2, "lambda_tv": 0.10, "tv_iters": 2},
    "PRISM_AGG": {"lambda_l2": 1e-2, "lambda_tv": 0.20, "tv_iters": 1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--protocol", choices=sorted(PREPROCESS_PROTOCOLS), default=DEFAULT_PROTOCOL_NAME)
    parser.add_argument("--feature-mode", choices=["normalized", "corrected"], default="normalized")
    return parser.parse_args()


def parse_xy(name: str) -> tuple[int | None, int | None]:
    mx = re.search(r"-X(\d+)-", name)
    my = re.search(r"-Y(\d+)-", name)
    return (int(mx.group(1)) if mx else None, int(my.group(1)) if my else None)


def load_mapping(input_root: Path, sample_dir: Path, feature_mode: str, protocol_name: str):
    resolved = sample_dir if sample_dir.is_absolute() else input_root / sample_dir
    csv_files = sorted(resolved.glob("DATA-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No DATA-*.csv files found in {resolved}")
    spectra, xs, ys = [], [], []
    for path in csv_files:
        record = load_spectrum(path, input_root)
        _, corrected, normalized, _ = preprocess_record(record, protocol_name=protocol_name)
        spectra.append(normalized if feature_mode == "normalized" else corrected)
        x, y = parse_xy(path.name)
        xs.append(x)
        ys.append(y)
    spectra_arr = np.stack(spectra).astype(np.float32)
    xs_arr, ys_arr = np.array(xs), np.array(ys)
    width = int(xs_arr.max() - xs_arr.min() + 1)
    height = int(ys_arr.max() - ys_arr.min() + 1)
    order = np.lexsort((xs_arr, ys_arr))
    return spectra_arr[order], height, width


def row_normalize_nonneg(prediction: np.ndarray) -> np.ndarray:
    clipped = np.clip(prediction, 0.0, None)
    row_sum = clipped.sum(axis=1, keepdims=True)
    safe = np.where(row_sum > 1e-12, row_sum, 1.0)
    return (clipped / safe).astype(np.float32)


def evaluate(method_name, abundances, reconstructed, spectra, component_names, absent_name, elapsed):
    abundance = row_normalize_nonneg(abundances)
    residual = spectra - reconstructed
    recon_rmse = float(np.sqrt(np.mean(residual * residual)))
    absent_idx = component_names.index(absent_name)
    absent_per_pixel = abundance[:, absent_idx]
    row = {
        "method": method_name,
        "recon_rmse": recon_rmse,
        f"mean_abundance_{absent_name}_absent": float(absent_per_pixel.mean()),
        f"p95_abundance_{absent_name}_absent": float(np.percentile(absent_per_pixel, 95)),
        f"max_abundance_{absent_name}_absent": float(absent_per_pixel.max()),
        f"fraction_pixels_{absent_name}_over_0.10": float((absent_per_pixel > 0.10).mean()),
        "elapsed_s": float(elapsed),
    }
    for i, name in enumerate(component_names):
        row[f"mean_abundance_{name}"] = float(abundance[:, i].mean())
    return row


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    library = build_default_endmember_library(
        input_root=args.input_root,
        include_components=("PE", "PP", "starch"),
        starch_source="baseline",
        feature_mode=args.feature_mode,
        protocol_name=args.protocol,
    )
    component_names = library.names
    print(f"library components: {component_names}")

    all_rows: list[dict] = []
    for sample in SAMPLES:
        sample_name = sample["name"]
        absent = sample["absent"]
        print(f"\n=== {sample_name}  absent={absent}  (forced full 3-EM library) ===")
        try:
            spectra, height, width = load_mapping(
                input_root=args.input_root, sample_dir=sample["sample_dir"],
                feature_mode=args.feature_mode, protocol_name=args.protocol,
            )
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue

        t0 = time.perf_counter()
        nnls = unmix_spectra(spectra, library=library, method="nnls")
        row = evaluate("NNLS", nnls.abundances, nnls.reconstructed, spectra, component_names, absent,
                       time.perf_counter() - t0)
        row["sample"] = sample_name
        all_rows.append(row)

        for cfg_name, params in PRISM_CONFIGS.items():
            t0 = time.perf_counter()
            prism = prism_unmix_spectra(
                spectra, library=library, image_shape=(height, width),
                lambda_l2=params["lambda_l2"], lambda_tv=params["lambda_tv"],
                tv_iters=params["tv_iters"], weight_mode="endmember_std",
            )
            row = evaluate(cfg_name, prism.abundances, prism.reconstructed, spectra,
                           component_names, absent, time.perf_counter() - t0)
            row["sample"] = sample_name
            all_rows.append(row)

        sample_df = pd.DataFrame([r for r in all_rows if r["sample"] == sample_name])
        cols = ["method", "recon_rmse",
                f"mean_abundance_{absent}_absent",
                f"p95_abundance_{absent}_absent",
                f"max_abundance_{absent}_absent",
                f"fraction_pixels_{absent}_over_0.10",
                "mean_abundance_PE", "mean_abundance_PP", "mean_abundance_starch",
                "elapsed_s"]
        print(sample_df[cols].to_string(index=False, float_format="%.4f"))

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(args.output_root / "prism_absent_check_summary.csv", index=False, encoding="utf-8-sig")
    (args.output_root / "prism_absent_check_summary.json").write_text(
        json.dumps(
            {"library_components": list(component_names),
             "configs": {"NNLS": {}, **PRISM_CONFIGS},
             "rows": all_rows}, ensure_ascii=False, indent=2
        ), encoding="utf-8"
    )
    print(f"\n[absent-check] full summary -> {args.output_root}")


if __name__ == "__main__":
    main()
