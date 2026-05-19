"""Sweep PRISM hyperparameters (lambda_l2 / lambda_tv / tv_iters / weight_mode) on synthetic ground-truth data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.endmembers import EndmemberLibrary
from preprocessing.preprocess import DEFAULT_PROTOCOL_NAME, PREPROCESS_PROTOCOLS, SpectrumRecord, preprocess_record
from unmixing.unmix import prism_unmix_spectra, unmix_spectra


DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/prism_param_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-roots",
        nargs="+",
        type=Path,
        default=[
            ROOT / "outputs/synthetic_unmixing/formal_v1_als_l2",
            ROOT / "outputs/synthetic_unmixing/formal_v1_clean_als_l2",
        ],
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--protocol", choices=sorted(PREPROCESS_PROTOCOLS), default=DEFAULT_PROTOCOL_NAME)
    return parser.parse_args()


def load_synthetic_bundle(synthetic_root: Path):
    metadata = json.loads((synthetic_root / "metadata.json").read_text(encoding="utf-8"))
    component_names = tuple(metadata["component_names"])
    axis = np.load(synthetic_root / "axis.npy")
    endmember_matrix = np.load(synthetic_root / "endmember_matrix.npy")
    abundances = np.load(synthetic_root / "abundances.npy")
    spectra = np.load(synthetic_root / "spectra.npy")
    library = EndmemberLibrary(
        names=component_names,
        axis=axis.astype(np.float32),
        matrix=endmember_matrix.astype(np.float32),
        feature_mode="normalized",
        source_paths={name: Path(name) for name in component_names},
    )
    return library, abundances.astype(np.float32), spectra.astype(np.float32), int(metadata["height"]), int(metadata["width"])


def preprocess_synthetic_spectra(axis: np.ndarray, spectra: np.ndarray, protocol_name: str) -> np.ndarray:
    processed = []
    for index, spectrum in enumerate(spectra):
        record = SpectrumRecord(
            relative_path=Path(f"synthetic_{index:05d}.csv"),
            axis=axis,
            intensity=spectrum,
            axis_type="raman_shift_cm-1",
            source_format="synthetic",
            header_axis="RamanShift_cm-1",
            header_intensity="Intensity",
        )
        _, _, normalized, _ = preprocess_record(record, protocol_name=protocol_name)
        processed.append(normalized)
    return np.stack(processed).astype(np.float32)


def row_normalize_nonneg(prediction: np.ndarray) -> np.ndarray:
    clipped = np.clip(prediction, 0.0, None)
    row_sum = clipped.sum(axis=1, keepdims=True)
    safe = np.where(row_sum > 1e-12, row_sum, 1.0)
    return (clipped / safe).astype(np.float32)


def per_component_pearson(truth_flat: np.ndarray, pred_flat: np.ndarray) -> float:
    rs = []
    for col in range(truth_flat.shape[1]):
        t, p = truth_flat[:, col], pred_flat[:, col]
        if t.std() < 1e-12 or p.std() < 1e-12:
            rs.append(0.0)
            continue
        rs.append(float(np.mean((t - t.mean()) * (p - p.mean())) / (t.std() * p.std())))
    return float(np.mean(rs))


def evaluate_run(truth_flat, pred_abund, recon, spectra_norm, height, width, elapsed) -> dict:
    pred = row_normalize_nonneg(pred_abund)
    err = pred - truth_flat
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    pearson_r = per_component_pearson(truth_flat, pred)
    residual = spectra_norm - recon
    recon_rmse = float(np.sqrt(np.mean(residual * residual)))
    pred_map = pred.reshape(height, width, -1)
    dx = np.abs(np.diff(pred_map, axis=1)).mean()
    dy = np.abs(np.diff(pred_map, axis=0)).mean()
    spatial_tv = float((dx + dy) / 2)
    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "recon_rmse": recon_rmse,
        "spatial_tv": spatial_tv,
        "elapsed_s": float(elapsed),
    }


def sweep_on_dataset(library, truth_flat, spectra_norm, height, width,
                     param_grid: list[dict], dataset_name: str) -> list[dict]:
    rows: list[dict] = []
    t0 = time.perf_counter()
    nnls_result = unmix_spectra(spectra_norm, library=library, method="nnls")
    nnls_metrics = evaluate_run(truth_flat, nnls_result.abundances, nnls_result.reconstructed,
                                spectra_norm, height, width, time.perf_counter() - t0)
    rows.append({"dataset": dataset_name, "method": "nnls",
                 "lambda_l2": np.nan, "lambda_tv": np.nan, "tv_iters": 0, "weight_mode": "n/a",
                 **nnls_metrics})

    for params in param_grid:
        t0 = time.perf_counter()
        prism_result = prism_unmix_spectra(
            spectra_norm,
            library=library,
            image_shape=(height, width) if params["lambda_tv"] > 0 else None,
            lambda_l2=params["lambda_l2"],
            lambda_tv=params["lambda_tv"],
            tv_iters=params["tv_iters"],
            weight_mode=params["weight_mode"],
        )
        metrics = evaluate_run(truth_flat, prism_result.abundances, prism_result.reconstructed,
                               spectra_norm, height, width, time.perf_counter() - t0)
        rows.append({"dataset": dataset_name, "method": "prism",
                     **params, **metrics})
    return rows


def build_param_grid() -> list[dict]:
    """Phase 1 (no TV) sweeps L2 + weight_mode; Phase 2 sweeps TV at fixed L2/weight."""
    grid: list[dict] = []
    lambda_l2_values = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
    weight_modes = ["uniform", "endmember_std"]
    for lambda_l2, weight_mode in product(lambda_l2_values, weight_modes):
        grid.append({
            "lambda_l2": lambda_l2,
            "lambda_tv": 0.0,
            "tv_iters": 0,
            "weight_mode": weight_mode,
        })

    lambda_tv_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    tv_iters_values = [1, 2, 3, 5]
    fixed_l2 = 1e-4
    fixed_weight = "endmember_std"
    for lambda_tv, tv_iters in product(lambda_tv_values, tv_iters_values):
        grid.append({
            "lambda_l2": fixed_l2,
            "lambda_tv": lambda_tv,
            "tv_iters": tv_iters,
            "weight_mode": fixed_weight,
        })
    return grid


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    param_grid = build_param_grid()
    all_rows: list[dict] = []
    for synthetic_root in args.synthetic_roots:
        library, truth_abundances, spectra_raw, height, width = load_synthetic_bundle(synthetic_root)
        spectra_norm = preprocess_synthetic_spectra(library.axis, spectra_raw, protocol_name=args.protocol)
        truth_flat = truth_abundances.reshape(-1, library.n_endmembers)
        print(f"\n=== Sweeping {synthetic_root.name} ({height}x{width}, {len(param_grid)} configs) ===")
        rows = sweep_on_dataset(library, truth_flat, spectra_norm, height, width,
                                param_grid, dataset_name=synthetic_root.name)
        all_rows.extend(rows)
        # Per-dataset best
        df = pd.DataFrame(rows)
        prism_df = df[df["method"] == "prism"].copy()
        best_mae = prism_df.loc[prism_df["mae"].idxmin()]
        best_pearson = prism_df.loc[prism_df["pearson_r"].idxmax()]
        nnls_row = df[df["method"] == "nnls"].iloc[0]
        print(f"NNLS baseline:    MAE={nnls_row['mae']:.4f}  RMSE={nnls_row['rmse']:.4f}  pearson_r={nnls_row['pearson_r']:.4f}")
        print(
            f"Best MAE config:  MAE={best_mae['mae']:.4f}  RMSE={best_mae['rmse']:.4f}  pearson_r={best_mae['pearson_r']:.4f}  "
            f"lambda_l2={best_mae['lambda_l2']:.0e}  lambda_tv={best_mae['lambda_tv']:.3f}  "
            f"tv_iters={int(best_mae['tv_iters'])}  weight_mode={best_mae['weight_mode']}"
        )
        print(
            f"Best Pearson cfg: MAE={best_pearson['mae']:.4f}  RMSE={best_pearson['rmse']:.4f}  pearson_r={best_pearson['pearson_r']:.4f}  "
            f"lambda_l2={best_pearson['lambda_l2']:.0e}  lambda_tv={best_pearson['lambda_tv']:.3f}  "
            f"tv_iters={int(best_pearson['tv_iters'])}  weight_mode={best_pearson['weight_mode']}"
        )

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(args.output_root / "prism_param_sweep_full.csv", index=False, encoding="utf-8-sig")
    print(f"\n[sweep] full grid -> {args.output_root / 'prism_param_sweep_full.csv'}")


if __name__ == "__main__":
    main()
