"""Synthetic-data STD vs UNI weighting comparison to close the 4-quadrant matrix."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.endmembers import EndmemberLibrary
from preprocessing.preprocess import DEFAULT_PROTOCOL_NAME, PREPROCESS_PROTOCOLS, SpectrumRecord, preprocess_record
from unmixing.unmix import prism_unmix_spectra, unmix_spectra


DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/prism_synth_std_vs_uni"

PRISM_CONFIGS = {
    "PRISM_OLD_STD": {"lambda_l2": 1e-4, "lambda_tv": 0.02, "tv_iters": 2, "weight_mode": "endmember_std"},
    "PRISM_OLD_UNI": {"lambda_l2": 1e-4, "lambda_tv": 0.02, "tv_iters": 2, "weight_mode": "uniform"},
    "PRISM_MID_STD": {"lambda_l2": 1e-2, "lambda_tv": 0.10, "tv_iters": 2, "weight_mode": "endmember_std"},
    "PRISM_MID_UNI": {"lambda_l2": 1e-2, "lambda_tv": 0.10, "tv_iters": 2, "weight_mode": "uniform"},
    "PRISM_AGG_STD": {"lambda_l2": 1e-2, "lambda_tv": 0.20, "tv_iters": 1, "weight_mode": "endmember_std"},
    "PRISM_AGG_UNI": {"lambda_l2": 1e-2, "lambda_tv": 0.20, "tv_iters": 1, "weight_mode": "uniform"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-roots", nargs="+", type=Path, default=[
        ROOT / "outputs/synthetic_unmixing/formal_v1_als_l2",
        ROOT / "outputs/synthetic_unmixing/formal_v1_clean_als_l2",
    ])
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
            axis=axis, intensity=spectrum,
            axis_type="raman_shift_cm-1", source_format="synthetic",
            header_axis="RamanShift_cm-1", header_intensity="Intensity",
        )
        _, _, normalized, _ = preprocess_record(record, protocol_name=protocol_name)
        processed.append(normalized)
    return np.stack(processed).astype(np.float32)


def row_normalize_nonneg(pred: np.ndarray) -> np.ndarray:
    clipped = np.clip(pred, 0.0, None)
    s = clipped.sum(axis=1, keepdims=True)
    safe = np.where(s > 1e-12, s, 1.0)
    return (clipped / safe).astype(np.float32)


def per_component_pearson(t: np.ndarray, p: np.ndarray) -> float:
    rs = []
    for col in range(t.shape[1]):
        tc, pc = t[:, col], p[:, col]
        if tc.std() < 1e-12 or pc.std() < 1e-12:
            rs.append(0.0)
            continue
        rs.append(float(np.mean((tc - tc.mean()) * (pc - pc.mean())) / (tc.std() * pc.std())))
    return float(np.mean(rs))


def evaluate(method, truth_flat, abundances, reconstructed, spectra, height, width, elapsed) -> dict:
    pred = row_normalize_nonneg(abundances)
    err = pred - truth_flat
    residual = spectra - reconstructed
    pmap = pred.reshape(height, width, -1)
    dx = np.abs(np.diff(pmap, axis=1)).mean()
    dy = np.abs(np.diff(pmap, axis=0)).mean()
    return {
        "method": method,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "pearson_r": per_component_pearson(truth_flat, pred),
        "recon_rmse": float(np.sqrt(np.mean(residual * residual))),
        "spatial_tv": float((dx + dy) / 2),
        "elapsed_s": float(elapsed),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for synthetic_root in args.synthetic_roots:
        library, truth_abundances, spectra_raw, height, width = load_synthetic_bundle(synthetic_root)
        spectra_norm = preprocess_synthetic_spectra(library.axis, spectra_raw, protocol_name=args.protocol)
        truth_flat = truth_abundances.reshape(-1, library.n_endmembers)
        print(f"\n=== {synthetic_root.name} ({height}x{width}) ===")

        t0 = time.perf_counter()
        nnls = unmix_spectra(spectra_norm, library=library, method="nnls")
        row = evaluate("NNLS", truth_flat, nnls.abundances, nnls.reconstructed, spectra_norm, height, width,
                       time.perf_counter() - t0)
        row["dataset"] = synthetic_root.name
        all_rows.append(row)

        for cfg_name, params in PRISM_CONFIGS.items():
            t0 = time.perf_counter()
            prism = prism_unmix_spectra(
                spectra_norm, library=library, image_shape=(height, width),
                lambda_l2=params["lambda_l2"], lambda_tv=params["lambda_tv"],
                tv_iters=params["tv_iters"], weight_mode=params["weight_mode"],
            )
            row = evaluate(cfg_name, truth_flat, prism.abundances, prism.reconstructed, spectra_norm,
                           height, width, time.perf_counter() - t0)
            row["dataset"] = synthetic_root.name
            all_rows.append(row)

        sample_df = pd.DataFrame([r for r in all_rows if r["dataset"] == synthetic_root.name])
        print(sample_df[["method", "mae", "rmse", "pearson_r", "recon_rmse", "spatial_tv", "elapsed_s"]].to_string(
            index=False, float_format="%.4f"))

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(args.output_root / "prism_synth_std_vs_uni_summary.csv", index=False, encoding="utf-8-sig")
    (args.output_root / "prism_synth_std_vs_uni_summary.json").write_text(
        json.dumps({"configs": {"NNLS": {}, **PRISM_CONFIGS}, "rows": all_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[synth-std-vs-uni] full summary -> {args.output_root}")


if __name__ == "__main__":
    main()
