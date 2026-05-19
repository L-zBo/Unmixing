"""Quick PRISM vs NNLS check on synthetic ground-truth data."""

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


DEFAULT_SYNTHETIC_ROOT = ROOT / "outputs/synthetic_unmixing/smoke_test_als_l2"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/prism_quick_check"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--protocol", choices=sorted(PREPROCESS_PROTOCOLS), default=DEFAULT_PROTOCOL_NAME)
    parser.add_argument("--lambda-l2", type=float, default=1e-4)
    parser.add_argument("--lambda-tv", type=float, default=0.02)
    parser.add_argument("--tv-iters", type=int, default=2)
    return parser.parse_args()


def load_synthetic_bundle(synthetic_root: Path):
    """Read metadata + arrays from a synthetic dataset directory."""
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
    height = int(metadata["height"])
    width = int(metadata["width"])
    return library, abundances.astype(np.float32), spectra.astype(np.float32), height, width


def preprocess_synthetic_spectra(axis: np.ndarray, spectra: np.ndarray, protocol_name: str):
    """Apply the same preprocessing pipeline to synthetic spectra."""
    processed = []
    corrected_norms = []
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
        _, corrected, normalized, _ = preprocess_record(record, protocol_name=protocol_name)
        processed.append(normalized)
        corrected_norms.append(float(np.linalg.norm(corrected)))
    return np.stack(processed).astype(np.float32), np.asarray(corrected_norms, dtype=np.float32)


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


def abundance_metrics(truth_flat: np.ndarray, pred_flat: np.ndarray) -> dict[str, float]:
    err = pred_flat - truth_flat
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "pearson_r": per_component_pearson(truth_flat, pred_flat),
    }


def spatial_tv(abundance_map: np.ndarray) -> float:
    """Mean total variation of an (H, W, K) abundance map; lower = smoother."""
    dx = np.abs(np.diff(abundance_map, axis=1))
    dy = np.abs(np.diff(abundance_map, axis=0))
    return float((dx.mean() + dy.mean()) / 2.0)


def evaluate(method_name: str, truth_flat: np.ndarray, abundances: np.ndarray,
             reconstructed: np.ndarray, spectra_norm: np.ndarray, height: int, width: int,
             elapsed: float) -> dict[str, float | str]:
    pred = row_normalize_nonneg(abundances)
    metrics = abundance_metrics(truth_flat, pred)
    residual = spectra_norm - reconstructed
    metrics["recon_rmse"] = float(np.sqrt(np.mean(residual * residual)))
    metrics["spatial_tv"] = spatial_tv(pred.reshape(height, width, -1))
    active = (pred > 1e-3).sum(axis=1).astype(np.float32)
    metrics["mean_active_endmembers"] = float(active.mean())
    metrics["elapsed_s"] = float(elapsed)
    return {"method": method_name, **metrics}


def format_table(rows: list[dict]) -> str:
    cols = ["method", "mae", "rmse", "pearson_r", "recon_rmse", "spatial_tv", "mean_active_endmembers", "elapsed_s"]
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols
        ))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    library, truth_abundances, spectra_raw, height, width = load_synthetic_bundle(args.synthetic_root)
    truth_flat = truth_abundances.reshape(-1, library.n_endmembers)
    spectra_norm, _ = preprocess_synthetic_spectra(library.axis, spectra_raw, protocol_name=args.protocol)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    t0 = time.perf_counter()
    nnls_result = unmix_spectra(spectra_norm, library=library, method="nnls")
    rows.append(evaluate("nnls", truth_flat, nnls_result.abundances, nnls_result.reconstructed,
                         spectra_norm, height, width, time.perf_counter() - t0))

    t0 = time.perf_counter()
    prism_core = prism_unmix_spectra(spectra_norm, library=library,
                                     image_shape=None, lambda_l2=args.lambda_l2,
                                     weight_mode="endmember_std")
    rows.append(evaluate("prism_core_wL2", truth_flat, prism_core.abundances, prism_core.reconstructed,
                         spectra_norm, height, width, time.perf_counter() - t0))

    t0 = time.perf_counter()
    prism_full = prism_unmix_spectra(spectra_norm, library=library,
                                     image_shape=(height, width),
                                     lambda_l2=args.lambda_l2,
                                     lambda_tv=args.lambda_tv,
                                     tv_iters=args.tv_iters,
                                     weight_mode="endmember_std")
    rows.append(evaluate("prism_full_wL2_TV", truth_flat, prism_full.abundances, prism_full.reconstructed,
                         spectra_norm, height, width, time.perf_counter() - t0))

    df = pd.DataFrame(rows)
    df.to_csv(args.output_root / "prism_quick_check_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== PRISM quick check on {args.synthetic_root.name} ({height}x{width}) ===")
    print(format_table(rows))

    summary_payload = {
        "synthetic_root": args.synthetic_root.as_posix(),
        "protocol": args.protocol,
        "image_shape": [height, width],
        "n_pixels": int(height * width),
        "lambda_l2": float(args.lambda_l2),
        "lambda_tv": float(args.lambda_tv),
        "tv_iters": int(args.tv_iters),
        "prism_full_config": prism_full.config,
        "rows": rows,
    }
    (args.output_root / "prism_quick_check_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
