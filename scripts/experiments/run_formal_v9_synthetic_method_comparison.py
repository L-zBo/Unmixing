from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demixing.data.endmembers import EndmemberLibrary
from demixing.data.preprocess import DEFAULT_PROTOCOL_NAME, PREPROCESS_PROTOCOLS, SpectrumRecord, preprocess_record
from demixing.evaluation.classical_unmixing import (
    align_blind_nmf_to_reference,
    blind_nmf_unmix_spectra,
    unmix_spectra,
)


DEFAULT_SYNTHETIC_ROOT = ROOT / "outputs/synthetic_unmixing/smoke_test_als_l2"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/formal_v9_synthetic_method_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OLS/NNLS/FCLS/NMF on a synthetic Raman unmixing dataset with ground truth.")
    parser.add_argument("--synthetic-root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--nmf-components", type=int, default=None)
    parser.add_argument("--protocol", choices=sorted(PREPROCESS_PROTOCOLS), default=DEFAULT_PROTOCOL_NAME)
    return parser.parse_args()


def load_synthetic_bundle(synthetic_root: Path) -> tuple[EndmemberLibrary, np.ndarray, np.ndarray]:
    metadata = json.loads((synthetic_root / "metadata.json").read_text(encoding="utf-8"))
    component_names = tuple(metadata["component_names"])
    axis = np.load(synthetic_root / "axis.npy")
    endmember_matrix = np.load(synthetic_root / "endmember_matrix.npy")
    abundances = np.load(synthetic_root / "abundances.npy").reshape(-1, len(component_names))
    spectra = np.load(synthetic_root / "spectra.npy")
    library = EndmemberLibrary(
        names=component_names,
        axis=axis.astype(np.float32),
        matrix=endmember_matrix.astype(np.float32),
        feature_mode="normalized",
        source_paths={name: Path(name) for name in component_names},
    )
    return library, abundances.astype(np.float32), spectra.astype(np.float32)


def abundance_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error * error)))
    truth_centered = truth - truth.mean(axis=0, keepdims=True)
    ss_tot = float(np.sum(truth_centered * truth_centered))
    ss_res = float(np.sum(error * error))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def spectral_angle_mapper(truth: np.ndarray, prediction: np.ndarray) -> float:
    numerator = np.sum(truth * prediction, axis=1)
    denominator = np.linalg.norm(truth, axis=1) * np.linalg.norm(prediction, axis=1)
    cosine = np.divide(numerator, np.maximum(denominator, 1e-12))
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.mean(np.arccos(cosine)))


def summarize_method(method: str, truth: np.ndarray, prediction: np.ndarray, residual_rmse: np.ndarray, reconstructed: np.ndarray, spectra: np.ndarray) -> dict[str, object]:
    metrics = abundance_metrics(truth, prediction)
    metrics.update(
        {
            "method": method,
            "mean_residual_rmse": float(np.mean(residual_rmse)),
            "mean_sam": spectral_angle_mapper(spectra, reconstructed),
        }
    )
    return metrics


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


def main() -> None:
    args = parse_args()
    library, truth_abundances, spectra = load_synthetic_bundle(args.synthetic_root)
    spectra = preprocess_synthetic_spectra(library.axis, spectra, protocol_name=args.protocol)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for method in ("ols", "nnls", "fcls"):
        result = unmix_spectra(spectra, library=library, method=method)
        rows.append(
            summarize_method(
                method=method,
                truth=truth_abundances,
                prediction=result.abundances,
                residual_rmse=result.residual_rmse,
                reconstructed=result.reconstructed,
                spectra=spectra,
            )
        )

    nmf_components = args.nmf_components or library.n_endmembers
    nmf_result = blind_nmf_unmix_spectra(spectra, n_components=nmf_components)
    nmf_result, similarity_df = align_blind_nmf_to_reference(nmf_result, library)
    similarity_df.to_csv(args.output_root / "nmf_reference_similarity.csv", encoding="utf-8-sig")
    rows.append(
        summarize_method(
            method="nmf",
            truth=truth_abundances,
            prediction=nmf_result.abundances,
            residual_rmse=nmf_result.residual_rmse,
            reconstructed=nmf_result.reconstructed,
            spectra=spectra,
        )
    )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.output_root / "synthetic_method_comparison_summary.csv", index=False, encoding="utf-8-sig")
    summary_json = {
        "synthetic_root": args.synthetic_root.as_posix(),
        "protocol": args.protocol,
        "component_names": list(library.names),
        "n_pixels": int(spectra.shape[0]),
        "n_points": int(spectra.shape[1]),
        "note": "When the evaluation protocol includes spectrum-wise normalization such as ALS+L2, abundance ground truth and post-normalization linear coefficients are not strictly identical. Interpret abundance MAE/RMSE together with reconstruction metrics.",
        "methods": rows,
    }
    (args.output_root / "synthetic_method_comparison_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_json)


if __name__ == "__main__":
    main()
