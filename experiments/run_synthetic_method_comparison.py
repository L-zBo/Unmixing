from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.endmembers import EndmemberLibrary
from preprocessing.preprocess import DEFAULT_PROTOCOL_NAME, PREPROCESS_PROTOCOLS, SpectrumRecord, preprocess_record
from unmixing.unmix import (
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


def abundance_metrics(truth: np.ndarray, prediction: np.ndarray, prefix: str) -> dict[str, float]:
    error = prediction - truth
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error * error)))
    truth_centered = truth - truth.mean(axis=0, keepdims=True)
    ss_tot = float(np.sum(truth_centered * truth_centered))
    ss_res = float(np.sum(error * error))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {f"{prefix}_mae": mae, f"{prefix}_rmse": rmse, f"{prefix}_r2": r2}


def spectral_angle_mapper(truth: np.ndarray, prediction: np.ndarray) -> float:
    numerator = np.sum(truth * prediction, axis=1)
    denominator = np.linalg.norm(truth, axis=1) * np.linalg.norm(prediction, axis=1)
    cosine = np.divide(numerator, np.maximum(denominator, 1e-12))
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.mean(np.arccos(cosine)))


def row_normalize_nonneg(prediction: np.ndarray) -> np.ndarray:
    clipped = np.clip(prediction, 0.0, None)
    row_sum = clipped.sum(axis=1, keepdims=True)
    safe = np.where(row_sum > 1e-12, row_sum, 1.0)
    return (clipped / safe).astype(np.float32, copy=False)


def summarize_method(
    method: str,
    truth_orig: np.ndarray,
    truth_proj: np.ndarray,
    prediction: np.ndarray,
    residual_rmse: np.ndarray,
    reconstructed: np.ndarray,
    spectra_norm: np.ndarray,
) -> dict[str, object]:
    prediction_renorm = row_normalize_nonneg(prediction)
    metrics: dict[str, object] = {"method": method}
    metrics.update(abundance_metrics(truth_orig, prediction_renorm, prefix="orig"))
    metrics.update(abundance_metrics(truth_proj, prediction.astype(np.float32, copy=False), prefix="proj"))
    metrics.update(
        {
            "mean_residual_rmse": float(np.mean(residual_rmse)),
            "mean_sam_rad": spectral_angle_mapper(spectra_norm, reconstructed),
        }
    )
    return metrics


def preprocess_synthetic_spectra(
    axis: np.ndarray,
    spectra: np.ndarray,
    protocol_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    processed: list[np.ndarray] = []
    corrected_norms: list[float] = []
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
    return (
        np.stack(processed).astype(np.float32),
        np.asarray(corrected_norms, dtype=np.float32),
    )


def build_projection_truth(abundances: np.ndarray, corrected_norms: np.ndarray) -> np.ndarray:
    safe = np.where(corrected_norms > 1e-12, corrected_norms, 1.0)
    return (abundances / safe[:, None]).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    library, truth_abundances, spectra_raw = load_synthetic_bundle(args.synthetic_root)
    spectra_norm, corrected_norms = preprocess_synthetic_spectra(
        library.axis, spectra_raw, protocol_name=args.protocol
    )
    truth_proj = build_projection_truth(truth_abundances, corrected_norms)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for method in ("ols", "nnls", "fcls"):
        result = unmix_spectra(spectra_norm, library=library, method=method)
        rows.append(
            summarize_method(
                method=method,
                truth_orig=truth_abundances,
                truth_proj=truth_proj,
                prediction=result.abundances,
                residual_rmse=result.residual_rmse,
                reconstructed=result.reconstructed,
                spectra_norm=spectra_norm,
            )
        )

    nmf_components = args.nmf_components or library.n_endmembers
    nmf_result = blind_nmf_unmix_spectra(spectra_norm, n_components=nmf_components)
    nmf_result, similarity_df = align_blind_nmf_to_reference(nmf_result, library)
    similarity_df.to_csv(args.output_root / "nmf_reference_similarity.csv", encoding="utf-8-sig")
    rows.append(
        summarize_method(
            method="nmf",
            truth_orig=truth_abundances,
            truth_proj=truth_proj,
            prediction=nmf_result.abundances,
            residual_rmse=nmf_result.residual_rmse,
            reconstructed=nmf_result.reconstructed,
            spectra_norm=spectra_norm,
        )
    )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.output_root / "synthetic_method_comparison_summary.csv", index=False, encoding="utf-8-sig")

    metric_glossary = {
        "orig_mae/orig_rmse/orig_r2": (
            "估计丰度先 clip 到非负再按行归一化为相对占比，与原始 0..1 像素丰度真值直接比较；"
            "解释为「相对组分占比恢复」，与丰度图视觉直觉对齐。"
        ),
        "proj_mae/proj_rmse/proj_r2": (
            "把估计系数与「原始丰度 / 预处理后(corrected)谱的 L2 范数」的理论投影真值做对比；"
            "解释为「预处理归一化空间内的系数恢复」，无需对方法做行归一化，能直接区分各方法在归一化谱上的精度。"
        ),
        "mean_residual_rmse": "归一化空间内每像素重构 RMSE 的平均值。",
        "mean_sam_rad": "归一化空间内输入谱与重构谱的逐像素 SAM 弧度的平均值。",
    }

    summary_json = {
        "synthetic_root": args.synthetic_root.as_posix(),
        "protocol": args.protocol,
        "component_names": list(library.names),
        "n_pixels": int(spectra_norm.shape[0]),
        "n_points": int(spectra_norm.shape[1]),
        "corrected_norm_stats": {
            "mean": float(np.mean(corrected_norms)),
            "std": float(np.std(corrected_norms)),
            "min": float(np.min(corrected_norms)),
            "max": float(np.max(corrected_norms)),
        },
        "metric_glossary": metric_glossary,
        "interpretation_note": (
            "ALS+L2 等含逐谱归一化的协议下，原始丰度真值与解混系数不在同一标度。"
            "请把 orig_* 与 proj_* 两组指标和 mean_residual_rmse / mean_sam_rad 联合解读，"
            "不要单看任一项数值。"
        ),
        "methods": rows,
    }
    (args.output_root / "synthetic_method_comparison_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_json)


if __name__ == "__main__":
    main()
