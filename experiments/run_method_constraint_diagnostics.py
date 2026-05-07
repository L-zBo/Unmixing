"""Diagnose constraint behavior of OLS / NNLS / FCLS / NMF on real Raman mappings.

Targets two PPT-grade claims that the existing showcase data can't yet prove:

1. **OLS gives unphysical negative abundances; NNLS does not.**
   Existing batch_method_comparison aggregates *means* and they happen to be
   positive on baseline endmember setups (OLS naturally non-negative). This
   experiment reports the **per-pixel** fraction of negative coefficients,
   focusing on the matched-endmember 泛化 scenarios where OLS does drift
   negative (see preliminary findings in conversation 2026-05-06).

2. **NMF learns endmembers that drift away from the physical references.**
   NMF's reconstruction R² > 0.99 looks great, but its abundance estimates of
   PE / PP are systematically inflated by 1.5x – 2.5x. This script measures
   the spectral angle (SAM) between the aligned NMF endmember and the physical
   reference pure spectrum — a direct readout of "how non-physical is the
   endmember NMF picked".

A bonus 3rd output:
- NNLS active-endmember count per pixel (sparsity proxy for interpretability).

Outputs
-------
- ``method_constraint_summary.csv``: per (label, method, component) row with
  ``neg_coef_fraction`` / ``min_coef`` / ``mean_abundance``.
- ``nnls_sparsity_summary.csv``: per-scenario NNLS active-endmember histogram.
- ``nmf_endmember_sam.csv``: per (label, component) SAM (rad) of NMF vs ref.
- ``negative_coef_fraction_bars.png`` and ``nmf_endmember_sam_bars.png``.

Read-only on dataset/. Mirrors final artefacts to ``outputs/showcase/method_constraints/``
unless ``--no-showcase`` is passed.
"""
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

from preprocessing.endmembers import build_default_endmember_library
from preprocessing.preprocess import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_PROTOCOL_NAME,
    PREPROCESS_PROTOCOLS,
    load_spectrum,
    preprocess_record,
)
from unmixing.unmix import (
    align_blind_nmf_to_reference,
    blind_nmf_unmix_spectra,
    unmix_spectra,
)
from visualization.method_comparison import (
    plot_negative_abundance_pct_bars,
    plot_nmf_endmember_sam_bars,
)


PE_STARCH_DIR = "PE+淀粉"
PP_STARCH_DIR = "PP+淀粉"
PP_PE_STARCH_DIR = "PP+PE+淀粉"

DIAGNOSTIC_PRESETS: list[dict[str, object]] = [
    # Main test set (baseline starch endmember).
    {
        "label": "main_PE_starch",
        "sample_dir": Path(PE_STARCH_DIR) / "1 785mw 2s 2 2 40 40",
        "starch_source": "baseline",
        "components": ("PE", "starch"),
    },
    {
        "label": "main_PP_starch",
        "sample_dir": Path(PP_STARCH_DIR) / "1 785mw 2s 2 2 40 40",
        "starch_source": "baseline",
        "components": ("PP", "starch"),
    },
    {
        "label": "main_PP_PE_starch",
        "sample_dir": Path("test") / PP_PE_STARCH_DIR / "1 785mw 2s 1 1 40 40",
        "starch_source": "baseline",
        "components": ("PE", "PP", "starch"),
    },
    # Generalization with matched (per-sample) starch endmember.
    # OLS-negative hotspots — matched mode forces a slight endmember mismatch
    # against ALS+L2 normalized spectra.
    {
        "label": "gen_zhanyi_PE_matched",
        "sample_dir": Path("泛化/展艺玉米淀粉/PE+淀粉/1 785mw 2s 1 1 40 40"),
        "starch_source": "展艺玉米淀粉",
        "components": ("PE", "starch"),
    },
    {
        "label": "gen_zhanyi_PP_matched",
        "sample_dir": Path("泛化/展艺玉米淀粉/PP+淀粉/1 785mw 2s 1 1 40 40"),
        "starch_source": "展艺玉米淀粉",
        "components": ("PP", "starch"),
    },
    {
        "label": "gen_xinliang_PE_matched",
        "sample_dir": Path("泛化/新良小麦淀粉/PE+淀粉/1 785mw 2s 2 2 40 40"),
        "starch_source": "新良小麦淀粉",
        "components": ("PE", "starch"),
    },
    {
        "label": "gen_xinliang_PP_matched",
        "sample_dir": Path("泛化/新良小麦淀粉/PP+淀粉/1 785mw 2s 2 2 40 40"),
        "starch_source": "新良小麦淀粉",
        "components": ("PP", "starch"),
    },
    {
        "label": "gen_ganzhiyuan_PE_matched",
        "sample_dir": Path("泛化/甘汁园小麦淀粉/PE+淀粉/1 785mw 2s 2 2 40 40"),
        "starch_source": "甘汁园小麦淀粉",
        "components": ("PE", "starch"),
    },
    {
        "label": "gen_ganzhiyuan_PP_matched",
        "sample_dir": Path("泛化/甘汁园小麦淀粉/PP+淀粉/1 785mw 2s 2 2 40 40"),
        "starch_source": "甘汁园小麦淀粉",
        "components": ("PP", "starch"),
    },
]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/formal_v13_method_constraint_diagnostics"
SHOWCASE_OUTPUT_ROOT = ROOT / "outputs/showcase/method_constraints"
ACTIVE_THRESHOLD = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose negative-coefficient / sparsity / NMF endmember-drift behavior of OLS/NNLS/FCLS/NMF.",
    )
    parser.add_argument("--input-root", type=Path, default=ROOT / DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--feature-mode", choices=["normalized", "corrected"], default="normalized")
    parser.add_argument("--protocol", choices=sorted(PREPROCESS_PROTOCOLS), default=DEFAULT_PROTOCOL_NAME)
    parser.add_argument("--methods", nargs="+", default=["ols", "nnls", "fcls", "nmf"])
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=ACTIVE_THRESHOLD,
        help="Relative-abundance threshold above which an endmember is counted as 'active' for sparsity.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-showcase", action="store_true", help="Skip mirroring artefacts to outputs/showcase/method_constraints/.")
    return parser.parse_args()


def load_mapping_spectra(
    input_root: Path,
    sample_dir: Path,
    feature_mode: str,
    protocol_name: str,
    limit: int | None = None,
) -> np.ndarray:
    resolved_dir = sample_dir if sample_dir.is_absolute() else input_root / sample_dir
    csv_files = sorted(resolved_dir.glob("DATA-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No DATA-*.csv files found in {resolved_dir}")
    spectra: list[np.ndarray] = []
    for index, csv_path in enumerate(csv_files):
        record = load_spectrum(csv_path, input_root)
        _, corrected, normalized, _ = preprocess_record(record, protocol_name=protocol_name)
        spectra.append(normalized if feature_mode == "normalized" else corrected)
        if limit is not None and (index + 1) >= limit:
            break
    return np.stack(spectra).astype(np.float32)


def diagnose_classical_method(
    *,
    coefficients: np.ndarray,
    component_names: tuple[str, ...],
    method: str,
    label: str,
    sample_dir: Path,
    starch_source: str,
) -> list[dict[str, object]]:
    n_pixels = coefficients.shape[0]
    nonneg = np.clip(coefficients, 0.0, None)
    row_sum = nonneg.sum(axis=1, keepdims=True)
    row_sum_safe = np.where(row_sum > 1e-12, row_sum, 1.0)
    abundance = nonneg / row_sum_safe
    rows: list[dict[str, object]] = []
    for c_index, name in enumerate(component_names):
        coef = coefficients[:, c_index]
        rows.append(
            {
                "label": label,
                "sample_dir": sample_dir.as_posix(),
                "starch_source": starch_source,
                "method": method,
                "component": name,
                "n_pixels": n_pixels,
                "neg_coef_count": int(np.sum(coef < 0)),
                "neg_coef_fraction": float(np.mean(coef < 0)),
                "min_coef": float(coef.min()),
                "max_coef": float(coef.max()),
                "mean_coef": float(coef.mean()),
                "mean_abundance": float(abundance[:, c_index].mean()),
                "std_abundance": float(abundance[:, c_index].std()),
            }
        )
    return rows


def diagnose_nnls_sparsity(
    *,
    coefficients: np.ndarray,
    label: str,
    sample_dir: Path,
    starch_source: str,
    active_threshold: float,
) -> dict[str, object]:
    nonneg = np.clip(coefficients, 0.0, None)
    row_sum = nonneg.sum(axis=1, keepdims=True)
    row_sum_safe = np.where(row_sum > 1e-12, row_sum, 1.0)
    abundance = nonneg / row_sum_safe
    active_count = (abundance > active_threshold).sum(axis=1)
    n_components = int(coefficients.shape[1])
    histogram = {f"active_{k}": int(np.sum(active_count == k)) for k in range(n_components + 1)}
    return {
        "label": label,
        "sample_dir": sample_dir.as_posix(),
        "starch_source": starch_source,
        "method": "nnls",
        "n_pixels": int(coefficients.shape[0]),
        "n_components": n_components,
        "active_threshold": active_threshold,
        "mean_active_count": float(np.mean(active_count)),
        "max_active_count": int(np.max(active_count)),
        **histogram,
    }


def sam_angle_rad(reference: np.ndarray, learned: np.ndarray) -> float:
    num = float(np.dot(reference, learned))
    den = float(np.linalg.norm(reference) * np.linalg.norm(learned))
    if den < 1e-12:
        return float("nan")
    cos = max(-1.0, min(1.0, num / den))
    return float(np.arccos(cos))


def diagnose_nmf_endmember_drift(
    *,
    aligned_endmembers: np.ndarray,
    reference_matrix: np.ndarray,
    component_names: tuple[str, ...],
    label: str,
    sample_dir: Path,
    starch_source: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for c_index, name in enumerate(component_names):
        ref = reference_matrix[:, c_index].astype(np.float64)
        learned = aligned_endmembers[:, c_index].astype(np.float64)
        rows.append(
            {
                "label": label,
                "sample_dir": sample_dir.as_posix(),
                "starch_source": starch_source,
                "component": name,
                "sam_rad": sam_angle_rad(ref, learned),
                "ref_l2": float(np.linalg.norm(ref)),
                "learned_l2": float(np.linalg.norm(learned)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    constraint_rows: list[dict[str, object]] = []
    sparsity_rows: list[dict[str, object]] = []
    nmf_drift_rows: list[dict[str, object]] = []

    for preset in DIAGNOSTIC_PRESETS:
        label: str = str(preset["label"])
        sample_dir: Path = preset["sample_dir"]  # type: ignore[assignment]
        starch_source: str = str(preset["starch_source"])
        components: tuple[str, ...] = preset["components"]  # type: ignore[assignment]

        spectra = load_mapping_spectra(
            input_root=args.input_root,
            sample_dir=sample_dir,
            feature_mode=args.feature_mode,
            protocol_name=args.protocol,
            limit=args.limit,
        )
        library = build_default_endmember_library(
            input_root=args.input_root,
            include_components=components,
            starch_source=starch_source,
            feature_mode=args.feature_mode,
            protocol_name=args.protocol,
        )

        for method in ("ols", "nnls", "fcls"):
            if method not in args.methods:
                continue
            result = unmix_spectra(spectra, library=library, method=method)
            constraint_rows.extend(
                diagnose_classical_method(
                    coefficients=result.coefficients,
                    component_names=result.component_names,
                    method=method,
                    label=label,
                    sample_dir=sample_dir,
                    starch_source=starch_source,
                )
            )
            if method == "nnls":
                sparsity_rows.append(
                    diagnose_nnls_sparsity(
                        coefficients=result.coefficients,
                        label=label,
                        sample_dir=sample_dir,
                        starch_source=starch_source,
                        active_threshold=args.active_threshold,
                    )
                )

        if "nmf" in args.methods:
            nmf_result = blind_nmf_unmix_spectra(spectra, n_components=len(components), max_iter=10000)
            aligned, _ = align_blind_nmf_to_reference(nmf_result, library)
            nmf_drift_rows.extend(
                diagnose_nmf_endmember_drift(
                    aligned_endmembers=aligned.endmember_matrix,
                    reference_matrix=library.matrix,
                    component_names=aligned.component_names,
                    label=label,
                    sample_dir=sample_dir,
                    starch_source=starch_source,
                )
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    constraint_df = pd.DataFrame(constraint_rows)
    sparsity_df = pd.DataFrame(sparsity_rows)
    nmf_drift_df = pd.DataFrame(nmf_drift_rows)
    constraint_df.to_csv(args.output_root / "method_constraint_summary.csv", index=False, encoding="utf-8-sig")
    sparsity_df.to_csv(args.output_root / "nnls_sparsity_summary.csv", index=False, encoding="utf-8-sig")
    nmf_drift_df.to_csv(args.output_root / "nmf_endmember_sam.csv", index=False, encoding="utf-8-sig")

    plot_negative_abundance_pct_bars(
        constraint_df=constraint_df,
        output_path=args.output_root / "negative_coef_fraction_bars.png",
        title="OLS / NNLS / FCLS  per-pixel negative-coefficient fraction",
    )
    plot_nmf_endmember_sam_bars(
        nmf_drift_df=nmf_drift_df,
        output_path=args.output_root / "nmf_endmember_sam_bars.png",
        title="NMF endmember vs reference pure spectrum (SAM, rad)",
    )

    if not args.no_showcase:
        SHOWCASE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        for fname in (
            "method_constraint_summary.csv",
            "nnls_sparsity_summary.csv",
            "nmf_endmember_sam.csv",
            "negative_coef_fraction_bars.png",
            "nmf_endmember_sam_bars.png",
        ):
            (SHOWCASE_OUTPUT_ROOT / fname).write_bytes((args.output_root / fname).read_bytes())

    summary_json = {
        "input_root": args.input_root.as_posix(),
        "output_root": args.output_root.as_posix(),
        "protocol": args.protocol,
        "feature_mode": args.feature_mode,
        "methods": args.methods,
        "active_threshold": args.active_threshold,
        "n_presets": len(DIAGNOSTIC_PRESETS),
        "n_constraint_rows": len(constraint_rows),
        "n_sparsity_rows": len(sparsity_rows),
        "n_nmf_drift_rows": len(nmf_drift_rows),
        "interpretation_note": (
            "neg_coef_fraction = per-pixel fraction of negative coefficients per component. "
            "OLS expected to spike on matched-endmember 泛化 mappings (gen_*_matched); "
            "NNLS/FCLS = 0 by construction. nmf_endmember_sam = drift between aligned NMF "
            "endmember and physical reference. Large SAM => NMF learned a non-physical "
            "endmember even when reconstruction R^2 looks good."
        ),
    }
    (args.output_root / "method_constraint_diagnostics.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_constraint_rows": len(constraint_rows),
                "n_sparsity_rows": len(sparsity_rows),
                "n_nmf_drift_rows": len(nmf_drift_rows),
                "output_root": args.output_root.as_posix(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
