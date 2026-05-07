"""Compare ALS+L2 / ALS+max / none+L2 on consistency and fingerprint-peak retention.

Targets the second PPT claim ("ALS+L2 is the best preprocessing combination")
where reconstruction-R² alone cannot tell L2 from max (both 0.9153 — see
preliminary findings 2026-05-06). Two new lines of evidence:

1. **Cross-pixel abundance consistency (CV)**: under each protocol, run NNLS on
   real mappings and compute the per-component coefficient of variation
   (std/mean) across pixels. Lower CV = more consistent recovery. L2-based
   protocols are expected to be more robust than max (literature: MDPI 2025).
2. **Fingerprint-peak retention**: in normalized spectra, measure the peak
   intensity in a window around literature-known fingerprint wavenumbers
   (PE 1062/1295, PP 808/841, starch 478/1124 cm-1) and compare across
   protocols. L2 normalization preserves relative peak intensities; max
   normalization distorts them when a single peak dominates.

Outputs
-------
- ``protocol_consistency_summary.csv``: per (sample, component, protocol) row
  with abundance mean / std / CV across pixels.
- ``fingerprint_retention_summary.csv``: per (sample, component, peak, protocol)
  row with mean relative intensity (normalized spectrum) at the peak window.
- Two PNGs via ``visualization.preprocessing.protocol_consistency``.

Read-only on dataset/. Mirrors final artefacts to ``outputs/showcase/protocol_consistency/``.
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
    PREPROCESS_PROTOCOLS,
    TARGET_AXIS,
    load_spectrum,
    preprocess_record,
)
from unmixing.unmix import unmix_spectra
from visualization.preprocessing import (
    plot_fingerprint_retention_bars,
    plot_protocol_cv_bars,
    plot_protocol_reconstruction_r2_bars,
)


PE_STARCH_DIR = "PE+淀粉"
PP_STARCH_DIR = "PP+淀粉"
PP_PE_STARCH_DIR = "PP+PE+淀粉"

CONSISTENCY_PRESETS: list[dict[str, object]] = [
    {
        "label": "PE+starch",
        "sample_dir": Path(PE_STARCH_DIR) / "1 785mw 2s 2 2 40 40",
        "components": ("PE", "starch"),
    },
    {
        "label": "PP+starch",
        "sample_dir": Path(PP_STARCH_DIR) / "1 785mw 2s 2 2 40 40",
        "components": ("PP", "starch"),
    },
    {
        "label": "PP+PE+starch",
        "sample_dir": Path("test") / PP_PE_STARCH_DIR / "1 785mw 2s 1 1 40 40",
        "components": ("PE", "PP", "starch"),
    },
]

# Literature fingerprint peaks (cm-1). Sources cited in conversation 2026-05-06.
FINGERPRINT_PEAKS: dict[str, tuple[int, ...]] = {
    "PE": (1062, 1295),
    "PP": (808, 841),
    "starch": (478, 1124),
}
PEAK_HALF_WINDOW_CM1 = 15  # search +/- 15 cm-1 for the local max

PROTOCOLS = ("als_l2", "als_max", "none_l2")
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/formal_v14_protocol_consistency"
SHOWCASE_OUTPUT_ROOT = ROOT / "outputs/showcase/protocol_consistency"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ALS+L2 / ALS+max / none+L2 on cross-pixel CV and fingerprint-peak retention.",
    )
    parser.add_argument("--input-root", type=Path, default=ROOT / DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--starch-source", default="baseline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-showcase", action="store_true")
    return parser.parse_args()


def load_mapping_corrected_and_normalized(
    *,
    input_root: Path,
    sample_dir: Path,
    protocol_name: str,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (corrected, normalized) spectra arrays for a mapping under one protocol."""
    resolved_dir = sample_dir if sample_dir.is_absolute() else input_root / sample_dir
    csv_files = sorted(resolved_dir.glob("DATA-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No DATA-*.csv files found in {resolved_dir}")
    corrected_list: list[np.ndarray] = []
    normalized_list: list[np.ndarray] = []
    for index, csv_path in enumerate(csv_files):
        record = load_spectrum(csv_path, input_root)
        _, corrected, normalized, _ = preprocess_record(record, protocol_name=protocol_name)
        corrected_list.append(corrected)
        normalized_list.append(normalized)
        if limit is not None and (index + 1) >= limit:
            break
    return (
        np.stack(corrected_list).astype(np.float32),
        np.stack(normalized_list).astype(np.float32),
    )


def compute_consistency_rows(
    *,
    sample_label: str,
    sample_dir: Path,
    components: tuple[str, ...],
    protocol_name: str,
    abundances: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for c_index, name in enumerate(components):
        col = abundances[:, c_index]
        mean_val = float(col.mean())
        std_val = float(col.std())
        cv = std_val / mean_val if mean_val > 1e-12 else float("nan")
        rows.append(
            {
                "label": f"{sample_label}|{name}",
                "sample_label": sample_label,
                "sample_dir": sample_dir.as_posix(),
                "component": name,
                "protocol": protocol_name,
                "n_pixels": int(abundances.shape[0]),
                "mean_abundance": mean_val,
                "std_abundance": std_val,
                "cv": cv,
            }
        )
    return rows


def find_peak_index(axis: np.ndarray, target_cm1: float, half_window: float) -> tuple[int, int]:
    lo = float(target_cm1 - half_window)
    hi = float(target_cm1 + half_window)
    mask = (axis >= lo) & (axis <= hi)
    indices = np.where(mask)[0]
    if indices.size == 0:
        # Fall back to nearest single index.
        nearest = int(np.argmin(np.abs(axis - target_cm1)))
        return nearest, nearest
    return int(indices.min()), int(indices.max())


def compute_fingerprint_retention_rows(
    *,
    sample_label: str,
    sample_dir: Path,
    components_present: tuple[str, ...],
    protocol_name: str,
    normalized_spectra: np.ndarray,
    axis: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for component in components_present:
        peaks = FINGERPRINT_PEAKS.get(component, ())
        for peak_cm1 in peaks:
            lo_index, hi_index = find_peak_index(axis, float(peak_cm1), float(PEAK_HALF_WINDOW_CM1))
            window = normalized_spectra[:, lo_index : hi_index + 1]
            peak_intensity = window.max(axis=1) if window.size > 0 else np.zeros(normalized_spectra.shape[0])
            rows.append(
                {
                    "sample_label": sample_label,
                    "sample_dir": sample_dir.as_posix(),
                    "component": component,
                    "peak_cm1": int(peak_cm1),
                    "protocol": protocol_name,
                    "n_pixels": int(normalized_spectra.shape[0]),
                    "relative_intensity_mean": float(peak_intensity.mean()),
                    "relative_intensity_std": float(peak_intensity.std()),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    consistency_rows: list[dict[str, object]] = []
    retention_rows: list[dict[str, object]] = []
    r2_rows: list[dict[str, object]] = []

    for preset in CONSISTENCY_PRESETS:
        sample_label: str = str(preset["label"])
        sample_dir: Path = preset["sample_dir"]  # type: ignore[assignment]
        components: tuple[str, ...] = preset["components"]  # type: ignore[assignment]

        for protocol_name in PROTOCOLS:
            _, normalized_spectra = load_mapping_corrected_and_normalized(
                input_root=args.input_root,
                sample_dir=sample_dir,
                protocol_name=protocol_name,
                limit=args.limit,
            )
            library = build_default_endmember_library(
                input_root=args.input_root,
                include_components=components,
                starch_source=args.starch_source,
                feature_mode="normalized",
                protocol_name=protocol_name,
            )
            result = unmix_spectra(normalized_spectra, library=library, method="nnls")
            r2_rows.append(
                {
                    "sample_label": sample_label,
                    "sample_dir": sample_dir.as_posix(),
                    "protocol": protocol_name,
                    "n_pixels": int(result.residual_r2.shape[0]),
                    "mean_residual_r2": float(result.residual_r2.mean()),
                    "mean_residual_rmse": float(result.residual_rmse.mean()),
                }
            )
            consistency_rows.extend(
                compute_consistency_rows(
                    sample_label=sample_label,
                    sample_dir=sample_dir,
                    components=components,
                    protocol_name=protocol_name,
                    abundances=result.abundances,
                )
            )
            retention_rows.extend(
                compute_fingerprint_retention_rows(
                    sample_label=sample_label,
                    sample_dir=sample_dir,
                    components_present=components,
                    protocol_name=protocol_name,
                    normalized_spectra=normalized_spectra,
                    axis=TARGET_AXIS,
                )
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    consistency_df = pd.DataFrame(consistency_rows)
    retention_df = pd.DataFrame(retention_rows)
    r2_df = pd.DataFrame(r2_rows)
    consistency_df.to_csv(args.output_root / "protocol_consistency_summary.csv", index=False, encoding="utf-8-sig")
    retention_df.to_csv(args.output_root / "fingerprint_retention_summary.csv", index=False, encoding="utf-8-sig")
    r2_df.to_csv(args.output_root / "protocol_reconstruction_r2_summary.csv", index=False, encoding="utf-8-sig")

    plot_protocol_reconstruction_r2_bars(
        r2_df=r2_df,
        output_path=args.output_root / "protocol_reconstruction_r2_bars.png",
        title="NNLS reconstruction R^2 by preprocessing protocol",
    )
    plot_protocol_cv_bars(
        cv_df=consistency_df,
        output_path=args.output_root / "protocol_cv_bars.png",
        title="Cross-pixel abundance CV by protocol (NNLS, lower = more consistent)",
    )
    plot_fingerprint_retention_bars(
        retention_df=retention_df,
        output_path=args.output_root / "fingerprint_retention_bars.png",
        title="Fingerprint-peak intensity in normalized spectrum by protocol",
    )

    if not args.no_showcase:
        SHOWCASE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        for fname in (
            "protocol_consistency_summary.csv",
            "fingerprint_retention_summary.csv",
            "protocol_reconstruction_r2_summary.csv",
            "protocol_reconstruction_r2_bars.png",
        ):
            (SHOWCASE_OUTPUT_ROOT / fname).write_bytes((args.output_root / fname).read_bytes())

    summary_json = {
        "input_root": args.input_root.as_posix(),
        "output_root": args.output_root.as_posix(),
        "protocols_tested": list(PROTOCOLS),
        "starch_source": args.starch_source,
        "fingerprint_peaks_cm1": {k: list(v) for k, v in FINGERPRINT_PEAKS.items()},
        "peak_half_window_cm1": PEAK_HALF_WINDOW_CM1,
        "n_consistency_rows": len(consistency_rows),
        "n_retention_rows": len(retention_rows),
        "interpretation_note": (
            "CV (std/mean) lower = abundance recovery more consistent across pixels of the "
            "same physical sample — this is the L2-vs-max stability indicator that "
            "reconstruction-R² could not distinguish. Fingerprint retention shows how each "
            "protocol distorts (or preserves) the literature peak intensities; max-normalization "
            "is expected to underweight non-dominant peaks."
        ),
    }
    (args.output_root / "protocol_consistency_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_consistency_rows": len(consistency_rows),
                "n_retention_rows": len(retention_rows),
                "output_root": args.output_root.as_posix(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
