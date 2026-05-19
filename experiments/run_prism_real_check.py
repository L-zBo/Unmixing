"""Check PRISM (old / mid / aggressive) vs NNLS on real Raman mapping samples without ground truth."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/prism_real_check"

PE_STARCH_DIR = "PE+淀粉"
PP_STARCH_DIR = "PP+淀粉"
PP_PE_STARCH_DIR = "PP+PE+淀粉"


SAMPLES = [
    {
        "name": "PE_starch_train",
        "sample_dir": Path(PE_STARCH_DIR) / "1 785mw 2s 2 2 40 40",
        "components": ("PE", "starch"),
        "absent_components": ("PP",),
    },
    {
        "name": "PP_starch_train",
        "sample_dir": Path(PP_STARCH_DIR) / "1 785mw 2s 2 2 40 40",
        "components": ("PP", "starch"),
        "absent_components": ("PE",),
    },
    {
        "name": "PE_PP_starch_test",
        "sample_dir": Path("test") / PP_PE_STARCH_DIR / "1 785mw 2s 1 1 40 40",
        "components": ("PE", "PP", "starch"),
        "absent_components": (),
    },
]


PRISM_CONFIGS = {
    "PRISM_OLD":  {"lambda_l2": 1e-4, "lambda_tv": 0.02, "tv_iters": 2},
    "PRISM_MID":  {"lambda_l2": 1e-2, "lambda_tv": 0.10, "tv_iters": 2},
    "PRISM_AGG":  {"lambda_l2": 1e-2, "lambda_tv": 0.20, "tv_iters": 1},
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
    if mx is None or my is None:
        return None, None
    return int(mx.group(1)), int(my.group(1))


def load_mapping(input_root: Path, sample_dir: Path, feature_mode: str, protocol_name: str):
    """Load a mapping directory; returns (xy_array, spectra, height, width)."""
    resolved = sample_dir if sample_dir.is_absolute() else input_root / sample_dir
    csv_files = sorted(resolved.glob("DATA-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No DATA-*.csv files found in {resolved}")
    spectra = []
    xs, ys = [], []
    for path in csv_files:
        record = load_spectrum(path, input_root)
        _, corrected, normalized, _ = preprocess_record(record, protocol_name=protocol_name)
        spectra.append(normalized if feature_mode == "normalized" else corrected)
        x, y = parse_xy(path.name)
        xs.append(x)
        ys.append(y)
    spectra_arr = np.stack(spectra).astype(np.float32)
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    width = int(xs_arr.max() - xs_arr.min() + 1)
    height = int(ys_arr.max() - ys_arr.min() + 1)
    order = np.lexsort((xs_arr, ys_arr))
    return xs_arr[order], ys_arr[order], spectra_arr[order], height, width


def row_normalize_nonneg(prediction: np.ndarray) -> np.ndarray:
    clipped = np.clip(prediction, 0.0, None)
    row_sum = clipped.sum(axis=1, keepdims=True)
    safe = np.where(row_sum > 1e-12, row_sum, 1.0)
    return (clipped / safe).astype(np.float32)


def evaluate_method(
    method_name: str,
    component_names: tuple[str, ...],
    abundances: np.ndarray,
    reconstructed: np.ndarray,
    spectra: np.ndarray,
    absent_components: tuple[str, ...],
    height: int,
    width: int,
    elapsed: float,
) -> dict:
    abundance = row_normalize_nonneg(abundances)
    residual = spectra - reconstructed
    recon_rmse = float(np.sqrt(np.mean(residual * residual)))
    centered = spectra - spectra.mean(axis=1, keepdims=True)
    ss_tot = float(np.sum(centered * centered))
    ss_res = float(np.sum(residual * residual))
    recon_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    abundance_map = abundance.reshape(height, width, -1)
    dx = np.abs(np.diff(abundance_map, axis=1)).mean()
    dy = np.abs(np.diff(abundance_map, axis=0)).mean()
    spatial_tv = float((dx + dy) / 2)

    abundances_per_component = {name: float(abundance[:, i].mean()) for i, name in enumerate(component_names)}
    absent_load_total = float(sum(abundances_per_component.get(c, 0.0) for c in absent_components))

    mean_active = float((abundance > 1e-3).sum(axis=1).mean())
    negative_fraction = float((abundances.min(axis=1) < 0).mean())

    row = {
        "method": method_name,
        "recon_rmse": recon_rmse,
        "recon_r2": recon_r2,
        "spatial_tv": spatial_tv,
        "absent_load": absent_load_total,
        "mean_active": mean_active,
        "negative_fraction": negative_fraction,
        "elapsed_s": float(elapsed),
    }
    for name, value in abundances_per_component.items():
        row[f"mean_{name}"] = value
    return row, abundance_map


def plot_abundance_grid(sample_name: str, component_names: tuple[str, ...], maps: dict[str, np.ndarray],
                        output_path: Path) -> None:
    method_order = ["NNLS", "PRISM_OLD", "PRISM_MID", "PRISM_AGG"]
    n_rows = len(method_order)
    n_cols = len(component_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows), squeeze=False)
    for row, method in enumerate(method_order):
        amap = maps[method]
        for col, name in enumerate(component_names):
            ax = axes[row, col]
            im = ax.imshow(amap[..., col], vmin=0.0, vmax=1.0, cmap="viridis", origin="lower", aspect="equal")
            if row == 0:
                ax.set_title(name, fontsize=12)
            if col == 0:
                ax.set_ylabel(method, fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{sample_name} — abundance maps: NNLS vs PRISM variants", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for sample in SAMPLES:
        sample_name = sample["name"]
        components = sample["components"]
        absent = sample["absent_components"]
        print(f"\n=== {sample_name}  components={components}  absent={absent} ===")

        try:
            _, _, spectra, height, width = load_mapping(
                input_root=args.input_root, sample_dir=sample["sample_dir"],
                feature_mode=args.feature_mode, protocol_name=args.protocol,
            )
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue

        library = build_default_endmember_library(
            input_root=args.input_root,
            include_components=components,
            starch_source="baseline",
            feature_mode=args.feature_mode,
            protocol_name=args.protocol,
        )

        maps: dict[str, np.ndarray] = {}

        t0 = time.perf_counter()
        nnls = unmix_spectra(spectra, library=library, method="nnls")
        nnls_row, nnls_map = evaluate_method("NNLS", library.names, nnls.abundances, nnls.reconstructed,
                                             spectra, absent, height, width, time.perf_counter() - t0)
        nnls_row["sample"] = sample_name
        all_rows.append(nnls_row)
        maps["NNLS"] = nnls_map

        for config_name, params in PRISM_CONFIGS.items():
            t0 = time.perf_counter()
            prism = prism_unmix_spectra(
                spectra, library=library,
                image_shape=(height, width),
                lambda_l2=params["lambda_l2"],
                lambda_tv=params["lambda_tv"],
                tv_iters=params["tv_iters"],
                weight_mode="endmember_std",
            )
            row, amap = evaluate_method(config_name, library.names, prism.abundances, prism.reconstructed,
                                        spectra, absent, height, width, time.perf_counter() - t0)
            row["sample"] = sample_name
            all_rows.append(row)
            maps[config_name] = amap

        df_sample = pd.DataFrame([r for r in all_rows if r["sample"] == sample_name])
        cols = ["method", "recon_rmse", "recon_r2", "spatial_tv", "absent_load",
                "mean_active", "negative_fraction", "elapsed_s"]
        print(df_sample[cols].to_string(index=False, float_format="%.4f"))

        plot_abundance_grid(
            sample_name=sample_name,
            component_names=library.names,
            maps=maps,
            output_path=args.output_root / f"{sample_name}_abundance_grid.png",
        )

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(args.output_root / "prism_real_check_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "samples": [s["name"] for s in SAMPLES],
        "configs": {**{"NNLS": {}}, **PRISM_CONFIGS},
        "feature_mode": args.feature_mode,
        "protocol": args.protocol,
        "rows": all_rows,
    }
    (args.output_root / "prism_real_check_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[real-check] full summary -> {args.output_root}")


if __name__ == "__main__":
    main()
