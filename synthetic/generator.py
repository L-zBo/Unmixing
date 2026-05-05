from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from preprocessing.endmembers import EndmemberLibrary


@dataclass(frozen=True)
class SyntheticMapConfig:
    width: int = 40
    height: int = 40
    smooth_sigma: float = 3.0
    noise_std: float = 0.01
    baseline_scale: float = 0.02
    scale_jitter: float = 0.05
    random_seed: int = 42


@dataclass(frozen=True)
class SyntheticMapResult:
    component_names: tuple[str, ...]
    axis: np.ndarray
    abundances: np.ndarray
    spectra: np.ndarray
    endmember_matrix: np.ndarray
    width: int
    height: int

    def flatten_abundance_frame(self) -> pd.DataFrame:
        rows: list[dict[str, float | int | str]] = []
        pixel_index = 0
        for y_idx in range(self.height):
            for x_idx in range(self.width):
                row: dict[str, float | int | str] = {
                    "pixel_index": pixel_index,
                    "x_idx": x_idx,
                    "y_idx": y_idx,
                }
                for component_index, name in enumerate(self.component_names):
                    row[f"abundance_{name}"] = float(self.abundances[y_idx, x_idx, component_index])
                rows.append(row)
                pixel_index += 1
        return pd.DataFrame(rows)


def generate_smooth_abundance_map(
    component_names: tuple[str, ...] | list[str],
    config: SyntheticMapConfig,
) -> np.ndarray:
    rng = np.random.default_rng(config.random_seed)
    raw = rng.random((len(component_names), config.height, config.width), dtype=np.float32)
    smoothed = np.stack([gaussian_filter(layer, sigma=config.smooth_sigma) for layer in raw], axis=0)
    denominator = np.sum(smoothed, axis=0, keepdims=True)
    denominator = np.maximum(denominator, 1e-12)
    normalized = smoothed / denominator
    return np.moveaxis(normalized.astype(np.float32), 0, -1)


def _baseline_curve(axis: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    if scale <= 0:
        return np.zeros_like(axis, dtype=np.float32)
    axis_scaled = (axis - axis.min()) / max(float(axis.max() - axis.min()), 1e-12)
    coeffs = rng.normal(loc=0.0, scale=scale, size=3).astype(np.float32)
    baseline = coeffs[0] + coeffs[1] * axis_scaled + coeffs[2] * (axis_scaled**2)
    return baseline.astype(np.float32)


def synthesize_from_abundances(
    library: EndmemberLibrary,
    abundances: np.ndarray,
    config: SyntheticMapConfig,
) -> SyntheticMapResult:
    if abundances.ndim != 3:
        raise ValueError("abundances must be a 3D array with shape (height, width, n_components).")
    if abundances.shape[2] != library.n_endmembers:
        raise ValueError(
            f"Abundance component count {abundances.shape[2]} does not match library size {library.n_endmembers}."
        )

    rng = np.random.default_rng(config.random_seed)
    flat_abundances = abundances.reshape(-1, abundances.shape[2]).astype(np.float32)
    spectra = flat_abundances @ library.matrix.T

    for index in range(spectra.shape[0]):
        scale = float(1.0 + rng.normal(0.0, config.scale_jitter))
        baseline = _baseline_curve(library.axis, rng, config.baseline_scale)
        noise = rng.normal(0.0, config.noise_std, size=library.n_points).astype(np.float32)
        spectra[index] = np.clip(scale * spectra[index] + baseline + noise, 0.0, None)

    return SyntheticMapResult(
        component_names=library.names,
        axis=library.axis.copy(),
        abundances=abundances.astype(np.float32, copy=False),
        spectra=spectra.astype(np.float32, copy=False),
        endmember_matrix=library.matrix.copy(),
        width=config.width,
        height=config.height,
    )


def generate_synthetic_map(
    library: EndmemberLibrary,
    config: SyntheticMapConfig = SyntheticMapConfig(),
) -> SyntheticMapResult:
    abundances = generate_smooth_abundance_map(library.names, config)
    return synthesize_from_abundances(library, abundances, config)


def save_synthetic_map(result: SyntheticMapResult, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    np.save(output_root / "spectra.npy", result.spectra)
    np.save(output_root / "abundances.npy", result.abundances)
    np.save(output_root / "axis.npy", result.axis)
    np.save(output_root / "endmember_matrix.npy", result.endmember_matrix)

    metadata = {
        "component_names": list(result.component_names),
        "width": int(result.width),
        "height": int(result.height),
        "n_pixels": int(result.width * result.height),
        "n_points": int(result.axis.size),
    }
    (output_root / "metadata.json").write_text(pd.Series(metadata).to_json(force_ascii=False, indent=2), encoding="utf-8")

    result.flatten_abundance_frame().to_csv(output_root / "abundance_truth.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {"RamanShift_cm-1": result.axis, **{f"endmember_{name}": result.endmember_matrix[:, idx] for idx, name in enumerate(result.component_names)}}
    ).to_csv(output_root / "endmembers.csv", index=False, encoding="utf-8-sig")
