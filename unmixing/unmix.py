from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize, nnls
from sklearn.decomposition import NMF as SklearnNMF

from preprocessing.endmembers import EndmemberLibrary


UnmixingMethod = Literal["ols", "nnls", "fcls"]


@dataclass(frozen=True)
class ClassicalUnmixingResult:
    component_names: tuple[str, ...]
    method: str
    coefficients: np.ndarray
    abundances: np.ndarray
    reconstructed: np.ndarray
    residual_l2: np.ndarray
    residual_rmse: np.ndarray
    residual_r2: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, float | int | str]] = []
        for index in range(self.coefficients.shape[0]):
            row: dict[str, float | int | str] = {
                "spectrum_index": index,
                "method": self.method,
                "residual_l2": float(self.residual_l2[index]),
                "residual_rmse": float(self.residual_rmse[index]),
                "residual_r2": float(self.residual_r2[index]),
            }
            for component_index, name in enumerate(self.component_names):
                row[f"coef_{name}"] = float(self.coefficients[index, component_index])
                row[f"abundance_{name}"] = float(self.abundances[index, component_index])
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class BlindNMFResult:
    component_names: tuple[str, ...]
    abundances: np.ndarray
    reconstructed: np.ndarray
    endmember_matrix: np.ndarray
    residual_l2: np.ndarray
    residual_rmse: np.ndarray
    residual_r2: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, float | int | str]] = []
        for index in range(self.abundances.shape[0]):
            row: dict[str, float | int | str] = {
                "spectrum_index": index,
                "method": "nmf",
                "residual_l2": float(self.residual_l2[index]),
                "residual_rmse": float(self.residual_rmse[index]),
                "residual_r2": float(self.residual_r2[index]),
            }
            for component_index, name in enumerate(self.component_names):
                row[f"abundance_{name}"] = float(self.abundances[index, component_index])
            rows.append(row)
        return pd.DataFrame(rows)


def _ensure_2d_spectra(spectra: np.ndarray) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.ndim == 1:
        return spectra.reshape(1, -1)
    if spectra.ndim != 2:
        raise ValueError(f"spectra must be 1D or 2D, got ndim={spectra.ndim}")
    return spectra


def _normalize_coefficients(coefficients: np.ndarray) -> np.ndarray:
    sums = coefficients.sum(axis=1, keepdims=True)
    return np.divide(
        coefficients,
        sums,
        out=np.zeros_like(coefficients, dtype=np.float32),
        where=sums > 0,
    )


def _compute_r2(spectra: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    centered = spectra - np.mean(spectra, axis=1, keepdims=True)
    ss_tot = np.sum(centered * centered, axis=1)
    residual = spectra - reconstructed
    ss_res = np.sum(residual * residual, axis=1)
    return np.divide(
        ss_tot - ss_res,
        ss_tot,
        out=np.zeros_like(ss_tot, dtype=np.float32),
        where=ss_tot > 0,
    )


def solve_single_spectrum(
    spectrum: np.ndarray,
    library: EndmemberLibrary,
    method: UnmixingMethod = "nnls",
) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=np.float32)
    if spectrum.ndim != 1:
        raise ValueError("solve_single_spectrum expects a 1D spectrum.")
    if spectrum.shape[0] != library.n_points:
        raise ValueError(f"Spectrum length {spectrum.shape[0]} does not match library length {library.n_points}.")

    matrix = np.asarray(library.matrix, dtype=np.float32)
    if method == "nnls":
        coefficients, _ = nnls(matrix, spectrum)
        return coefficients.astype(np.float32, copy=False)
    if method == "ols":
        coefficients, _, _, _ = np.linalg.lstsq(matrix, spectrum, rcond=None)
        return coefficients.astype(np.float32, copy=False)
    if method == "fcls":
        initial, _ = nnls(matrix, spectrum)
        initial_sum = float(initial.sum())
        if initial_sum > 0:
            x0 = initial / initial_sum
        else:
            x0 = np.full(matrix.shape[1], 1.0 / matrix.shape[1], dtype=np.float32)

        def objective(coefficients: np.ndarray) -> float:
            residual = matrix @ coefficients - spectrum
            return 0.5 * float(np.dot(residual, residual))

        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=[(0.0, None)] * matrix.shape[1],
            constraints=[{"type": "eq", "fun": lambda values: float(np.sum(values) - 1.0)}],
            options={"maxiter": 200, "ftol": 1e-9},
        )
        if result.success:
            return np.clip(result.x, 0.0, None).astype(np.float32, copy=False)
        return x0.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported method={method!r}. Expected 'ols', 'nnls', or 'fcls'.")


def unmix_spectra(
    spectra: np.ndarray,
    library: EndmemberLibrary,
    method: UnmixingMethod = "nnls",
) -> ClassicalUnmixingResult:
    spectra = _ensure_2d_spectra(spectra)
    if spectra.shape[1] != library.n_points:
        raise ValueError(f"Spectra length {spectra.shape[1]} does not match library length {library.n_points}.")

    coefficients = np.vstack([solve_single_spectrum(row, library=library, method=method) for row in spectra]).astype(np.float32)
    reconstructed = coefficients @ library.matrix.T
    residual = spectra - reconstructed
    residual_l2 = np.linalg.norm(residual, axis=1).astype(np.float32)
    residual_rmse = np.sqrt(np.mean(residual * residual, axis=1)).astype(np.float32)
    residual_r2 = _compute_r2(spectra, reconstructed).astype(np.float32)
    abundances = _normalize_coefficients(coefficients)
    return ClassicalUnmixingResult(
        component_names=library.names,
        method=method,
        coefficients=coefficients,
        abundances=abundances,
        reconstructed=reconstructed.astype(np.float32, copy=False),
        residual_l2=residual_l2,
        residual_rmse=residual_rmse,
        residual_r2=residual_r2,
    )


def blind_nmf_unmix_spectra(
    spectra: np.ndarray,
    n_components: int,
    random_state: int = 0,
    max_iter: int = 3000,
) -> BlindNMFResult:
    spectra = _ensure_2d_spectra(spectra)
    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    nonnegative_spectra = np.clip(spectra, 0.0, None)
    model = SklearnNMF(
        n_components=n_components,
        init="nndsvda",
        random_state=random_state,
        max_iter=max_iter,
    )
    coefficients = model.fit_transform(nonnegative_spectra).astype(np.float32, copy=False)
    reconstructed = model.inverse_transform(coefficients).astype(np.float32, copy=False)
    endmember_matrix = model.components_.T.astype(np.float32, copy=False)
    residual = nonnegative_spectra - reconstructed
    residual_l2 = np.linalg.norm(residual, axis=1).astype(np.float32)
    residual_rmse = np.sqrt(np.mean(residual * residual, axis=1)).astype(np.float32)
    residual_r2 = _compute_r2(nonnegative_spectra, reconstructed).astype(np.float32)
    abundances = _normalize_coefficients(coefficients)
    component_names = tuple(f"nmf_{index + 1}" for index in range(n_components))
    return BlindNMFResult(
        component_names=component_names,
        abundances=abundances,
        reconstructed=reconstructed,
        endmember_matrix=endmember_matrix,
        residual_l2=residual_l2,
        residual_rmse=residual_rmse,
        residual_r2=residual_r2,
    )


def align_blind_nmf_to_reference(
    result: BlindNMFResult,
    reference_library: EndmemberLibrary,
) -> tuple[BlindNMFResult, pd.DataFrame]:
    if result.endmember_matrix.shape[1] != reference_library.n_endmembers:
        raise ValueError(
            "Blind NMF component count does not match reference library size: "
            f"{result.endmember_matrix.shape[1]} vs {reference_library.n_endmembers}"
        )
    if result.endmember_matrix.shape[0] != reference_library.n_points:
        raise ValueError(
            "Blind NMF spectrum length does not match reference library length: "
            f"{result.endmember_matrix.shape[0]} vs {reference_library.n_points}"
        )

    blind = result.endmember_matrix.astype(np.float64, copy=False)
    reference = reference_library.matrix.astype(np.float64, copy=False)
    blind_norm = np.linalg.norm(blind, axis=0, keepdims=True)
    reference_norm = np.linalg.norm(reference, axis=0, keepdims=True)
    similarity = (blind.T @ reference) / np.maximum(blind_norm.T @ reference_norm, 1e-12)
    row_ind, col_ind = linear_sum_assignment(-similarity)

    order = [row for _, row in sorted(zip(col_ind.tolist(), row_ind.tolist()))]
    ordered_names = tuple(reference_library.names[index] for index in sorted(col_ind.tolist()))
    aligned = BlindNMFResult(
        component_names=ordered_names,
        abundances=result.abundances[:, order].astype(np.float32, copy=False),
        reconstructed=result.reconstructed,
        endmember_matrix=result.endmember_matrix[:, order].astype(np.float32, copy=False),
        residual_l2=result.residual_l2,
        residual_rmse=result.residual_rmse,
        residual_r2=result.residual_r2,
    )

    similarity_df = pd.DataFrame(
        similarity,
        index=result.component_names,
        columns=reference_library.names,
    )
    return aligned, similarity_df
