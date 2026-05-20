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


@dataclass(frozen=True)
class PrismUnmixingResult:
    component_names: tuple[str, ...]
    method: str
    coefficients: np.ndarray
    abundances: np.ndarray
    reconstructed: np.ndarray
    residual_l2: np.ndarray
    residual_rmse: np.ndarray
    residual_r2: np.ndarray
    config: dict

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


def _endmember_pairwise_sam(matrix: np.ndarray) -> tuple[float, list[float]]:
    """Mean and per-pair SAM (rad) across all distinct endmember pairs."""
    A = matrix.astype(np.float64)
    n_endmembers = A.shape[1]
    pairs: list[float] = []
    for i in range(n_endmembers):
        for j in range(i + 1, n_endmembers):
            denom = float(np.linalg.norm(A[:, i]) * np.linalg.norm(A[:, j]))
            if denom < 1e-12:
                pairs.append(0.0)
                continue
            cos = float(np.clip(np.dot(A[:, i], A[:, j]) / denom, -1.0, 1.0))
            pairs.append(float(np.arccos(cos)))
    mean_sam = float(np.mean(pairs)) if pairs else 0.0
    return mean_sam, pairs


def _auto_select_weight_mode(matrix: np.ndarray, threshold_rad: float = 1.05) -> tuple[str, float]:
    """Pick uniform when mean pairwise SAM < threshold (conservative), else endmember_std.

    Note: 1.05 rad (~60 deg) is intentionally conservative — endmember_std requires both
    truly orthogonal endmembers AND clean fingerprint-difference bands; real Raman data
    typically fails the second condition due to baseline bias on endmember spectra,
    so auto defaults to uniform for typical Raman libraries (PE/PP/starch mean SAM ~0.85).
    """
    mean_sam, _ = _endmember_pairwise_sam(matrix)
    mode = "uniform" if mean_sam < threshold_rad else "endmember_std"
    return mode, mean_sam


def _prism_band_weights(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Per-band weight emphasising discriminative bands across endmembers."""
    if mode == "endmember_std":
        w = matrix.std(axis=1)
        w = w / (w.max() + 1e-12) + 0.1
        return w.astype(np.float32)
    if mode == "uniform":
        return np.ones(matrix.shape[0], dtype=np.float32)
    raise ValueError(f"Unknown weight_mode={mode!r}. Expected 'endmember_std', 'uniform', or 'auto'.")


def _solve_weighted_l2_nnls(
    spectra: np.ndarray,
    matrix: np.ndarray,
    weights_sqrt: np.ndarray,
    lambda_l2: float,
) -> np.ndarray:
    """Per-pixel min ||W^{1/2}(Aw - y)||^2 + lambda||w||^2 s.t. w >= 0."""
    n_endmembers = matrix.shape[1]
    a_weighted = matrix * weights_sqrt[:, None]
    sqrt_lambda = float(np.sqrt(max(lambda_l2, 0.0)))
    a_aug = np.vstack([a_weighted, sqrt_lambda * np.eye(n_endmembers, dtype=np.float32)]).astype(np.float64)
    coefs = np.zeros((spectra.shape[0], n_endmembers), dtype=np.float32)
    zeros_k = np.zeros(n_endmembers, dtype=np.float64)
    for i, y in enumerate(spectra):
        y_aug = np.concatenate([(y * weights_sqrt).astype(np.float64), zeros_k])
        w, _ = nnls(a_aug, y_aug)
        coefs[i] = w.astype(np.float32)
    return coefs


def _solve_weighted_l2_nnls_with_anchor(
    spectra: np.ndarray,
    matrix: np.ndarray,
    weights_sqrt: np.ndarray,
    lambda_l2: float,
    anchor: np.ndarray,
    lambda_anchor: float,
) -> np.ndarray:
    """Per-pixel min ||W^{1/2}(Aw - y)||^2 + l_L2||w||^2 + l_a||w - a||^2 s.t. w >= 0."""
    n_endmembers = matrix.shape[1]
    a_weighted = matrix * weights_sqrt[:, None]
    sqrt_lambda_l2 = float(np.sqrt(max(lambda_l2, 0.0)))
    sqrt_lambda_a = float(np.sqrt(max(lambda_anchor, 0.0)))
    a_aug = np.vstack(
        [
            a_weighted,
            sqrt_lambda_l2 * np.eye(n_endmembers, dtype=np.float32),
            sqrt_lambda_a * np.eye(n_endmembers, dtype=np.float32),
        ]
    ).astype(np.float64)
    coefs = np.zeros((spectra.shape[0], n_endmembers), dtype=np.float32)
    zeros_k = np.zeros(n_endmembers, dtype=np.float64)
    for i in range(spectra.shape[0]):
        y_w = (spectra[i] * weights_sqrt).astype(np.float64)
        y_aug = np.concatenate([y_w, zeros_k, sqrt_lambda_a * anchor[i].astype(np.float64)])
        w, _ = nnls(a_aug, y_aug)
        coefs[i] = w.astype(np.float32)
    return coefs


def prism_unmix_spectra(
    spectra: np.ndarray,
    library: EndmemberLibrary,
    *,
    image_shape: tuple[int, int] | None = None,
    lambda_l2: float = 1e-2,
    weight_mode: Literal["endmember_std", "uniform", "auto"] = "uniform",
    auto_sam_threshold: float = 1.05,
    lambda_tv: float = 0.10,
    tv_iters: int = 2,
    lambda_anchor_scale: float = 5.0,
) -> PrismUnmixingResult:
    """PRISM = weighted L2 NNLS + optional spatial TV-anchor iteration.

    Outputs:
        PrismUnmixingResult — row-normalised abundances and reconstructed spectra,
        with config dict capturing the hyperparameters used (including resolved
        weight_mode and endmember mean SAM when weight_mode='auto').
    """
    spectra = _ensure_2d_spectra(spectra)
    if spectra.shape[1] != library.n_points:
        raise ValueError(f"Spectra length {spectra.shape[1]} does not match library length {library.n_points}.")
    matrix = np.asarray(library.matrix, dtype=np.float32)
    n_pixels, _ = spectra.shape
    n_endmembers = matrix.shape[1]

    weight_mode_input = weight_mode
    endmember_mean_sam, _ = _endmember_pairwise_sam(matrix)
    if weight_mode == "auto":
        weight_mode, _ = _auto_select_weight_mode(matrix, threshold_rad=auto_sam_threshold)

    weights = _prism_band_weights(matrix, weight_mode)
    weights_sqrt = np.sqrt(weights).astype(np.float32)

    coefficients = _solve_weighted_l2_nnls(
        spectra=spectra,
        matrix=matrix,
        weights_sqrt=weights_sqrt,
        lambda_l2=lambda_l2,
    )

    if image_shape is not None and tv_iters > 0 and lambda_tv > 0:
        from skimage.restoration import denoise_tv_chambolle

        height, width = image_shape
        if height * width != n_pixels:
            raise ValueError(f"image_shape={image_shape} -> {height*width} pixels, expected {n_pixels}.")
        lambda_anchor = lambda_tv * lambda_anchor_scale
        for _ in range(tv_iters):
            coef_map = coefficients.reshape(height, width, n_endmembers).astype(np.float64, copy=False)
            anchor_map = np.empty_like(coef_map)
            for k in range(n_endmembers):
                anchor_map[..., k] = denoise_tv_chambolle(coef_map[..., k], weight=lambda_tv, channel_axis=None)
            anchor_flat = anchor_map.reshape(n_pixels, n_endmembers).astype(np.float32)
            coefficients = _solve_weighted_l2_nnls_with_anchor(
                spectra=spectra,
                matrix=matrix,
                weights_sqrt=weights_sqrt,
                lambda_l2=lambda_l2,
                anchor=anchor_flat,
                lambda_anchor=lambda_anchor,
            )

    reconstructed = (coefficients @ matrix.T).astype(np.float32)
    residual = spectra - reconstructed
    residual_l2 = np.linalg.norm(residual, axis=1).astype(np.float32)
    residual_rmse = np.sqrt(np.mean(residual * residual, axis=1)).astype(np.float32)
    residual_r2 = _compute_r2(spectra, reconstructed).astype(np.float32)
    abundances = _normalize_coefficients(coefficients)
    return PrismUnmixingResult(
        component_names=library.names,
        method="prism",
        coefficients=coefficients,
        abundances=abundances,
        reconstructed=reconstructed,
        residual_l2=residual_l2,
        residual_rmse=residual_rmse,
        residual_r2=residual_r2,
        config={
            "lambda_l2": float(lambda_l2),
            "weight_mode": weight_mode,
            "weight_mode_input": weight_mode_input,
            "endmember_mean_sam_rad": float(endmember_mean_sam),
            "auto_sam_threshold": float(auto_sam_threshold),
            "lambda_tv": float(lambda_tv),
            "tv_iters": int(tv_iters),
            "image_shape": image_shape,
            "lambda_anchor_scale": float(lambda_anchor_scale),
        },
    )


@dataclass(frozen=True)
class McrAlsResult:
    component_names: tuple[str, ...]
    method: str
    coefficients: np.ndarray
    abundances: np.ndarray
    reconstructed: np.ndarray
    residual_l2: np.ndarray
    residual_rmse: np.ndarray
    residual_r2: np.ndarray
    n_iter: int
    config: dict

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


def mcr_als_unmix_spectra(
    spectra: np.ndarray,
    library: EndmemberLibrary,
    *,
    fix_endmembers: bool = True,
    sum_to_one: bool = True,
    max_iter: int = 100,
    tol_increase: float = 0.0,
    tol_n_increase: int = 10,
    tol_err_change: float = 1e-10,
) -> McrAlsResult:
    """MCR-ALS unmixing with endmember library either hard-locked or used as init only.

    Outputs:
        McrAlsResult — row-normalised abundances + reconstructed spectra; n_iter is
        actual iterations run; config records fix_endmembers / sum_to_one / tolerances;
        when fix_endmembers=False, the optimised ST is exposed via config["st_opt"].
    """
    from pymcr.mcr import McrAR
    from pymcr.regressors import NNLS as PymcrNNLS
    from pymcr.constraints import ConstraintNonneg, ConstraintNorm

    spectra = _ensure_2d_spectra(spectra)
    if spectra.shape[1] != library.n_points:
        raise ValueError(f"Spectra length {spectra.shape[1]} does not match library length {library.n_points}.")

    matrix = np.asarray(library.matrix, dtype=np.float64)
    n_endmembers = matrix.shape[1]
    spectra_d = spectra.astype(np.float64)

    st_init = matrix.T.copy()
    c_init = np.zeros((spectra_d.shape[0], n_endmembers), dtype=np.float64)
    for i in range(spectra_d.shape[0]):
        c_init[i], _ = nnls(matrix, spectra_d[i])

    c_constraints = [ConstraintNonneg()]
    if sum_to_one:
        c_constraints.append(ConstraintNorm())

    mcrar = McrAR(
        max_iter=max_iter,
        st_regr=PymcrNNLS(),
        c_regr=PymcrNNLS(),
        c_constraints=c_constraints,
        st_constraints=[ConstraintNonneg()],
        tol_increase=tol_increase,
        tol_n_increase=tol_n_increase,
        tol_err_change=tol_err_change,
    )
    st_fix = list(range(n_endmembers)) if fix_endmembers else []
    mcrar.fit(
        spectra_d,
        C=c_init,
        ST=st_init,
        st_fix=st_fix,
        c_fix=[],
        c_first=True,
        verbose=False,
    )

    coefficients = mcrar.C_opt_.astype(np.float32)
    st_opt = mcrar.ST_opt_.astype(np.float32)
    reconstructed = (coefficients @ st_opt).astype(np.float32)
    residual = spectra - reconstructed
    residual_l2 = np.linalg.norm(residual, axis=1).astype(np.float32)
    residual_rmse = np.sqrt(np.mean(residual * residual, axis=1)).astype(np.float32)
    residual_r2 = _compute_r2(spectra, reconstructed).astype(np.float32)
    abundances = _normalize_coefficients(coefficients)
    return McrAlsResult(
        component_names=library.names,
        method="mcr_als" if fix_endmembers else "mcr_als_semi_blind",
        coefficients=coefficients,
        abundances=abundances,
        reconstructed=reconstructed,
        residual_l2=residual_l2,
        residual_rmse=residual_rmse,
        residual_r2=residual_r2,
        n_iter=int(getattr(mcrar, "n_iter", -1)),
        config={
            "fix_endmembers": bool(fix_endmembers),
            "sum_to_one": bool(sum_to_one),
            "max_iter": int(max_iter),
            "tol_increase": float(tol_increase),
            "tol_n_increase": int(tol_n_increase),
            "tol_err_change": float(tol_err_change),
            "st_fix": "all" if fix_endmembers else "none",
            "c_first": True,
            "st_opt_drift_sam_rad": _endmember_drift_sam(matrix.T, st_opt),
        },
    )


def _endmember_drift_sam(st_init: np.ndarray, st_opt: np.ndarray) -> list[float]:
    """Per-endmember SAM (rad) between init and optimised ST; 0 = locked."""
    drifts: list[float] = []
    for k in range(st_init.shape[0]):
        a = st_init[k].astype(np.float64)
        b = st_opt[k].astype(np.float64)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-12:
            drifts.append(0.0)
            continue
        cos = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
        drifts.append(float(np.arccos(cos)))
    return drifts
