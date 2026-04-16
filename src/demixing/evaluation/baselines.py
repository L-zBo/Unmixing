from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls


@dataclass(frozen=True)
class BaselineResult:
    predictions: pd.DataFrame


def score_to_label(score: float) -> int:
    if score < 0.4:
        return 0
    if score < 0.75:
        return 1
    return 2


def run_anchor_nnls_baseline(
    prediction_df_template: pd.DataFrame,
    traces: dict[str, list[list[float]]],
    anchor_matrix: np.ndarray,
) -> BaselineResult:
    anchor_matrix = np.asarray(anchor_matrix, dtype=float)
    A = anchor_matrix.T
    rows: list[dict[str, object]] = []
    for idx, row in prediction_df_template.reset_index(drop=True).iterrows():
        x = np.asarray(traces["x"][idx], dtype=float)
        coeffs, _ = nnls(A, x)
        coeff_sum = coeffs.sum()
        abundances = coeffs / coeff_sum if coeff_sum > 0 else np.zeros_like(coeffs)
        mask = np.asarray([float(v) for v in row["microplastic_mask"].split(",")], dtype=float) if isinstance(row.get("microplastic_mask"), str) else None
        if mask is None:
            if row["family"] == "pp_starch":
                mask = np.array([1.0, 0.0, 0.0], dtype=float)
            elif row["family"] == "pe_starch":
                mask = np.array([0.0, 1.0, 0.0], dtype=float)
            elif row["family"] == "pp_pe_starch":
                mask = np.array([1.0, 1.0, 0.0], dtype=float)
            else:
                mask = np.array([0.0, 0.0, 0.0], dtype=float)
        microplastic_score = float(np.sum(abundances[:3] * mask))
        out = {
            "relative_path": row["relative_path"],
            "quality_tier": row["quality_tier"],
            "family": row["family"],
            "source_kind": row["source_kind"],
            "label": int(row["label"]),
            "pred_label": score_to_label(microplastic_score),
            "weight": float(row["weight"]),
            "weak_label_available": int(row["weak_label_available"]),
            "microplastic_score": microplastic_score,
        }
        for j, value in enumerate(abundances[:3]):
            out[f"abundance_{j + 1}"] = float(value)
        rows.append(out)
    return BaselineResult(predictions=pd.DataFrame(rows))
