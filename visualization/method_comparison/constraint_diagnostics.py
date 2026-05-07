"""Constraint-diagnostic plots for OLS/NNLS/FCLS/NMF.

Two plotting helpers:
- ``plot_negative_abundance_pct_bars``: per-scenario fraction of pixels with at
  least one negative coefficient. OLS is expected to spike on matched-endmember
  泛化 scenarios where NNLS/FCLS stay at 0 by construction.
- ``plot_nmf_endmember_sam_bars``: spectral-angle drift (rad) between aligned
  NMF endmember and the physical reference spectrum. Large SAM = NMF learned a
  non-physical endmember.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._common import _ensure_parent


def _main_first_sort_key(label: str) -> tuple[int, str]:
    """Sort labels so ``main_*`` scenarios appear before ``gen_*``.

    Within each group keeps alphabetical order for stable layout.
    """
    return (0, str(label)) if str(label).startswith("main_") else (1, str(label))


def plot_negative_abundance_pct_bars(
    constraint_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Bar plot of per-scenario worst-component negative-coefficient fraction.

    Parameters
    ----------
    constraint_df: DataFrame with columns ``label``, ``method``, ``component``,
        ``neg_coef_fraction``.
    """
    if constraint_df.empty or "method" not in constraint_df.columns:
        return
    pivot = constraint_df.pivot_table(
        index="label",
        columns="method",
        values="neg_coef_fraction",
        aggfunc="max",
    )
    if pivot.empty:
        return
    method_order = [m for m in ("ols", "nnls", "fcls") if m in pivot.columns]
    if not method_order:
        return
    pivot = pivot[method_order]
    pivot = pivot.reindex(sorted(pivot.index, key=_main_first_sort_key))

    _ensure_parent(output_path)
    labels = pivot.index.tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.8 / max(1, len(method_order))
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 4.0, 5.2))
    for idx, method in enumerate(method_order):
        offsets = x - 0.4 + width / 2.0 + idx * width
        ax.bar(offsets, pivot[method].values * 100.0, width=width, label=method.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Negative-coef pixel fraction (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_nmf_endmember_sam_bars(
    nmf_drift_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Bar plot of NMF endmember SAM drift across scenarios, grouped by component.

    Parameters
    ----------
    nmf_drift_df: DataFrame with columns ``label``, ``component``, ``sam_rad``.
    """
    if nmf_drift_df.empty or "component" not in nmf_drift_df.columns:
        return
    pivot = nmf_drift_df.pivot_table(
        index="label",
        columns="component",
        values="sam_rad",
        aggfunc="mean",
    )
    if pivot.empty:
        return
    pivot = pivot.reindex(sorted(pivot.index, key=_main_first_sort_key))

    _ensure_parent(output_path)
    labels = pivot.index.tolist()
    components = pivot.columns.tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.8 / max(1, len(components))
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 4.0, 5.2))
    for idx, component in enumerate(components):
        offsets = x - 0.4 + width / 2.0 + idx * width
        ax.bar(offsets, pivot[component].values, width=width, label=str(component))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("SAM (rad)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
