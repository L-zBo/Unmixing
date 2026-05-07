"""Protocol-comparison plots focused on consistency and fingerprint retention.

Two helpers paired with ``run_protocol_consistency_analysis``:
- ``plot_protocol_cv_bars``: per-component coefficient-of-variation across pixels,
  grouped by protocol. Lower CV = more consistent abundance recovery, which
  PPT-supports the "ALS+L2 is more stable than ALS+max / none+L2" claim.
- ``plot_fingerprint_retention_bars``: per-protocol relative retention of PE /
  PP / starch fingerprint-peak intensities (max in a 30 cm-1 window around the
  literature peak, normalized by spectrum L2). PPT-supports the physical
  interpretability claim of L2 normalization.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._common import _ensure_parent


_PROTOCOL_ORDER = ("als_l2", "als_max", "none_l2")


def plot_protocol_cv_bars(
    cv_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Bar plot of per (sample, component) abundance CV grouped by protocol.

    Parameters
    ----------
    cv_df: DataFrame with ``label`` (e.g. "PE+淀粉|PE"), ``protocol``, ``cv``
        columns. CV is unitless (std/mean).
    """
    if cv_df.empty or "protocol" not in cv_df.columns:
        return
    pivot = cv_df.pivot_table(index="label", columns="protocol", values="cv", aggfunc="mean")
    if pivot.empty:
        return
    protocol_order = [p for p in _PROTOCOL_ORDER if p in pivot.columns]
    if not protocol_order:
        return
    pivot = pivot[protocol_order]

    _ensure_parent(output_path)
    labels = pivot.index.tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.8 / max(1, len(protocol_order))
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 4.0, 5.0))
    for idx, protocol in enumerate(protocol_order):
        offsets = x - 0.4 + width / 2.0 + idx * width
        ax.bar(offsets, pivot[protocol].values, width=width, label=protocol)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Abundance CV (std/mean) across pixels — lower = more consistent")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_fingerprint_retention_bars(
    retention_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Bar plot of per-protocol fingerprint-peak retention by component peak.

    Parameters
    ----------
    retention_df: DataFrame with ``component`` (PE/PP/starch), ``peak_cm1``
        (literature wavenumber), ``protocol``, ``relative_intensity_mean``.
    """
    if retention_df.empty or "protocol" not in retention_df.columns:
        return
    df = retention_df.copy()
    df["peak_label"] = df["component"].astype(str) + "@" + df["peak_cm1"].astype(int).astype(str)
    pivot = df.pivot_table(index="peak_label", columns="protocol", values="relative_intensity_mean", aggfunc="mean")
    if pivot.empty:
        return
    protocol_order = [p for p in _PROTOCOL_ORDER if p in pivot.columns]
    if not protocol_order:
        return
    pivot = pivot[protocol_order]

    _ensure_parent(output_path)
    labels = pivot.index.tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.8 / max(1, len(protocol_order))
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 4.0, 5.0))
    for idx, protocol in enumerate(protocol_order):
        offsets = x - 0.4 + width / 2.0 + idx * width
        ax.bar(offsets, pivot[protocol].values, width=width, label=protocol)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean relative peak intensity (normalized spectrum)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_protocol_reconstruction_r2_bars(
    r2_df,
    output_path,
    title: str,
) -> None:
    """Bar plot of NNLS reconstruction R^2 by protocol on each sample.

    This is the primary "ALS necessity" indicator: protocols with ALS baseline
    correction (als_l2 / als_max) hold R^2 around 0.91, while none_l2 drops to
    ~0.77 — the cleanest single-figure justification for the ALS step.

    Parameters
    ----------
    r2_df: DataFrame with ``sample_label``, ``protocol``, ``mean_residual_r2``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if r2_df.empty or "protocol" not in r2_df.columns:
        return
    pivot = r2_df.pivot_table(
        index="sample_label",
        columns="protocol",
        values="mean_residual_r2",
        aggfunc="mean",
    )
    if pivot.empty:
        return
    protocol_order = [p for p in _PROTOCOL_ORDER if p in pivot.columns]
    if not protocol_order:
        return
    pivot = pivot[protocol_order]

    _ensure_parent(output_path)
    labels = pivot.index.tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.8 / max(1, len(protocol_order))
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 4.0, 5.2))
    for idx, protocol in enumerate(protocol_order):
        offsets = x - 0.4 + width / 2.0 + idx * width
        ax.bar(offsets, pivot[protocol].values, width=width, label=protocol)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("NNLS reconstruction R^2 (higher = better fit to observed spectrum)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
