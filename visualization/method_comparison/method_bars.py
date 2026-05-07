from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._common import _ensure_parent


def plot_method_metric_bars(
    summary_df: pd.DataFrame,
    metric_cols: list[str] | tuple[str, ...],
    output_path: Path,
    title: str,
) -> None:
    if summary_df.empty or "method" not in summary_df.columns:
        return
    available_cols = [col for col in metric_cols if col in summary_df.columns]
    if not available_cols:
        return

    _ensure_parent(output_path)
    methods = summary_df["method"].astype(str).tolist()
    x = np.arange(len(methods), dtype=float)
    width = 0.8 / len(available_cols)
    fig, ax = plt.subplots(figsize=(1.8 * len(methods) + 3.0, 5.0))
    for idx, col in enumerate(available_cols):
        offsets = x - 0.4 + width / 2.0 + idx * width
        ax.bar(offsets, summary_df[col].astype(float), width=width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_method_abundance_bars(
    summary_df: pd.DataFrame,
    component_names: list[str] | tuple[str, ...],
    output_path: Path,
    title: str,
) -> None:
    abundance_cols = [f"mean_abundance_{name}" for name in component_names if f"mean_abundance_{name}" in summary_df.columns]
    if summary_df.empty or "method" not in summary_df.columns or not abundance_cols:
        return

    _ensure_parent(output_path)
    methods = summary_df["method"].astype(str).tolist()
    x = np.arange(len(methods), dtype=float)
    width = 0.8 / len(abundance_cols)
    fig, ax = plt.subplots(figsize=(1.8 * len(methods) + 3.0, 5.0))
    for idx, col in enumerate(abundance_cols):
        offsets = x - 0.4 + width / 2.0 + idx * width
        ax.bar(offsets, summary_df[col].astype(float), width=width, label=col.replace("mean_abundance_", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_synthetic_metric_subplots(
    summary_df: pd.DataFrame,
    metric_cols: list[str] | tuple[str, ...],
    output_path: Path,
    title: str,
    metric_labels: list[str] | tuple[str, ...] | None = None,
    metric_directions: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Multi-panel bar chart, one panel per metric, methods on x-axis.

    Drop-in companion to ``plot_method_metric_bars`` for the case where metrics
    live on different scales (e.g. RMSE in [0, 0.3] vs R^2 in [0, 1]) and a
    shared y-axis would compress one of them.

    metric_directions: per-metric flag, one of ``"lower"`` (lower-is-better)
    or ``"higher"`` (higher-is-better); appended as ↓/↑ to each panel title.
    """
    if summary_df.empty or "method" not in summary_df.columns:
        return
    pairs = [
        (idx, col)
        for idx, col in enumerate(metric_cols)
        if col in summary_df.columns
    ]
    if not pairs:
        return
    available = [col for _, col in pairs]
    src_indices = [idx for idx, _ in pairs]
    labels = (
        [metric_labels[i] for i in src_indices]
        if metric_labels is not None
        else list(available)
    )
    directions = (
        [metric_directions[i] for i in src_indices]
        if metric_directions is not None
        else ["?"] * len(available)
    )

    _ensure_parent(output_path)
    methods = summary_df["method"].astype(str).tolist()
    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels + 1.0, 4.5))
    if n_panels == 1:
        axes = [axes]
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    colors = palette[: len(methods)]
    for ax, col, label, direction in zip(axes, available, labels, directions):
        values = summary_df[col].astype(float).to_numpy()
        bars = ax.bar(methods, values, color=colors)
        suffix = " ↓" if direction == "lower" else " ↑" if direction == "higher" else ""
        ax.set_title(f"{label}{suffix}")
        ax.grid(axis="y", alpha=0.25)
        ymax = float(np.nanmax(values)) if values.size else 1.0
        ymin = float(np.nanmin(values)) if values.size else 0.0
        if ymin >= 0 and ymax > 0:
            ax.set_ylim(0.0, ymax * 1.18)
        for bar, v in zip(bars, values):
            offset = (ymax - ymin) * 0.02 if ymax > ymin else 0.01
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + offset,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
