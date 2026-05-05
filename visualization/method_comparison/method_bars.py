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
