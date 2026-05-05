from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _coordinate_frame(df: pd.DataFrame) -> pd.DataFrame:
    coord_df = df.copy()
    if "x_idx" in coord_df.columns and "y_idx" in coord_df.columns:
        coord_df = coord_df.dropna(subset=["x_idx", "y_idx"]).copy()
        coord_df["x_idx"] = coord_df["x_idx"].astype(int)
        coord_df["y_idx"] = coord_df["y_idx"].astype(int)
        return coord_df

    if "relative_path" not in coord_df.columns:
        return coord_df.iloc[0:0].copy()

    x_values: list[int | None] = []
    y_values: list[int | None] = []
    for rel in coord_df["relative_path"]:
        name = Path(str(rel)).name
        match_x = re.search(r"-X(\d+)-", name)
        match_y = re.search(r"-Y(\d+)-", name)
        x_values.append(int(match_x.group(1)) if match_x else None)
        y_values.append(int(match_y.group(1)) if match_y else None)
    coord_df["x_idx"] = x_values
    coord_df["y_idx"] = y_values
    coord_df = coord_df.dropna(subset=["x_idx", "y_idx"]).copy()
    coord_df["x_idx"] = coord_df["x_idx"].astype(int)
    coord_df["y_idx"] = coord_df["y_idx"].astype(int)
    return coord_df


def _grid_from_frame(df: pd.DataFrame, value_col: str) -> np.ndarray | None:
    coord_df = _coordinate_frame(df)
    if coord_df.empty or value_col not in coord_df.columns:
        return None
    grid = coord_df.pivot_table(index="y_idx", columns="x_idx", values=value_col, aggfunc="mean")
    grid = grid.sort_index(ascending=False)
    return grid.to_numpy(dtype=float)


def plot_abundance_maps(
    df: pd.DataFrame,
    component_names: tuple[str, ...] | list[str],
    output_path: Path,
    title: str,
) -> None:
    grids: list[np.ndarray] = []
    labels: list[str] = []
    for name in component_names:
        grid = _grid_from_frame(df, f"abundance_{name}")
        if grid is not None:
            grids.append(grid)
            labels.append(name)
    if not grids:
        return

    _ensure_parent(output_path)
    fig, axes = plt.subplots(1, len(grids), figsize=(4.5 * len(grids), 4.2), squeeze=False)
    for ax, label, grid in zip(axes[0], labels, grids):
        image = ax.imshow(grid, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"{label} abundance")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_residual_map(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    value_col: str = "residual_rmse",
) -> None:
    grid = _grid_from_frame(df, value_col)
    if grid is None:
        return
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(grid, cmap="inferno", vmin=0.0)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=value_col)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_reconstruction_examples(
    axis: np.ndarray,
    spectra: np.ndarray,
    reconstructed: np.ndarray,
    output_path: Path,
    labels: list[str] | None = None,
    max_examples: int = 6,
) -> None:
    if spectra.size == 0 or reconstructed.size == 0:
        return
    n_examples = min(max_examples, spectra.shape[0])
    if n_examples <= 0:
        return

    _ensure_parent(output_path)
    picks = np.linspace(0, spectra.shape[0] - 1, n_examples, dtype=int)
    fig, axes = plt.subplots(n_examples, 1, figsize=(10, 2.2 * n_examples), sharex=True)
    if n_examples == 1:
        axes = [axes]
    for ax, idx in zip(axes, picks):
        ax.plot(axis, spectra[idx], label="input", linewidth=1.1)
        ax.plot(axis, reconstructed[idx], label="reconstruction", linewidth=1.1, alpha=0.9)
        ax.grid(alpha=0.25)
        title = labels[idx] if labels is not None and idx < len(labels) else f"spectrum_{idx}"
        title = str(title).encode("ascii", errors="ignore").decode("ascii").strip() or f"spectrum_{idx}"
        ax.set_title(str(title))
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Raman Shift (cm^-1)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


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


def plot_single_spectrum_preprocessing(
    axis: np.ndarray,
    raw: np.ndarray,
    corrected: np.ndarray,
    normalized: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    _ensure_parent(output_path)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(axis, raw, color="#355070", linewidth=1.1)
    axes[0].set_title(f"{title} - raw")
    axes[0].grid(alpha=0.25)
    axes[1].plot(axis, corrected, color="#b56576", linewidth=1.1)
    axes[1].set_title("after baseline correction")
    axes[1].grid(alpha=0.25)
    axes[2].plot(axis, normalized, color="#6d597a", linewidth=1.1)
    axes[2].set_title("after normalization")
    axes[2].grid(alpha=0.25)
    axes[2].set_xlabel("Raman Shift (cm^-1)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_protocol_spectrum_triptych(
    axis: np.ndarray,
    protocol_curves: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    title: str,
) -> None:
    if not protocol_curves:
        return
    _ensure_parent(output_path)
    stages = [("raw", "raw"), ("corrected", "corrected"), ("normalized", "normalized")]
    protocols = list(protocol_curves.keys())
    fig, axes = plt.subplots(len(protocols), len(stages), figsize=(14, 3.8 * len(protocols)), sharex=True)
    if len(protocols) == 1:
        axes = np.asarray([axes])
    for row_idx, protocol in enumerate(protocols):
        curves = protocol_curves[protocol]
        for col_idx, (key, stage_title) in enumerate(stages):
            ax = axes[row_idx, col_idx]
            ax.plot(axis, curves[key], linewidth=1.0)
            if row_idx == 0:
                ax.set_title(stage_title)
            if col_idx == 0:
                ax.set_ylabel(protocol)
            ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Raman Shift (cm^-1)")
    axes[-1, 1].set_xlabel("Raman Shift (cm^-1)")
    axes[-1, 2].set_xlabel("Raman Shift (cm^-1)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_protocol_abundance_grid(
    protocol_prediction_frames: dict[str, pd.DataFrame],
    component_names: tuple[str, ...] | list[str],
    output_path: Path,
    title: str,
) -> None:
    protocols = list(protocol_prediction_frames.keys())
    components = [name for name in component_names]
    if not protocols or not components:
        return
    _ensure_parent(output_path)
    fig, axes = plt.subplots(len(protocols), len(components), figsize=(4.3 * len(components), 3.9 * len(protocols)), squeeze=False)
    for row_idx, protocol in enumerate(protocols):
        df = protocol_prediction_frames[protocol]
        for col_idx, component in enumerate(components):
            ax = axes[row_idx, col_idx]
            grid = _grid_from_frame(df, f"abundance_{component}")
            if grid is None:
                ax.axis("off")
                continue
            image = ax.imshow(grid, cmap="magma", vmin=0.0, vmax=1.0)
            if row_idx == 0:
                ax.set_title(component)
            if col_idx == 0:
                ax.set_ylabel(protocol)
            ax.set_xlabel("X")
            ax.set_ylabel(protocol if col_idx == 0 else "Y")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
