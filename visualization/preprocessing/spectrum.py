from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._common import _ensure_parent, _grid_from_frame


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
            ax.set_xlabel("X")
            ax.set_ylabel(protocol if col_idx == 0 else "Y")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
