from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._common import _ensure_parent, _grid_from_frame


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
