from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization._common import _ensure_parent, _grid_from_frame


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
