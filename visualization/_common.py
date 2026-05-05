"""Shared helpers for visualization modules.

Internal utilities reused across plot subdirectories:
- _ensure_parent: create output directory if missing
- _coordinate_frame: extract x_idx/y_idx from a prediction frame
- _grid_from_frame: build a 2D grid from a prediction frame for imshow
"""
from __future__ import annotations

import re
from pathlib import Path

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
