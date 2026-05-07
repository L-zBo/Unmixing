"""Endmember fingerprint-peak overlay plot.

Plots the three endmember (PE / PP / starch) pure spectra after preprocessing,
with literature-known fingerprint peaks annotated. Annotations sit in a clear
band above the data and adjacent labels (within 90 cm-1 on the x-axis) are
staggered into multiple rows so they never overlap. Each label is connected to
its peak by a thin leader line.

Peak positions:
- PE 1062 / 1130 / 1295 / 1440 cm-1 (Sage 2021)
- PP 808 / 841 / 1330 cm-1 (RSC Analyst 2024)
- starch 478 / 1124 cm-1 (Raman of glucan backbone, common literature)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization._common import _ensure_parent


_DEFAULT_PEAK_COLORS = {
    "PE": "#1f77b4",
    "PP": "#ff7f0e",
    "starch": "#2ca02c",
    "PE+PP": "#9467bd",
}

_LABEL_STAGGER_THRESHOLD_CM1 = 90.0
_N_LABEL_ROWS = 3


def plot_endmember_fingerprints(
    *,
    axis: np.ndarray,
    endmember_matrix: np.ndarray,
    component_names: tuple[str, ...] | list[str],
    peaks_by_component: dict[str, tuple[int, ...] | list[int]],
    output_path: Path,
    title: str,
    protocol_name: str = "als_l2",
) -> None:
    """Overlay endmember pure spectra with annotated fingerprint peaks.

    Parameters
    ----------
    axis: shape (n_points,) wavenumber axis (cm-1).
    endmember_matrix: shape (n_points, n_components) preprocessed endmembers.
    component_names: ordered names matching the columns of ``endmember_matrix``.
    peaks_by_component: mapping ``component_name -> tuple of peak wavenumbers``.
    """
    if endmember_matrix.ndim != 2:
        raise ValueError(f"endmember_matrix must be 2D, got ndim={endmember_matrix.ndim}")
    if endmember_matrix.shape[1] != len(component_names):
        raise ValueError(
            f"endmember_matrix has {endmember_matrix.shape[1]} columns "
            f"but {len(component_names)} component names"
        )

    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    y_max = float(np.max(endmember_matrix)) if endmember_matrix.size else 1.0
    if y_max <= 0:
        y_max = 1.0

    for c_index, name in enumerate(component_names):
        spectrum = endmember_matrix[:, c_index]
        color = _DEFAULT_PEAK_COLORS.get(str(name), None)
        ax.plot(axis, spectrum, label=str(name), linewidth=1.6, color=color, alpha=0.95)

    annotations: list[tuple[int, float, float, str | None]] = []
    for c_index, name in enumerate(component_names):
        spectrum = endmember_matrix[:, c_index]
        color = _DEFAULT_PEAK_COLORS.get(str(name), None)
        for peak_cm1 in peaks_by_component.get(str(name), ()):
            nearest_index = int(np.argmin(np.abs(axis - float(peak_cm1))))
            x_value = float(axis[nearest_index])
            y_value = float(spectrum[nearest_index])
            ax.scatter(
                [x_value],
                [y_value],
                marker="v",
                s=44,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
            )
            annotations.append((int(peak_cm1), x_value, y_value, color))

    annotations.sort(key=lambda item: item[1])
    last_x_per_row: dict[int, float] = {}
    chosen_rows: list[int] = []
    for _peak_cm1, x_value, _y, _c in annotations:
        chosen = -1
        for row in range(_N_LABEL_ROWS):
            previous_x = last_x_per_row.get(row)
            if previous_x is None or (x_value - previous_x) >= _LABEL_STAGGER_THRESHOLD_CM1:
                chosen = row
                break
        if chosen < 0:
            chosen = (chosen_rows[-1] + 1) % _N_LABEL_ROWS if chosen_rows else 0
        chosen_rows.append(chosen)
        last_x_per_row[chosen] = x_value

    label_band_base = y_max * 1.18
    label_row_step = y_max * 0.10
    for (peak_cm1, x_value, y_value, color), row in zip(annotations, chosen_rows):
        label_y = label_band_base + row * label_row_step
        ax.annotate(
            f"{peak_cm1}",
            xy=(x_value, y_value),
            xytext=(x_value, label_y),
            fontsize=9,
            ha="center",
            color=color if color is not None else "black",
            arrowprops=dict(
                arrowstyle="-",
                linewidth=0.7,
                color=color if color is not None else "gray",
                alpha=0.55,
                shrinkA=0,
                shrinkB=2,
            ),
        )

    top_used_row = max(chosen_rows) if chosen_rows else 0
    ax.set_ylim(top=label_band_base + (top_used_row + 1) * label_row_step)

    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel(f"Intensity ({protocol_name} normalized)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
