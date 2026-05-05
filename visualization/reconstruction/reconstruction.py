from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization._common import _ensure_parent


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
