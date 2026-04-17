from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


plt.rcParams["figure.dpi"] = 140


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_loss_curve(history: list[dict[str, float]], output_path: Path) -> None:
    ensure_parent(output_path)
    epochs = [item["epoch"] for item in history]
    losses = [item["loss"] for item in history]
    val_losses = [item.get("val_loss") for item in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, marker="o", linewidth=2, label="train")
    if any(value is not None for value in val_losses):
        plt.plot(epochs, val_losses, marker="s", linewidth=2, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_endmembers(axis: np.ndarray, endmembers: np.ndarray, anchors: np.ndarray | None, output_path: Path) -> None:
    ensure_parent(output_path)
    plt.figure(figsize=(10, 6))
    for idx, curve in enumerate(endmembers[:3]):
        plt.plot(axis, curve, linewidth=2, label=f"learned_endmember_{idx + 1}")
    if anchors is not None and anchors.size > 0:
        for idx, curve in enumerate(anchors[:3]):
            plt.plot(axis, curve, linestyle="--", alpha=0.8, label=f"anchor_{idx + 1}")
    plt.xlabel("Raman Shift (cm^-1)")
    plt.ylabel("Intensity")
    plt.title("Learned Endmembers vs Anchors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_reconstruction_examples(
    axis: np.ndarray,
    traces: dict[str, list[list[float]]],
    df: pd.DataFrame,
    output_path: Path,
    max_examples: int = 6,
) -> None:
    ensure_parent(output_path)
    average_df = df[df["source_kind"] == "average"].copy()
    if average_df.empty:
        average_df = df.copy()
    average_df = average_df.sort_values(["label", "family", "relative_path"]).head(max_examples)
    if average_df.empty:
        return

    n_rows = len(average_df)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, average_df.iterrows()):
        idx = row.name
        x = np.asarray(traces["x"][idx], dtype=float)
        recon = np.asarray(traces["reconstruction"][idx], dtype=float)
        ax.plot(axis, x, label="input", linewidth=1.5)
        ax.plot(axis, recon, label="reconstruction", linewidth=1.5, alpha=0.8)
        ax.set_title(f"{row['family']} | label={row['label']} | sample_{idx}")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Raman Shift (cm^-1)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_average_abundance(df: pd.DataFrame, output_path: Path, max_bars: int = 40) -> None:
    ensure_parent(output_path)
    average_df = df[df["source_kind"] == "average"].copy()
    if average_df.empty:
        return
    abundance_cols = [col for col in df.columns if col.startswith("abundance_")]
    average_df = average_df.sort_values(["family", "label", "relative_path"]).head(max_bars)
    x = np.arange(len(average_df))
    bottom = np.zeros(len(average_df))
    plt.figure(figsize=(14, 5))
    for col in abundance_cols[:3]:
        values = average_df[col].to_numpy()
        plt.bar(x, values, bottom=bottom, label=col)
        bottom += values
    plt.xlabel("Average spectra samples")
    plt.ylabel("Estimated abundance")
    plt.title("Average-spectra abundance composition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_microplastic_score_boxplot(df: pd.DataFrame, output_path: Path) -> None:
    ensure_parent(output_path)
    labeled = df[df["label"] >= 0].copy()
    if labeled.empty:
        return
    groups = []
    labels = []
    for level, level_label in [(0, "low"), (1, "medium"), (2, "high")]:
        subset = labeled[labeled["label"] == level]["microplastic_score"].to_numpy()
        if subset.size == 0:
            continue
        groups.append(subset)
        labels.append(level_label)
    if not groups:
        return
    plt.figure(figsize=(6, 4))
    plt.boxplot(groups, tick_labels=labels, showfliers=False)
    plt.ylabel("Estimated microplastic score")
    plt.title("Microplastic score by weak label")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_model_vs_baseline_scores(
    model_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    output_path: Path,
) -> None:
    ensure_parent(output_path)
    labels = ["low", "medium", "high"]
    x = np.arange(len(labels))
    width = 0.35

    def means(df: pd.DataFrame) -> list[float]:
        out = []
        for label in [0, 1, 2]:
            subset = df[df["label"] == label]["microplastic_score"]
            out.append(float(subset.mean()) if not subset.empty else 0.0)
        return out

    model_means = means(model_df)
    baseline_means = means(baseline_df)

    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, model_means, width=width, label="model")
    plt.bar(x + width / 2, baseline_means, width=width, label="baseline")
    plt.xticks(x, labels)
    plt.ylabel("Mean microplastic score")
    plt.title("Model vs baseline by weak label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_family_grouped_scores(df: pd.DataFrame, output_path: Path) -> None:
    ensure_parent(output_path)
    average_df = df[df["source_kind"] == "average"].copy()
    if average_df.empty:
        return
    summary = average_df.groupby("family")["microplastic_score"].mean().sort_index()
    plt.figure(figsize=(7, 4))
    plt.bar(summary.index, summary.values)
    plt.ylabel("Mean microplastic score")
    plt.title("Average microplastic score by family")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_accuracy_comparison(
    model_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    output_path: Path,
) -> None:
    ensure_parent(output_path)
    families = sorted(set(model_df["family"]).intersection(set(baseline_df["family"])))
    model_acc = []
    baseline_acc = []
    for family in families:
        model_subset = model_df[model_df["family"] == family]
        baseline_subset = baseline_df[baseline_df["family"] == family]
        model_acc.append(float((model_subset["label"] == model_subset["pred_label"]).mean()) if not model_subset.empty else 0.0)
        baseline_acc.append(float((baseline_subset["label"] == baseline_subset["pred_label"]).mean()) if not baseline_subset.empty else 0.0)

    x = np.arange(len(families))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, model_acc, width=width, label="model")
    plt.bar(x + width / 2, baseline_acc, width=width, label="baseline")
    plt.xticks(x, families, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Validation accuracy by family")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_family_accuracy(
    family_accuracy: dict[str, float],
    output_path: Path,
    title: str = "Family accuracy",
) -> None:
    ensure_parent(output_path)
    if not family_accuracy:
        return
    items = sorted(family_accuracy.items())
    labels = [key for key, _ in items]
    values = [value for _, value in items]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    ensure_parent(output_path)
    if df.empty:
        return
    labels = [0, 1, 2]
    cm = confusion_matrix(df["label"], df["pred_label"], labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(["low", "medium", "high"])
    ax.set_yticklabels(["low", "medium", "high"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_prediction_map(df: pd.DataFrame, output_path: Path, title: str = "Prediction Map") -> None:
    ensure_parent(output_path)
    if df.empty:
        return
    coords = []
    for rel in df["relative_path"]:
        name = Path(rel).name
        match_x = re.search(r"-X(\d+)-", name)
        match_y = re.search(r"-Y(\d+)-", name)
        if match_x is None or match_y is None:
            coords.append((None, None))
        else:
            coords.append((int(match_x.group(1)), int(match_y.group(1))))
    coord_df = df.copy()
    coord_df["x_idx"] = [x for x, _ in coords]
    coord_df["y_idx"] = [y for _, y in coords]
    coord_df = coord_df.dropna(subset=["x_idx", "y_idx"]).copy()
    if coord_df.empty:
        return
    coord_df["x_idx"] = coord_df["x_idx"].astype(int)
    coord_df["y_idx"] = coord_df["y_idx"].astype(int)
    grid = coord_df.pivot_table(index="y_idx", columns="x_idx", values="pred_label", aggfunc="first")
    grid = grid.sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid.to_numpy(), cmap="viridis", vmin=0, vmax=2)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["low", "medium", "high"])
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_experiment_summary(summary: dict[str, object], output_path: Path) -> None:
    ensure_parent(output_path)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
