"""Plot PRISM tv_iters convergence/trade-off curves from prism_param_sweep results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    csv_path = ROOT / "outputs/experiments/prism_param_sweep/prism_param_sweep_full.csv"
    out_dir = ROOT / "outputs/showcase/prism_convergence"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[df["weight_mode"] == "endmember_std"].copy()
    df = df[df["lambda_l2"] == 1e-4]

    datasets = [("formal_v1_als_l2", "NOISY 40×40"),
                ("formal_v1_clean_als_l2", "CLEAN 40×40")]
    metrics = [("mae", "MAE ↓", "lower better"),
               ("pearson_r", "Pearson r ↑", "higher better"),
               ("spatial_tv", "Spatial TV ↓", "lower better")]

    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(11, 9), sharex=True)

    for col, (ds, ds_label) in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        lambda_tvs = sorted(sub["lambda_tv"].unique())
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(lambda_tvs)))
        for row, (metric, ylabel, _) in enumerate(metrics):
            ax = axes[row, col]
            for color, ltv in zip(cmap, lambda_tvs):
                slc = sub[sub["lambda_tv"] == ltv].sort_values("tv_iters")
                ax.plot(slc["tv_iters"], slc[metric], "o-",
                        color=color, label=f"λ_TV={ltv}", linewidth=1.5, markersize=5)
            ax.axvline(2, color="red", linestyle="--", alpha=0.5, linewidth=1)
            ax.set_ylabel(ylabel, fontsize=10)
            if row == 0:
                ax.set_title(f"{ds_label}", fontsize=11, fontweight="bold")
            if row == len(metrics) - 1:
                ax.set_xlabel("tv_iters", fontsize=10)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == len(datasets) - 1:
                ax.legend(fontsize=8, loc="best", framealpha=0.9)

    fig.suptitle("PRISM tv_iters trade-off curve (red dashed = v1 default tv_iters=2)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout()
    out_path = out_dir / "prism_tv_iters_tradeoff.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.relative_to(ROOT)}")

    rows = []
    for ds, ds_label in datasets:
        sub = df[df["dataset"] == ds]
        for ltv in sorted(sub["lambda_tv"].unique()):
            slc = sub[sub["lambda_tv"] == ltv].sort_values("tv_iters")
            row = {"dataset": ds_label, "lambda_tv": ltv}
            for _, r in slc.iterrows():
                ti = int(r["tv_iters"])
                row[f"mae_iter{ti}"] = round(float(r["mae"]), 4)
                row[f"pearson_iter{ti}"] = round(float(r["pearson_r"]), 3)
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "prism_tv_iters_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
