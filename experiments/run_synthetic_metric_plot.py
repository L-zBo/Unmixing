"""Plot RMSE / MAE / R² (Pearson²) bars from the v9 synthetic-truth summary.

Reads the v9 ``synthetic_method_comparison_summary.csv``, augments it with
``orig_r2`` / ``proj_r2`` columns (R² = Pearson²), and writes a 3-panel bar
chart.

Outputs
-------
- ``outputs/experiments/formal_v9_synthetic_method_comparison/synthetic_method_comparison_summary.csv`` (in-place augment)
- ``outputs/experiments/formal_v9_synthetic_method_comparison/synthetic_metric_comparison.png`` (new)
- ``outputs/showcase/synthetic_truth/synthetic_method_comparison_summary.csv`` (mirror)
- ``outputs/showcase/synthetic_truth/synthetic_metric_comparison.png`` (mirror)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.method_comparison import plot_synthetic_metric_subplots


DEFAULT_SOURCE_CSV = (
    ROOT
    / "outputs/experiments/formal_v9_synthetic_method_comparison/synthetic_method_comparison_summary.csv"
)
DEFAULT_OUTPUT_FIG = (
    ROOT
    / "outputs/experiments/formal_v9_synthetic_method_comparison/synthetic_metric_comparison.png"
)
SHOWCASE_CSV = (
    ROOT / "outputs/showcase/synthetic_truth/synthetic_method_comparison_summary.csv"
)
SHOWCASE_FIG = (
    ROOT / "outputs/showcase/synthetic_truth/synthetic_metric_comparison.png"
)


def augment_with_r2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "orig_pearson_r" in df.columns:
        df["orig_r2"] = df["orig_pearson_r"].astype(float) ** 2
    if "proj_pearson_r" in df.columns:
        df["proj_r2"] = df["proj_pearson_r"].astype(float) ** 2
    return df


def _build_chart(df: pd.DataFrame, fig_path: Path) -> None:
    plot_synthetic_metric_subplots(
        df,
        metric_cols=("orig_mae", "orig_rmse", "orig_r2"),
        metric_labels=("MAE", "RMSE", "R²"),
        metric_directions=("lower", "lower", "higher"),
        output_path=fig_path,
        title="Synthetic-truth abundance recovery (PE / PP / starch)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FIG)
    parser.add_argument("--no-showcase", action="store_true")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(
            f"Source CSV not found: {args.source}. "
            "Run experiments/run_synthetic_method_comparison.py first."
        )

    df = pd.read_csv(args.source)
    df = augment_with_r2(df)
    df.to_csv(args.source, index=False)
    _build_chart(df, args.output)
    print(f"[OK] CSV augmented in place: {args.source}")
    print(f"[OK] Figure written:        {args.output}")

    if not args.no_showcase:
        SHOWCASE_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SHOWCASE_CSV, index=False)
        _build_chart(df, SHOWCASE_FIG)
        print(f"[OK] Showcase CSV mirror:   {SHOWCASE_CSV}")
        print(f"[OK] Showcase fig mirror:   {SHOWCASE_FIG}")


if __name__ == "__main__":
    main()
