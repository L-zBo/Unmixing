"""Build two overall summary tables aggregating existing v8 / v9 / v13 / v14 CSVs.

Outputs
-------
- ``outputs/showcase/method_comparison/method_overall_summary.csv``
  Per (dimension, metric, direction) row × (OLS / NNLS / FCLS / NMF) columns,
  with ``best_method`` automatically populated from the direction.

- ``outputs/showcase/protocol_consistency/preprocessing_overall_summary.csv``
  Per protocol (als_l2 / als_max / none_l2) × (R² / CV / fingerprint retention)
  columns.

Read-only on the source CSVs. Run after ``run_synthetic_metric_plot.py`` so
that the v9 csv has the ``orig_r2`` column.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SHOWCASE = ROOT / "outputs/showcase"

V9_SYNTHETIC = SHOWCASE / "synthetic_truth/synthetic_method_comparison_summary.csv"
V8_REAL_BATCH = SHOWCASE / "method_comparison/batch_method_comparison_summary.csv"
V13_CONSTRAINT = SHOWCASE / "method_constraints/method_constraint_summary.csv"
V13_NMF_SAM = SHOWCASE / "method_constraints/nmf_endmember_sam.csv"
V13_NNLS_SPARSITY = SHOWCASE / "method_constraints/nnls_sparsity_summary.csv"

V14_R2 = SHOWCASE / "protocol_consistency/protocol_reconstruction_r2_summary.csv"
V14_CONSISTENCY = SHOWCASE / "protocol_consistency/protocol_consistency_summary.csv"
V14_RETENTION = SHOWCASE / "protocol_consistency/fingerprint_retention_summary.csv"

DEMIX_OUT = SHOWCASE / "method_comparison/method_overall_summary.csv"
PREPROC_OUT = SHOWCASE / "protocol_consistency/preprocessing_overall_summary.csv"

METHOD_ORDER = ("ols", "nnls", "fcls", "nmf")
PROTOCOL_ORDER = ("als_l2", "als_max", "none_l2")


def _by_method_value(df: pd.DataFrame, method_col: str, value_col: str, agg: str = "mean") -> dict[str, float]:
    out: dict[str, float] = {}
    if method_col not in df.columns or value_col not in df.columns:
        return {m: float("nan") for m in METHOD_ORDER}
    grouped = df.groupby(method_col)[value_col].agg(agg)
    for m in METHOD_ORDER:
        out[m] = float(grouped.get(m, np.nan))
    return out


def _best(values: dict[str, float], direction: str) -> str:
    """Return the method name(s) achieving the best value under direction."""
    valid = {k: v for k, v in values.items() if not np.isnan(v)}
    if not valid:
        return ""
    if direction == "lower":
        target = min(valid.values())
    elif direction == "higher":
        target = max(valid.values())
    else:
        return ""
    winners = [k for k, v in valid.items() if abs(v - target) < 1e-9]
    return " / ".join(w.upper() for w in winners)


def _row(dimension: str, metric: str, direction: str, values: dict[str, float], note: str = "") -> dict[str, object]:
    arrow = "↓" if direction == "lower" else "↑" if direction == "higher" else ""
    return {
        "dimension": dimension,
        "metric": f"{metric} {arrow}".strip(),
        "direction": direction,
        "OLS": values.get("ols", np.nan),
        "NNLS": values.get("nnls", np.nan),
        "FCLS": values.get("fcls", np.nan),
        "NMF": values.get("nmf", np.nan),
        "best_method": _best(values, direction),
        "note": note,
    }


def build_demixing_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    # 1. Abundance accuracy from v9 synthetic truth
    if V9_SYNTHETIC.exists():
        df9 = pd.read_csv(V9_SYNTHETIC)
        m = {row["method"]: row for _, row in df9.iterrows()}

        def col(name: str) -> dict[str, float]:
            return {k: float(v.get(name, np.nan)) for k, v in m.items()}

        rows.append(_row("丰度精度", "MAE", "lower", col("orig_mae"),
                         note="v9 合成真值；OLS=NNLS（合成数据非负，OLS 退化）"))
        rows.append(_row("丰度精度", "RMSE", "lower", col("orig_rmse"),
                         note="v9 合成真值"))
        if "orig_r2" in df9.columns:
            rows.append(_row("丰度精度", "R² (Pearson²)", "higher", col("orig_r2"),
                             note="v9 合成真值；NMF≈0 说明丰度跟真值零相关"))
        rows.append(_row("端元保真", "SAM (rad)", "lower", col("mean_sam_rad"),
                         note="v9 合成场景重构端元角；NMF 0.218 看似最低但端元偏离物理参考（见下行）"))

    # 2. Negative coefficient fraction across all v13 scenarios
    if V13_CONSTRAINT.exists():
        df13c = pd.read_csv(V13_CONSTRAINT)
        rows.append(_row("物理可解释性", "负丰度率(全部场景均值)", "lower",
                         _by_method_value(df13c, "method", "neg_coef_fraction", "mean"),
                         note="v13；19 个 (label, component) 平均"))
        rows.append(_row("物理可解释性", "负丰度率(最坏场景)", "lower",
                         _by_method_value(df13c, "method", "neg_coef_fraction", "max"),
                         note="v13；OLS 在展艺淀粉+PP 场景达 76.5%"))

    # 3. NMF endmember偏离（其他方法 N/A）
    if V13_NMF_SAM.exists():
        df_sam = pd.read_csv(V13_NMF_SAM)
        nmf_sam_mean = float(df_sam["sam_rad"].mean()) if "sam_rad" in df_sam.columns else np.nan
        rows.append(_row("端元偏离物理参考", "NMF 端元 SAM (rad)", "lower",
                         {"ols": np.nan, "nnls": np.nan, "fcls": np.nan, "nmf": nmf_sam_mean},
                         note="v13；OLS/NNLS/FCLS 用已知端元，无此问题"))

    # 4. NNLS sparsity（专属 NNLS）
    if V13_NNLS_SPARSITY.exists():
        df_sp = pd.read_csv(V13_NNLS_SPARSITY)
        nnls_active = float(df_sp["mean_active_count"].mean()) if "mean_active_count" in df_sp.columns else np.nan
        rows.append(_row("自然稀疏性", "平均活跃端元数", "lower",
                         {"ols": np.nan, "nnls": nnls_active, "fcls": np.nan, "nmf": np.nan},
                         note="v13；NNLS 平均仅 ~1.8 个端元活跃，自然稀疏，符合物理"))

    # 5. Reconstruction R² on real data from v8
    if V8_REAL_BATCH.exists():
        df8 = pd.read_csv(V8_REAL_BATCH)
        rows.append(_row("重构能力", "重构 R² (真实数据)", "higher",
                         _by_method_value(df8, "method", "mean_residual_r2", "mean"),
                         note="v8；NMF 重构看似最好但端元偏离（见上）"))

    return pd.DataFrame(rows)


def build_preprocessing_summary() -> pd.DataFrame:
    out: dict[str, dict[str, float]] = {p: {} for p in PROTOCOL_ORDER}

    if V14_R2.exists():
        df = pd.read_csv(V14_R2)
        if {"protocol", "mean_residual_r2"}.issubset(df.columns):
            grouped = df.groupby("protocol")["mean_residual_r2"].mean()
            for p in PROTOCOL_ORDER:
                out[p]["重构 R² ↑"] = float(grouped.get(p, np.nan))

    if V14_CONSISTENCY.exists():
        df = pd.read_csv(V14_CONSISTENCY)
        if {"protocol", "cv"}.issubset(df.columns):
            grouped = df.groupby("protocol")["cv"].mean()
            for p in PROTOCOL_ORDER:
                out[p]["跨像素 CV ↓"] = float(grouped.get(p, np.nan))

    if V14_RETENTION.exists():
        df = pd.read_csv(V14_RETENTION)
        if {"protocol", "relative_intensity_mean"}.issubset(df.columns):
            grouped = df.groupby("protocol")["relative_intensity_mean"].mean()
            for p in PROTOCOL_ORDER:
                out[p]["指纹峰保留率均值 ↑"] = float(grouped.get(p, np.nan))
            grouped_std = df.groupby("protocol")["relative_intensity_std"].mean()
            for p in PROTOCOL_ORDER:
                out[p]["指纹峰保留率离散度 ↓"] = float(grouped_std.get(p, np.nan))

    rows = []
    metric_cols = ["重构 R² ↑", "跨像素 CV ↓", "指纹峰保留率均值 ↑", "指纹峰保留率离散度 ↓"]
    for p in PROTOCOL_ORDER:
        row = {"protocol": p}
        for c in metric_cols:
            row[c] = out[p].get(c, np.nan)
        rows.append(row)
    df_out = pd.DataFrame(rows)

    best = {}
    for c in metric_cols:
        values = df_out[c].astype(float).to_numpy()
        if np.all(np.isnan(values)):
            best[c] = ""
            continue
        if c.endswith("↑"):
            idx = int(np.nanargmax(values))
        else:
            idx = int(np.nanargmin(values))
        best[c] = df_out.iloc[idx]["protocol"]
    best_row = {"protocol": "best"}
    for c in metric_cols:
        best_row[c] = best[c]
    df_out = pd.concat([df_out, pd.DataFrame([best_row])], ignore_index=True)
    return df_out


def main() -> None:
    DEMIX_OUT.parent.mkdir(parents=True, exist_ok=True)
    PREPROC_OUT.parent.mkdir(parents=True, exist_ok=True)

    df_demix = build_demixing_summary()
    df_demix.to_csv(DEMIX_OUT, index=False)
    print(f"[OK] Demixing overall summary:      {DEMIX_OUT} ({len(df_demix)} rows)")

    df_pre = build_preprocessing_summary()
    df_pre.to_csv(PREPROC_OUT, index=False)
    print(f"[OK] Preprocessing overall summary: {PREPROC_OUT} ({len(df_pre)} rows)")


if __name__ == "__main__":
    main()
