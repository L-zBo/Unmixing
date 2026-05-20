# Experiments 实验脚本入口（PRISM 主线 + 7 方法对比 + 论证 PPT 证据）

本目录平铺所有 PRISM 主线的实验脚本。每个脚本对应一个明确的实验场景，可直接从仓库根目录运行：`python experiments/run_xxx.py --help`。

主线方法 **PRISM**（Physics-Regularized Iterative Spectral Mixing，详见 [`../docs/prism_method.md`](../docs/prism_method.md)），其他方法（NNLS / OLS / FCLS / NMF / MCR-ALS）作为对比基线。**NNLS 是 PRISM 在 `weight_mode=uniform, λ_L2=0, tv_iters=0` 时的退化形式。**

---

## 按场景分组的脚本索引（共 20 个）

### 1. PRISM 主线（6 个）

| 文件 | 用途 | 主要产物 |
|---|---|---|
| `run_prism_quick_check.py` | 合成数据上 PRISM vs NNLS 快速验证 | `outputs/experiments/prism_quick_check*/` |
| `run_prism_abundance_viz.py` | 合成数据上 PRISM vs NNLS 丰度图阵列可视化 | `outputs/experiments/prism_abundance_viz*/` |
| `run_prism_real_check.py` | 3 个真实样本上 PRISM vs NNLS 对比（spatial_TV / 重构 RMSE） | `outputs/experiments/prism_real_check/` |
| `run_prism_absent_check.py` | absent_load 物理一致性测试（PP+淀粉 样本中"不应有 PE"的假阳性率） | `outputs/experiments/prism_absent_check/` |
| `run_prism_param_sweep.py` | 34 配置超参网格扫描（`lambda_l2 × lambda_tv × tv_iters × weight_mode`） | `outputs/experiments/prism_param_sweep/` |
| `run_prism_synth_std_vs_uni.py` | 加权策略消融（uniform vs endmember_std），证 weight_mode='auto' 的保守阈值合理 | `outputs/experiments/prism_synth_std_vs_uni/` |

### 2. MCR-ALS 对比（1 个，拉曼社区事实标准）

| 文件 | 用途 | 主要产物 |
|---|---|---|
| `run_mcr_als_check.py` | hard-constrained（端元锁死 = NNLS）+ semi-blind（端元 init 但漂移到 Pearson r ≈ 0 灾难） | `outputs/experiments/mcr_als_check_formal_v1/` |

### 3. 经典 4 方法主线对比（4 个）

| 文件 | 用途 | 主要产物 |
|---|---|---|
| `run_real_unmixing_single.py` | 单张真实面扫图上跑 OLS/NNLS/FCLS/NMF | `outputs/real_unmixing_single/` |
| `run_real_method_comparison.py` | 单张图上四方法详细对比（含图） | `outputs/real_method_comparison/` |
| `run_batch_method_comparison.py` | 多张典型图四方法批量对比汇总 | `outputs/batch_method_comparison/` |
| `run_synthetic_method_comparison.py` | 合成真值数据上跑四方法定量对比 | `outputs/synthetic_method_comparison/` |

### 4. 预处理协议对比（2 个，论点② ALS+L2 选型）

| 文件 | 用途 | 主要产物 |
|---|---|---|
| `run_real_preprocessing_comparison.py` | 单张图上对比三协议（als_l2/als_max/none_l2） | `outputs/real_preprocessing_comparison/` |
| `run_batch_preprocessing_comparison.py` | 多张图上批量对比三协议 | `outputs/batch_preprocessing_comparison/` |

### 5. 泛化稳定性（1 个）

| 文件 | 用途 | 主要产物 |
|---|---|---|
| `run_generalization_batch.py` | 跨淀粉来源（展艺/新良/甘汁园）泛化批量评估 | `outputs/generalization_batch/` |

### 6. PPT 证据补强 + 收敛性诊断（5 个）

| 文件 | 用途 | 主要产物 |
|---|---|---|
| `run_endmember_fingerprint_plot.py` | 三端元纯谱叠加 + 文献指纹峰标注（物理基础页） | `outputs/experiments/formal_v15_endmember_fingerprint/` |
| `run_method_constraint_diagnostics.py` | 逐像素负丰度率 / NMF 端元 SAM / NNLS 稀疏度 | `outputs/experiments/formal_v13_method_constraint_diagnostics/` |
| `run_protocol_consistency_analysis.py` | 三协议下逐像素 CV + 指纹峰保留率 | `outputs/experiments/formal_v14_protocol_consistency/` |
| `run_synthetic_metric_plot.py` | 合成真值 MAE/RMSE/R² 三联子图（v9 后续出图） | `outputs/showcase/synthetic_truth/` |
| `run_prism_convergence_plot.py` | tv_iters trade-off 收敛曲线（反驳 ADMM 收敛性质疑） | `outputs/showcase/prism_convergence/` |

### 7. 跨实验汇总（1 个，单一事实源）

| 文件 | 用途 | 主要产物 |
|---|---|---|
| `run_overall_summary.py` | **7 方法 × 9 指标 × 多数据集** 横向总表（论文 §3 引用单一事实源） | `outputs/showcase/method_comparison/method_overall_summary.csv` |

---

## 推荐运行顺序（论文写作场景）

```text
1. 单图调试   → run_real_unmixing_single.py / run_real_method_comparison.py
2. 多图批量   → run_batch_method_comparison.py
3. 合成真值   → run_synthetic_method_comparison.py + run_synthetic_metric_plot.py
4. PRISM 主线 → run_prism_quick_check.py → run_prism_param_sweep.py → run_prism_real_check.py
5. PRISM 物理 → run_prism_absent_check.py + run_prism_synth_std_vs_uni.py + run_prism_convergence_plot.py
6. MCR-ALS    → run_mcr_als_check.py（含 hard / semi-blind 双跑）
7. 泛化       → run_generalization_batch.py
8. PPT 证据   → run_endmember_fingerprint_plot.py / run_method_constraint_diagnostics.py / run_protocol_consistency_analysis.py
9. 汇总总表   → run_overall_summary.py（论文 §3 表 1 数据来源）
```

---

## import 约定

这些脚本会自动把仓库根目录加入 `sys.path`，因此内部统一使用顶层职责包：

- `preprocessing.*`
- `synthetic.*`
- `unmixing.*`（含 `prism_unmix_spectra` / `mcr_als_unmix_spectra` / `unmix_spectra` / `blind_nmf_unmix_spectra`）
- `visualization.*`
- `utils.*`

不再依赖旧的 `src/demixing/` 或 `scripts/experiments/nnls_unmixing/` 路径。
