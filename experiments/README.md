# Experiments 实验脚本入口

本目录平铺所有 NNLS 主线的实验脚本。每个脚本对应一个明确的实验场景，可以直接从仓库根目录运行。

## 文件列表

| 文件名 | 用途 | 主要产物目录 |
|---|---|---|
| `run_real_unmixing_single.py` | 单张真实面扫图上跑 OLS/NNLS/FCLS/NMF | `outputs/real_unmixing_single/` |
| `run_real_method_comparison.py` | 单张图上四方法详细对比（含图） | `outputs/real_method_comparison/` |
| `run_batch_method_comparison.py` | 多张典型图四方法批量对比汇总 | `outputs/batch_method_comparison/` |
| `run_synthetic_method_comparison.py` | 合成真值数据上跑四方法定量对比 | `outputs/synthetic_method_comparison/` |
| `run_real_preprocessing_comparison.py` | 单张图上对比三协议（als_l2/als_max/none_l2） | `outputs/real_preprocessing_comparison/` |
| `run_batch_preprocessing_comparison.py` | 多张图上批量对比三协议 | `outputs/batch_preprocessing_comparison/` |
| `run_generalization_batch.py` | 跨淀粉来源（展艺/新良/甘汁园）泛化批量评估 | `outputs/generalization_batch/` |
| `run_endmember_fingerprint_plot.py` | 三端元纯谱叠加 + 文献指纹峰标注（PPT 物理基础页） | `outputs/experiments/formal_v15_endmember_fingerprint/` |
| `run_method_constraint_diagnostics.py` | 逐像素负丰度率 / NMF 端元 SAM / NNLS 稀疏度 | `outputs/experiments/formal_v13_method_constraint_diagnostics/` |
| `run_protocol_consistency_analysis.py` | 三协议下逐像素 CV + 指纹峰保留率 | `outputs/experiments/formal_v14_protocol_consistency/` |

## 三大实验维度

### 1.方法对比（OLS vs NNLS vs FCLS vs NMF）

入口：

- `run_real_method_comparison.py`
- `run_batch_method_comparison.py`
- `run_synthetic_method_comparison.py`

### 2.预处理协议对比（als_l2 vs als_max vs none_l2）

入口：

- `run_real_preprocessing_comparison.py`
- `run_batch_preprocessing_comparison.py`

### 3.泛化稳定性（跨淀粉来源）

入口：

- `run_generalization_batch.py`

### 4.PPT 证据补强（论证 NNLS / ALS+L2 选型）

针对"NNLS 比对照更适合"和"ALS+L2 比对照更适合"两条 PPT 论点专门补的指标维度：

- `run_endmember_fingerprint_plot.py` — 端元纯谱 + 文献指纹峰标注（物理基础页）
- `run_method_constraint_diagnostics.py` — 逐像素负丰度率（OLS 暴露非物理性）/ NMF 端元 SAM / NNLS 稀疏度
- `run_protocol_consistency_analysis.py` — 三协议下逐像素 CV + 指纹峰保留率（区分 L2 与 max）

## 推荐运行顺序

```text
单图调试 → 多图批量 → 合成真值定量 → 泛化稳定性
   ↓           ↓            ↓               ↓
single      batch       synthetic     generalization
```

## import 约定

这些脚本会自动把仓库根目录加入 `sys.path`，因此内部统一使用顶层职责包：

- `preprocessing.*`
- `synthetic.*`
- `unmixing.*`
- `visualization.*`
- `utils.*`

不再依赖旧的 `src/demixing/` 或 `scripts/experiments/nnls_unmixing/` 路径。
