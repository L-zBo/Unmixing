# Unmixing —— 拉曼光谱解混项目

本仓库面向 PE / PP / 淀粉 体系的拉曼面扫数据，主线方法为 **PRISM**（Physics-Regularized Iterative Spectral Mixing，在 NNLS 基础上引入波段加权、L2 Tikhonov 正则化与空间 TV 一致性约束）；NNLS / OLS / FCLS / NMF 作为对比与消融基线保留。应用背景为食品基质（特别是淀粉类）中聚烯烃微塑料（PE / PP）的拉曼定量解混。早期家族分类基线代码与脚本已整体归档到 `archive/legacy_classification/`，作为历史参照保留。

## 主线一句话

> 以 `PRISM`（Physics-Regularized Iterative Spectral Mixing）为主线方法，在已知端元条件下开展拉曼光谱有监督解混；以经典 `NNLS / CLS-OLS / FCLS` 和盲解混 `NMF` 作为对比与消融基线。通过合成真值数据验证丰度恢复精度，通过真实拉曼面扫数据验证解混结果的重构能力、空间一致性、物理可解释性（absent_load 假阳性）和泛化能力。**应用背景为食品基质（特别是淀粉类）中聚烯烃微塑料（PE / PP）的拉曼定量解混**。

## 当前进展（2026-05-08）

按"PPT 汇报 + 毕业论文第一章预备介绍"双标准准备，论证两个核心论点。

### 论点① — NNLS 适合 PE / PP / 淀粉 拉曼解混

- **OLS 出局**：在泛化场景下出现高达 **76.5%** 非物理负丰度（v13 P1 图），NNLS / FCLS / NMF 全场景为 0%
- **NMF 出局**：学到的端元偏离物理参考 **0.40 弧度（≈23°）**，丰度与真值 Pearson R ≈ 0；"NMF 重构 R² 0.996 看似最优"是乱学端元拟合谱的副产物
- **NNLS = FCLS 二选一选 NNLS**：合成真值上精度接近（RMSE 差 7%），NNLS 不强制 sum-to-one 约束、对端元谱归一化误差更鲁棒、实现简洁、自然产生稀疏解（平均 1.67 端元活跃，符合"一像素只含少量物质"的物理直觉）

支撑图表（**PPT 主线 3 图 + 1 表**）：

| # | 文件 | 论证 |
|---|---|---|
| 图① | `outputs/showcase/method_constraints/negative_coef_fraction_bars.png` | OLS 物理性差 |
| 图② | `outputs/showcase/synthetic_truth/synthetic_metric_comparison.png` | 合成真值 MAE / RMSE / R² 三联子图 |
| 图③ | `outputs/showcase/method_constraints/nmf_endmember_sam_bars.png` | NMF 端元偏离物理 |
| 表 | `outputs/showcase/method_comparison/method_overall_summary.csv` | 9 行 7 维度全维度汇总 |

### 论点② — ALS+L2 是 NNLS 解混的数学规范选择

按 2026-05-08 冻结口径，预处理选型**仅用文字 + 文献依据**说明，不出 PPT 图表（产物在 `outputs/showcase/preprocessing/` 与 `outputs/showcase/protocol_consistency/` 保留作为补充材料）：

- L2 归一化保持向量 Euclidean 长度，与 NNLS 在 L2 空间的最小二乘优化目标天然匹配；max 归一化破坏向量长度结构，不利线性最小二乘解混
- ALS λ=10⁶ + p=0.05 + L2 归一化是 [Eilers & Boelens 2005](https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/dd7c1919-302c-4ba0-8f88-8aa61e86bb9d) 与 [Spectroscopy Online Raman 标准工作流](https://www.spectroscopyonline.com/view/key-steps-in-the-workflow-to-analyze-raman-spectra) 推荐的 baseline 工作流
- 重构能力上 ALS+L2 与 ALS+max 持平（R² 0.923 平手），显著优于 none+L2（R² 0.784，差 18%）——证实 ALS 基线校正的必要性
- max 归一化在指纹峰相对强度保留上更高（保留率 0.49 vs 0.08），但这是**光谱可视化**用途，对**线性解混**并非核心指标

### 论点③ — PRISM 进一步降低假阳性 + 空间一致性

在 NNLS 已经满足"非负 + 经典物理解释"的基础上，**PRISM** 通过波段加权 + L2 Tikhonov + 空间 TV anchor 三项物理正则进一步压制噪声与歧义偏置：

- **真实数据 spatial_TV ↓ 41~65%**：3 个真实样本（PE+淀粉 / PP+淀粉 / 三组分 test）一致改善
- **真实数据 max 假阳性 ↓ 19~54%**：PP+淀粉 样本下"不应有 PE"的 max 丰度从 11.75% 降到 9.52%
- **真实数据 假阳性灾难消除**：PP+淀粉 样本中假预测含 PE > 10% 的像素比例从 NNLS 的 0.13% 降到 PRISM 的 **0.0%**
- **重构 RMSE 完美持平 NNLS**：PRISM 不以"拟合谱"为代价换空间一致性
- **合成数据 MAE ↓ 14%**：合成 NOISY 40×40 上 MAE 从 0.0515 降到 0.0441，Pearson r 升 39%
- **NNLS 是 PRISM 的退化形式**：`weight_mode="uniform", λ_L2=0, tv_iters=0` 时 PRISM ≡ NNLS

完整方法说明、参数扫描、加权策略（uniform vs endmember_std）消融见 [`docs/prism_method.md`](docs/prism_method.md)。PRISM 实验产物在 `outputs/experiments/prism_*`。

### 故事线（PPT / 论文按这条讲）

```
1. 任务：食品基质（淀粉）中聚烯烃微塑料（PE/PP）三组分拉曼面扫像素级解混
   ↓
2. 预处理选 ALS+L2     ←  论点② 数学规范 + 重构不输 + 文献依据
   ↓
3. 解混经典对比定基线  ←  论点① OLS 物理性差、NMF 端元乱、FCLS ≈ NNLS 选 NNLS
   ↓
4. 解混升级到 PRISM    ←  论点③ 假阳性 frac>10% 从 0.13% 降到 0%、spatial_TV 降 41~65%
   ↓
5. 实测效果            ←  v6/v9/v12 + prism_quick_check / real_check / abundance_viz
   ↓
6. 承下                ←  PRISM 输出作为后续 PRISM + 深度学习的物理先验
```

完整论证统稿（含每张图的读图说明 / 物理含义 / 关键数字 / 论文写作 checklist）见 [`docs/thesis_chapter1_figures.md`](docs/thesis_chapter1_figures.md)。

## 目录结构

```
Unmixing/
├── README.md                  # 项目总览（本文件）
├── AGENTS.md                  # 协作与提交规范
├── 项目结构说明.md            # 仓库结构详解
├── 论证总览.md                # PPT / 论文图表索引与引用映射（写 PPT / 论文时看这份）
├── docs/                      # 详细路线说明
│   ├── nnls_unmixing_flow.md  # 主线：NNLS 解混落地说明
│   ├── legacy_classification_flow.md  # 旧路线：家族分类 + 组级空间评估
│   └── thesis_chapter1_figures.md  # 论证统稿（200~400 字读图说明 + 论文写作 checklist）
├── preprocessing/             # 端元加载、协议化预处理（als_l2/als_max/none_l2）
│   ├── endmembers.py
│   ├── preprocess.py
│   └── preprocess_dataset.py  # 入口：把原始 dataset/ 处理成 outputs/preprocessing/
├── synthetic/                 # 有真值的合成解混数据
│   ├── generator.py
│   └── generate_dataset.py    # 入口：生成 outputs/synthetic_unmixing/
├── unmixing/                  # 经典解混核心（OLS/NNLS/FCLS + 盲 NMF）
│   └── unmix.py
├── visualization/             # 按图类型组织的可视化子包
│   ├── abundance/             # 丰度热图
│   ├── residual/              # 残差空间图
│   ├── reconstruction/        # 输入谱与重构谱对比
│   ├── method_comparison/     # 方法横向对比柱图
│   └── preprocessing/         # 预处理协议对比三视图与丰度网格
├── experiments/               # 主线 NNLS 实验入口（平铺，按用途命名）
│   ├── run_real_unmixing_single.py
│   ├── run_real_method_comparison.py
│   ├── run_batch_method_comparison.py
│   ├── run_synthetic_method_comparison.py
│   ├── run_real_preprocessing_comparison.py
│   ├── run_batch_preprocessing_comparison.py
│   ├── run_generalization_batch.py
│   ├── run_endmember_fingerprint_plot.py
│   ├── run_method_constraint_diagnostics.py
│   └── run_protocol_consistency_analysis.py
├── utils/                     # 通用 IO 工具（save_predictions / save_experiment_summary）
├── archive/                   # 历史归档（不参与主线）
│   ├── legacy_classification/ # 旧家族分类路线（v1~v5、external_test）
│   ├── legacy_data_pipeline/  # 旧 manifest / quality 数据管线
│   └── legacy_training/       # 旧训练相关脚本
├── dataset/                   # 原始数据（git 忽略，只读）
└── outputs/                   # 实验产物（git 忽略）
    ├── preprocessing/
    ├── synthetic_unmixing/
    ├── experiments/
    └── showcase/              # 从 experiments/ 里筛出的"展示型结果包"
```

## 主线入口速查

| 维度 | 入口脚本 | 输出位置 |
|---|---|---|
| 单图解混 | `experiments/run_real_unmixing_single.py` | `outputs/real_unmixing_single/` |
| 单图四方法对比 | `experiments/run_real_method_comparison.py` | `outputs/real_method_comparison/` |
| 多图四方法批量 | `experiments/run_batch_method_comparison.py` | `outputs/batch_method_comparison/` |
| 合成真值四方法 | `experiments/run_synthetic_method_comparison.py` | `outputs/synthetic_method_comparison/` |
| 单图三协议对比 | `experiments/run_real_preprocessing_comparison.py` | `outputs/real_preprocessing_comparison/` |
| 多图三协议批量 | `experiments/run_batch_preprocessing_comparison.py` | `outputs/batch_preprocessing_comparison/` |
| 跨淀粉源泛化 | `experiments/run_generalization_batch.py` | `outputs/generalization_batch/` |
| 端元指纹峰可视化 | `experiments/run_endmember_fingerprint_plot.py` | `outputs/experiments/formal_v15_endmember_fingerprint/` |
| 方法约束诊断（OLS 负值率 / NMF 端元 SAM / NNLS 稀疏度） | `experiments/run_method_constraint_diagnostics.py` | `outputs/experiments/formal_v13_method_constraint_diagnostics/` |
| 协议一致性（CV + 指纹峰保留） | `experiments/run_protocol_consistency_analysis.py` | `outputs/experiments/formal_v14_protocol_consistency/` |
| **PRISM 合成快速验证** | `experiments/run_prism_quick_check.py` | `outputs/experiments/prism_quick_check*/` |
| **PRISM 真实样本对比** | `experiments/run_prism_real_check.py` | `outputs/experiments/prism_real_check/` |
| **PRISM 丰度图可视化** | `experiments/run_prism_abundance_viz.py` | `outputs/experiments/prism_abundance_viz*/` |
| **PRISM 超参网格扫描** | `experiments/run_prism_param_sweep.py` | `outputs/experiments/prism_param_sweep/` |
| **PRISM absent_load 物理一致性** | `experiments/run_prism_absent_check.py` | `outputs/experiments/prism_absent_check/` |
| **PRISM 加权策略消融（STD vs UNI）** | `experiments/run_prism_synth_std_vs_uni.py` | `outputs/experiments/prism_synth_std_vs_uni/` |

详细说明见 [docs/nnls_unmixing_flow.md](docs/nnls_unmixing_flow.md)（经典解混主线）与 [docs/prism_method.md](docs/prism_method.md)（PRISM 方法）。

## 核心模块速查

- 端元加载与协议化预处理：`preprocessing/endmembers.py`、`preprocessing/preprocess.py`
- **主线方法 PRISM**（加权 NNLS + L2 Tikhonov + 空间 TV）：`unmixing/unmix.py::prism_unmix_spectra`
- 经典对比基线（OLS / NNLS / FCLS / NMF）：`unmixing/unmix.py`
- 合成真值数据生成：`synthetic/generator.py` + `synthetic/generate_dataset.py`
- 解混可视化（丰度热图 / 残差 / 重构 / 方法对比 / 协议对比 / 指纹峰）：`visualization/`（顶层 re-export 14 个绘图函数）

## 顶层包 import 用法

仓库根目录加进 `sys.path` 之后即可：

```python
from preprocessing import preprocess, endmembers
from synthetic.generator import generate_synthetic_map
from unmixing import prism_unmix_spectra, unmix_spectra, blind_nmf_unmix_spectra  # 顶层 re-export
from utils.io import save_predictions, save_experiment_summary
from visualization import plot_abundance_maps, plot_residual_map  # 顶层 re-export
```

`experiments/` 与入口脚本 `preprocessing/preprocess_dataset.py`、`synthetic/generate_dataset.py` 已自动把仓库根目录加入 `sys.path`，可以 `python experiments/run_xxx.py --help` 直接调用。

## 数据与产物管理原则

- `dataset/` 只读：不覆盖、不移动、不入 git
- `outputs/` 只产物：不入 git
- 真实数据是「弱标签」，不能冒充像素级真值；合成数据才有像素级真值
- 默认预处理协议为 `ALS + L2`，对照协议保留 `ALS + max`、`none + L2`

## 协作规范

提交流程、提交信息格式见 [AGENTS.md](AGENTS.md)：每个独立任务完成后先准备 commit、由用户确认 message 后再执行 commit；代码改动与文档改动分开提交，严禁空 commit 与 prompt 原文 message。
