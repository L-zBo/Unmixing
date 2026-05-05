# Unmixing —— 拉曼光谱解混项目

本仓库面向 PE / PP / 淀粉 体系的拉曼面扫数据，主线方法为 NNLS 经典解混；早期家族分类基线代码与脚本已整体归档到 `archive/legacy_classification/`，作为历史参照保留。

## 主线一句话

> 以 `NNLS` 为主线方法，在已知端元条件下开展拉曼光谱有监督解混，并以 `CLS/OLS`、`FCLS` 和盲解混方法 `NMF` 作为对比；通过合成真值数据验证丰度恢复精度，通过真实拉曼面扫数据验证解混结果的重构能力、空间一致性和泛化能力。

## 目录结构

```
Unmixing/
├── README.md                  # 项目总览（本文件）
├── AGENTS.md                  # 协作与提交规范
├── 项目结构说明.md            # 仓库结构详解
├── docs/                      # 详细路线说明
│   ├── nnls_unmixing_flow.md  # 主线：NNLS 解混落地说明
│   └── legacy_classification_flow.md  # 旧路线：家族分类 + 组级空间评估
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
│   └── run_generalization_batch.py
├── utils/                     # 通用 IO 工具（save_predictions / save_experiment_summary）
├── archive/                   # 历史归档（不参与主线）
│   ├── legacy_classification/ # 旧家族分类路线（v1~v5、external_test）
│   ├── legacy_data_pipeline/  # 旧 manifest / quality 数据管线
│   └── legacy_training/       # 旧训练相关脚本
├── dataset/                   # 原始数据（git 忽略，只读）
└── outputs/                   # 实验产物（git 忽略）
    ├── preprocessing/
    ├── synthetic_unmixing/
    └── experiments/
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

详细说明见 [docs/nnls_unmixing_flow.md](docs/nnls_unmixing_flow.md)。

## 核心模块速查

- 端元加载与协议化预处理：`preprocessing/endmembers.py`、`preprocessing/preprocess.py`
- 经典解混（OLS / NNLS / FCLS / NMF）：`unmixing/unmix.py`
- 合成真值数据生成：`synthetic/generator.py` + `synthetic/generate_dataset.py`
- 解混可视化（丰度热图 / 残差 / 重构 / 方法对比 / 协议对比）：`visualization/`（顶层 re-export 8 个绘图函数）

## 顶层包 import 用法

仓库根目录加进 `sys.path` 之后即可：

```python
from preprocessing import preprocess, endmembers
from synthetic.generator import generate_synthetic_map
from unmixing.unmix import unmix_spectra, blind_nmf_unmix_spectra
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
