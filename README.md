# Unmixing —— 拉曼光谱解混项目

本仓库面向 PE / PP / 淀粉 体系的拉曼面扫数据，主线方法为 NNLS 经典解混；同时保留早期家族分类基线作为对照。

## 主线一句话

> 以 `NNLS` 为主线方法，在已知端元条件下开展拉曼光谱有监督解混，并以 `CLS/OLS`、`FCLS` 和盲解混方法 `NMF` 作为对比；通过合成真值数据验证丰度恢复精度，通过真实拉曼面扫数据验证解混结果的重构能力、空间一致性和泛化能力。

## 目录结构

```
Unmixing/
├── README.md                       # 项目总览（本文件）
├── AGENTS.md                       # 协作与提交规范
├── docs/                           # 详细路线说明
│   ├── nnls_unmixing_flow.md       # 主线：NNLS 解混落地说明
│   └── legacy_classification_flow.md  # 旧路线：家族分类 + 组级空间评估
├── src/demixing/                   # 核心代码
│   ├── data/                       # 数据/端元/预处理/合成数据
│   ├── evaluation/                 # 解混与基线评估
│   ├── models/                     # 统一解混网络、空间 CNN
│   ├── training/                   # 训练器与损失
│   └── visualization/              # 解混与分类可视化
├── scripts/
│   ├── data/                       # 数据预处理与合成数据生成
│   ├── train/                      # 训练入口
│   └── experiments/                # 正式实验脚本（按路线分组）
│       ├── nnls_unmixing/          # 主线：v6-v12
│       └── legacy_classification/  # 旧路线：v1-v5、external_test
├── dataset/                        # 原始数据（git 忽略，只读）
└── outputs/                        # 实验产物（git 忽略）
    ├── preprocessing/
    ├── synthetic_unmixing/
    └── experiments/
```

## 两条路线

| 路线 | 入口脚本 | 详细说明 |
| --- | --- | --- |
| NNLS 解混（主线） | `scripts/experiments/nnls_unmixing/run_formal_v6` ~ `v12` | [docs/nnls_unmixing_flow.md](docs/nnls_unmixing_flow.md) |
| 家族分类基线（历史） | `scripts/experiments/legacy_classification/run_formal_v1` ~ `v5`、`run_external_test_family_svc` | [docs/legacy_classification_flow.md](docs/legacy_classification_flow.md) |

## 核心模块速查

- 端元加载与协议化预处理：`src/demixing/data/endmembers.py`、`src/demixing/data/preprocess.py`
- 经典解混（OLS / NNLS / FCLS / NMF）：`src/demixing/evaluation/classical_unmixing.py`
- 合成真值数据生成：`src/demixing/data/synthetic_unmixing.py` + `scripts/data/generate_synthetic_unmixing_dataset.py`
- 解混可视化（丰度热图 / 残差 / 重构 / 汇总）：`src/demixing/visualization/classical_unmixing.py`

## 数据与产物管理原则

- `dataset/` 只读：不覆盖、不移动、不入 git
- `outputs/` 只产物：不入 git
- 真实数据是「弱标签」，不能冒充像素级真值；合成数据才有像素级真值
- 默认预处理协议为 `ALS + L2`，对照协议保留 `ALS + max`、`none + L2`

## 协作规范

提交流程、提交信息格式见 [AGENTS.md](AGENTS.md)：每个独立任务完成后立即一次提交并推送，代码改动与文档改动分开提交。
