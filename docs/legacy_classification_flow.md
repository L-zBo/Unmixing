# 旧分类路线流程说明（v1-v5）

> 本文档记录的是项目早期的「家族分类 + 组级空间评估」路线，对应实验脚本已归档到 `scripts/experiments/legacy_classification/`。当前主线已切换到 NNLS 解混，详见 [`nnls_unmixing_flow.md`](nnls_unmixing_flow.md)。本文保留作为历史背景与基线参照。

## 一句话理解项目

这个项目要做的事情是：

1. 读取拉曼二维图谱数据。
2. 把不同格式、不同坐标口径的数据统一预处理。
3. 按样本组而不是单条光谱来做浓度分档和空间结果分析。
4. 输出每组样本的最终判断结果，以及整张图上的空间分布图。

## 数据长什么样

每个样本组通常是一个文件夹，里面有很多`DATA-...-X...-Y....csv`文件。

- 一个`csv`文件：一个空间点上的一条拉曼光谱。
- 一整个文件夹：一组样本，也可以理解成一张二维拉曼图。
- `X,Y`：这条光谱在二维图上的空间坐标。

所以项目真正关心的不是某一条谱单独判得多准，而是：

- 整张图看起来对不对。
- 整组样本判断对不对。

## 第一层：原始数据

位置：

- `dataset/`

这里放最原始的数据，包括：

- 训练/开发数据
- 纯谱端元
- 外部测试数据`dataset/test/`

## 第二层：预处理

核心代码：

- `src/demixing/data/preprocess.py`
- `scripts/data/preprocess_dataset.py`

主要做的事：

1. 统一读取宽表和长表。
2. 统一横轴，必要时把波长转换成拉曼位移。
3. 重采样到统一的1024维波数轴。
4. 去尖峰、做基线校正、裁剪负值。
5. 同时保存校正后强度和归一化强度。

目的：

- 让后面的模型不用再处理格式混乱问题。

输出位置：

- `outputs/preprocessing/dataset_preprocessed_v1/`
- `outputs/preprocessing/test_preprocessed_v1/`

## 第三层：样本清单与质量分层

核心代码：

- `src/demixing/data/manifest.py`
- `src/demixing/data/quality.py`
- `src/demixing/data/splits.py`
- `scripts/data/build_quality_manifest.py`
- `scripts/data/build_sample_manifest.py`

主要做的事：

1. 识别样本属于哪个家族：
   - `pp_starch`
   - `pe_starch`
   - `pp_pe_starch`
2. 识别数据类型：
   - `raw`
   - `average`
   - `pure`
3. 识别浓度档位：
   - `low`
   - `medium`
   - `high`
4. 给每条谱一个质量等级：
   - `A`
   - `B`
   - `C`
5. 给每组样本生成`sample_group_id`，方便按组切分。

目的：

- 不把所有光谱乱混在一起训练。
- 按家族、按样本组、按质量来管理数据。

关键输出：

- `outputs/preprocessing/dataset_preprocessed_v1/_reports/sample_manifest.csv`

## 第四层：模型与方法

目前项目里有两条主线。

### 路线A：统一解混主模型

核心代码：

- `src/demixing/models/unified_unmixing.py`
- `src/demixing/training/losses.py`
- `src/demixing/training/trainer.py`

它想做的事情：

- 用一个统一框架同时兼容盲解混、端元引导、弱监督这些模式。
- 既保留端元解释性，又希望能做浓度分档。

它更适合做：

- 长期主方法
- 后续论文主线

### 路线B：家族分开强基线

核心代码：

- `src/demixing/evaluation/classical_models.py`

它做的事情：

- 按家族分别训练更适合自己的分类器。
- 目标是先把性能冲高。

它更适合做：

- 当前最强性能方案
- 给统一解混主模型做性能参照

## 第五层：实验脚本

主要实验脚本：

- `scripts/experiments/legacy_classification/run_formal_v1.py`
- `scripts/experiments/legacy_classification/run_formal_v2.py`
- `scripts/experiments/legacy_classification/run_formal_v3_family_svc.py`
- `scripts/experiments/legacy_classification/run_formal_v4_group_spatial.py`
- `scripts/experiments/legacy_classification/run_external_test_family_svc.py`

各自的作用：

- `formal_v1`：统一模型第一版
- `formal_v2`：统一模型第二版
- `formal_v3_family_svc`：家族分开强基线
- `formal_v4_group_spatial`：把组级/空间级作为主评估单位
- `external_test_family_svc`：对外部测试集做推理和空间图输出

## 第六层：结果输出

统一放在：

- `outputs/experiments/`

常见输出内容：

- `training/`：训练记录和模型文件
- `reports/`：预测结果表和实验摘要
- `figures/`：混淆矩阵、空间预测图、家族分组图等

## 现在项目的主评估口径

现在已经明确：

- 不再把“单条光谱”作为最终目标。
- 主要看“样本组/整张图”的结果。

也就是说，现在更重要的是：

1. 这组样本最后分到哪个浓度档。
2. 整张图的空间分布图合不合理。
3. 同一家族的结果是否稳定。

## 目前最重要的正式结果

### 1. 家族分开强基线

位置：

- `outputs/experiments/formal_v3_family_svc/`

它提供：

- 单谱级测试结果
- 组级投票结果
- 混淆矩阵

### 2. 组级/空间级正式结果

位置：

- `outputs/experiments/formal_v4_group_spatial/`

它提供：

- 组级主结果
- 组级混淆矩阵
- 组级家族准确率

### 3. 外部测试集推理

位置：

- `outputs/experiments/external_test_family_svc_v1/`

它提供：

- 外部测试预测表
- 每个家族一张40x40空间预测图

## 小白版总结

如果把整个项目比作看显微图：

- 一条谱 = 图上的一个像素点
- 一组文件 = 一张图
- 一整个样本组 = 我们真正要判断的对象

所以项目不是在干“每一条谱都单独分类”这么简单的事，而是在做：

**从很多像素谱里，综合判断这一整张图、这一整组样本到底属于哪个浓度档，并把空间分布画出来。**
