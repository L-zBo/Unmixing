# archive归档区说明

这个文件夹是**历史代码归档区**。

它的作用不是继续承载当前主线开发，而是：

- 把已经不属于当前主线的代码单独收起来
- 保留历史实验和旧路线的可追溯性
- 避免这些旧代码继续混在主线目录里干扰判断

当前主线是：

- 预处理主线：`ALS + L2`
- 解混主线：`NNLS`
- 对比方法：`CLS/OLS`、`FCLS`、`NMF`

所以放进这里的代码，不代表“没价值”，而是代表：

- 现在不是主线
- 现在不是优先维护对象
- 主要作为历史记录、旧结果来源和对照参考

## 1.当前归档内容

### `archive/legacy_classification/`

这部分是项目早期“家族分类/组级空间评估”路线的完整归档。

包含两块：

- `archive/legacy_classification/scripts_legacy_classification/`
- `archive/legacy_classification/src_demixing_legacy/`

#### 1.1脚本入口说明

- `archive/legacy_classification/scripts_legacy_classification/run_formal_v1.py`
  早期统一模型第一版正式实验入口。

- `archive/legacy_classification/scripts_legacy_classification/run_formal_v2.py`
  早期统一模型第二版正式实验入口，带锚点NNLS基线和更完整的训练/推理流程。

- `archive/legacy_classification/scripts_legacy_classification/run_formal_v3_family_svc.py`
  家族分开训练的SVC基线实验入口。

- `archive/legacy_classification/scripts_legacy_classification/run_formal_v4_group_spatial.py`
  按样本组/空间图做主评估单位的组级实验入口。

- `archive/legacy_classification/scripts_legacy_classification/run_formal_v5_spatial_cnn.py`
  基于空间图输入的CNN路线实验入口。

- `archive/legacy_classification/scripts_legacy_classification/run_external_test_family_svc.py`
  旧路线下外部测试集的SVC推理入口。

#### 1.2源码模块说明

##### `archive/legacy_classification/src_demixing_legacy/data/`

- `dataset.py`
  旧路线的一维光谱数据集读取逻辑。

- `group_dataset.py`
  旧路线的组级/空间图数据集构造逻辑。

- `splits.py`
  旧路线下训练/验证/测试划分规则。

##### `archive/legacy_classification/src_demixing_legacy/evaluation/`

- `baselines.py`
  旧路线锚点NNLS等基线评估逻辑。

- `classical_models.py`
  旧路线家族分类基线方法实现。

- `inference.py`
  旧路线模型推理与结果保存逻辑。

##### `archive/legacy_classification/src_demixing_legacy/models/`

- `spatial_cnn.py`
  旧路线空间CNN模型定义。

- `unified_unmixing.py`
  旧路线统一解混网络定义。

##### `archive/legacy_classification/src_demixing_legacy/training/`

- `losses.py`
  旧路线训练损失定义。

- `spatial_trainer.py`
  旧路线空间CNN训练逻辑。

- `trainer.py`
  旧路线统一模型训练逻辑。

##### `archive/legacy_classification/src_demixing_legacy/visualization/`

- `plots.py`
  旧路线相关图表输出逻辑。

### `archive/legacy_training/`

这部分是独立训练入口归档。

- `archive/legacy_training/scripts_train/train_unified_unmixing.py`
  旧统一解混网络训练脚本入口。

这类代码和当前`NNLS + ALS+L2`主线不是一回事，所以先单独收起来，避免误以为它还是当前推荐路线。

## 2.为什么这些代码被归档

原因很直接：

1. 它们不属于当前主线
2. 它们会干扰“现在项目到底在做什么”的判断
3. 它们仍然有历史价值，所以不直接删除

## 3.什么代码**不**该放进archive

下面这些仍然属于当前主线，不应该继续往archive里塞：

- `src/demixing/data/`
  里的主线预处理、端元、合成真值代码

- `src/demixing/evaluation/classical_unmixing.py`
  主线经典解混实现

- `src/demixing/visualization/classical_unmixing.py`
  主线可视化

- `scripts/data/generate_synthetic_unmixing_dataset.py`
  主线合成真值入口

- `scripts/experiments/nnls_unmixing/`
  当前主线实验入口

## 4.当前使用建议

如果你现在在做论文主线、写实验、继续开发，请优先看：

- `README.md`
- `docs/nnls_unmixing_flow.md`
- `scripts/experiments/nnls_unmixing/`
- `src/demixing/data/`
- `src/demixing/evaluation/classical_unmixing.py`

只有在下面这些场景，才需要回头看archive：

- 想查旧结果是怎么来的
- 想复盘项目历史路线
- 想从旧路线里借一些可复用思路

## 5.一句话总结

`archive/`不是垃圾桶，而是历史代码陈列室。

主线继续往前冲，旧路线老老实实待在这里，互不打架，这才像个像样的项目。
