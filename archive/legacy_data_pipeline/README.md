# legacy_data_pipeline 归档说明

这部分是**旧数据管线**的归档：早期为了配合家族分类路线（v1-v5）做"样本清单 + 质量过滤"的工具集，已经不属于当前 NNLS 解混主线。

## 1.归档内容

### `archive/legacy_data_pipeline/scripts/`

- `build_quality_manifest.py`
  根据预处理报告 csv 生成质量清单。
- `build_sample_manifest.py`
  在质量清单基础上构建样本级 manifest（家族分类路线训练时用）。

### `archive/legacy_data_pipeline/modules/`

- `manifest.py`
  样本 manifest 构建逻辑。
- `quality.py`
  质量阈值与过滤逻辑。

## 2.为什么归档

主线 NNLS 解混路线**完全不依赖** manifest / quality 这套工具：

- 解混在像素级直接对每条谱跑算法，不需要"先筛掉低质量样本再训练"
- 真实数据弱标签的角色是验证而非训练，不需要 manifest 划分训练/验证集

把它们留在 `src/demixing/data/` 和 `scripts/data/` 会让人误以为是主线工具。

## 3.什么时候可能用回来

如果将来 NNLS+深度学习的下一篇大论文里需要：

- 训练集/验证集划分
- 按样本质量过滤训练数据
- 弱监督标签构建

那时可以从这里把 `manifest.py` / `quality.py` 复用回去。

## 4.一句话总结

`archive/legacy_data_pipeline/` 不是垃圾箱，是**旧家族分类路线遗留的数据准备工具**——主线现在用不上，但留着以备深度学习阶段复用。
