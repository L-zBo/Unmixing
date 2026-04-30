# experiments目录说明

这个目录放的是“正式实验入口脚本”，不是底层库代码。

你可以把它理解成：

- `src/`里是零件和发动机
- `scripts/experiments/`里是按实验目的装好的整车

## 1.为什么这里还有这么多脚本

先说结论：现在主线脚本已经改成按“用途”命名，旧路线才保留`v1-v5`这类历史编号。

它们目前已经按路线分成两组：

- `[scripts/experiments/nnls_unmixing/](f:/VsCodeproject/Unmixing/scripts/experiments/nnls_unmixing)`
- `[scripts/experiments/legacy_classification/](f:/VsCodeproject/Unmixing/scripts/experiments/legacy_classification)`

也就是说：

- `legacy_classification/`是旧路线归档
- `nnls_unmixing/`是当前主线实验入口

## 2.当前主线脚本分别干嘛

### `[scripts/experiments/nnls_unmixing/run_real_unmixing_single.py](f:/VsCodeproject/Unmixing/scripts/experiments/nnls_unmixing/run_real_unmixing_single.py)`

作用：

- 在单张真实面扫图上跑一次指定方法
- 输出每像素丰度、端元表、残差和重构图

适合：

- 看某一张图到底解混成什么样

### `[scripts/experiments/nnls_unmixing/run_real_method_comparison.py](f:/VsCodeproject/Unmixing/scripts/experiments/nnls_unmixing/run_real_method_comparison.py)`

作用：

- 在单张真实面扫图上比较`OLS/NNLS/FCLS/NMF`

适合：

- 看同一张图上不同解混方法谁更稳

### `[scripts/experiments/nnls_unmixing/run_batch_method_comparison.py](f:/VsCodeproject/Unmixing/scripts/experiments/nnls_unmixing/run_batch_method_comparison.py)`

作用：

- 在多张典型真实图上批量比较`OLS/NNLS/FCLS/NMF`

适合：

- 做跨样本的总体方法比较

### `[scripts/experiments/nnls_unmixing/run_synthetic_method_comparison.py](f:/VsCodeproject/Unmixing/scripts/experiments/nnls_unmixing/run_synthetic_method_comparison.py)`

作用：

- 在有真值的合成数据上比较`OLS/NNLS/FCLS/NMF`

适合：

- 做严格的真值级方法比较

### `[scripts/experiments/nnls_unmixing/run_real_preprocessing_comparison.py](f:/VsCodeproject/Unmixing/scripts/experiments/nnls_unmixing/run_real_preprocessing_comparison.py)`

作用：

- 固定一种解混方法
- 在单张真实图上比较不同预处理协议

适合：

- 回答“为什么默认用`ALS+L2`”

### `[scripts/experiments/nnls_unmixing/run_batch_preprocessing_comparison.py](f:/VsCodeproject/Unmixing/scripts/experiments/nnls_unmixing/run_batch_preprocessing_comparison.py)`

作用：

- 固定一种解混方法
- 在多张典型真实图上批量比较不同预处理协议

适合：

- 做跨样本预处理结论

### `[scripts/experiments/nnls_unmixing/run_generalization_batch.py](f:/VsCodeproject/Unmixing/scripts/experiments/nnls_unmixing/run_generalization_batch.py)`

作用：

- 跑泛化样本批量实验

适合：

- 看不同淀粉来源下方法是不是还稳

## 3.为什么我现在没有直接删成只剩一个脚本

因为当前这些脚本虽然都属于NNLS主线，但它们负责的实验维度不同：

- 单图解混
- 单图方法比较
- 批量方法比较
- 合成真值比较
- 单图预处理比较
- 批量预处理比较
- 泛化批量比较

它们不是简单“同一个脚本不断改名留下来的垃圾”，而是不同实验入口。

如果现在硬删成只剩一个脚本，会带来两个问题：

- 参数会爆炸，最后一个脚本像变形金刚一样难维护
- 你回头找某类实验入口时更难找

所以当前更合理的做法不是乱删，而是：

- 保留少量职责明确的入口脚本
- 用这份README把职责解释清楚

## 4.旧路线脚本为什么保留

`[scripts/experiments/legacy_classification/](f:/VsCodeproject/Unmixing/scripts/experiments/legacy_classification)`里放的是旧分类路线：

- `run_formal_v1.py`
- `run_formal_v2.py`
- `run_formal_v3_family_svc.py`
- `run_formal_v4_group_spatial.py`
- `run_formal_v5_spatial_cnn.py`
- `run_external_test_family_svc.py`

它们现在的定位不是继续扩写，而是：

- 作为历史归档
- 作为旧结果来源说明
- 作为和当前NNLS主线对照的背景材料

## 5.如果以后还想再进一步收口

后续可以做，但建议在实验结论稳定以后再动。

可以考虑的收口方式：

1. 把`run_real_unmixing_single.py`和`run_real_method_comparison.py`再往一个更强总入口合并
2. 把`run_real_preprocessing_comparison.py`和`run_batch_preprocessing_comparison.py`再往一个更强总入口合并
3. 保留`run_synthetic_method_comparison.py`作为合成真值专用入口
4. 保留`run_generalization_batch.py`作为泛化专用入口

这样会比“现在立刻只留一个脚本”更稳。

## 6.一句话结论

现在`experiments/`目录已经不是杂乱堆积状态，而是：

- 主线实验归到`nnls_unmixing/`
- 旧路线实验归到`legacy_classification/`
- 用README说明每个入口脚本的职责

这比单纯删除文件更安全，也更容易后续维护。
