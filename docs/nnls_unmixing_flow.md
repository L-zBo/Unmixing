# 解混项目代码落实说明（PRISM 主线 + 经典对比基线）

> 本文档是项目当前主线方法 **PRISM**（Physics-Regularized Iterative Spectral Mixing，在 NNLS 基础上引入波段加权、L2 Tikhonov 正则化与空间 TV 一致性约束）的统一落地口径。**NNLS 同时是 PRISM 的退化形式**（`weight_mode="uniform", λ_L2=0, tv_iters=0`），并与 `CLS/OLS / FCLS / NMF / MCR-ALS` 一起构成对比基线。仓库已切换到「按职责扁平铺开」的结构，主线代码分布在 `preprocessing/` `synthetic/` `unmixing/` `visualization/` `experiments/` `utils/` 六个顶层目录；早期分类路线见 [`legacy_classification_flow.md`](legacy_classification_flow.md)，对应代码与脚本整体归档到 `archive/legacy_classification/`；PRISM 方法的完整数学公式 / 参数扫描 / 加权策略消融见 [`prism_method.md`](prism_method.md)。

## 1.本文档的定位

这份文档不是论文章节草稿，而是后续项目代码落实的统一口径。当前先把路线定死，避免后面一边写代码一边改目标，最后把自己绕进沟里。

当前结论如下：

- 项目主线方法：`PRISM`（NNLS + 波段加权 + L2 Tikhonov + 空间 TV anchor 迭代）
- NNLS 退化基线：`NNLS`（即 PRISM 在 `weight_mode=uniform, λ_L2=0, tv_iters=0` 时的退化）
- 有监督对比方法：`CLS/OLS`、`FCLS`、`MCR-ALS`（端元锁死 = NNLS；半盲 = 端元漂移灾难的反向证据）
- 盲解混对比方法：`NMF`
- 后续扩展方向：`PRISM + 深度学习`（PRISM 输出作为深度方法的物理先验，本仓库不实现深度方法本身）
- 数据原则：原始`dataset/`只读，不覆盖、不移动、不提交到远程仓库

## 2.当前数据集理解

基于当前私有数据工作区，已确认数据结构如下：

- `dataset/PE纯谱/采集图谱.csv`
- `dataset/PP纯谱/采集图谱.csv`
- `dataset/淀粉纯谱/玉米淀粉/DATA-105635-X0-Y30-8884.csv`
- `dataset/PE+淀粉/`
- `dataset/PP+淀粉/`
- `dataset/test/PE+淀粉/`
- `dataset/test/PP+淀粉/`
- `dataset/test/PP+PE+淀粉/`
- `dataset/泛化/展艺玉米淀粉/`
- `dataset/泛化/新良小麦淀粉/`
- `dataset/泛化/甘汁园小麦淀粉/`

当前已确认的事实：

- 每条拉曼谱是2列csv，首列是波数轴，第二列是强度
- 每条谱统一为1025行，波数范围约为`103.785~3696.98cm^-1`
- 混合样本子目录中每组有1600条谱，对应`40×40`拉曼面扫
- `PE+淀粉`和`PP+淀粉`是二元混合
- `dataset/test/PP+PE+淀粉/`是三元混合
- `dataset/泛化/`用于跨淀粉来源泛化验证

## 3.解混任务的核心定义

这里必须先说死：解混输出的本质不是标签，而是各端元的非负丰度系数。

以三组分为例，单个像素的输出应为：

`[a_PE, a_PP, a_淀粉]`

其中：

- `a_PE >= 0`
- `a_PP >= 0`
- `a_淀粉 >= 0`

如果做`FCLS`，还要求：

`a_PE + a_PP + a_淀粉 = 1`

“标签”只是丰度结果的派生形式，例如：

- 取最大丰度对应的主导组分
- 判断某个组分是否存在
- 根据阈值生成空间分布掩膜

所以后续代码和实验记录里，必须把“丰度图”和“标签图”分开，不允许混着叫。

## 4.为什么以PRISM为主线（NNLS 是其退化形式）

`PRISM` 是当前项目最合适的主线方法，理由如下：

- **物理可解释性**：NNLS 已经满足"非负 + 经典物理解释"，PRISM 在此之上叠加三层物理正则（波段加权 / L2 Tikhonov / 空间 TV），针对拉曼面扫数据的物理先验定制
- **真实数据假阳性消除**：PRISM 在 PP+淀粉 样本上把假阳性 frac>10% 从 NNLS 的 0.13% 降到 0.0%，spatial_TV ↓ 41~65%，重构 RMSE 完美持平
- **NNLS 是 PRISM 的退化形式**：`weight_mode="uniform", λ_L2=0, tv_iters=0` 时两者等价，**天然可消融**，论文论证逻辑严密
- **为后续 `PRISM + 深度学习` 铺垫**：PRISM 输出的物理可解释、空间平滑非负丰度可直接作为深度模型的物理先验或伪标签

同时也要明确边界：

- PRISM 仍然要求端元矩阵 `A` 已知（与 NNLS 一致）
- PRISM 不修正端元谱本身的测量偏差
- 端元谱重叠场景下丰度精度提升有限（真实 PE/PP 谱重叠场景 MAE 提升 14%，端元独立场景 36%）
- TV 平滑 λ_TV 过大会损失小颗粒信号（v1 默认 λ_TV=0.10 是经验上不损失颗粒的上限）

因此本项目的方法定位应当写成：

- `PRISM` 是主线（有监督，已知端元 + 物理正则化）
- `CLS/OLS`、`NNLS`、`FCLS`、`MCR-ALS（端元锁死）` 属于有监督对比基线
- `NMF`、`MCR-ALS（半盲）` 属于盲/半盲解混对比，**两者均出现端元漂移问题**，反向证明"已知端元有监督"主线选择正确
- 若后续做半盲流程，可采用"`NMF` 估计端元 + `PRISM` 估计丰度"（暂留 future work）

## 5.真值口径必须分层

这块最容易写炸，所以代码实现时也必须分层。

### 5.1像素级真值

只有合成数据才有严格像素级真值。

做法：

- 从纯谱中选取端元
- 人工生成二维丰度图
- 用设定好的丰度线性混合端元谱
- 叠加可控噪声、基线扰动和强度缩放

这样每个像素的真实丰度都已知，可直接评价解混精度。

### 5.2样本级真值

如果后续能拿到真实配样比例记录，例如整张图的`PE:淀粉 = 3:7`，那它只属于样本级真值。

能比较的是：

- 整张面扫图的平均丰度
- 各组分总体占比

不能比较的是：

- 单个像素的真实丰度

### 5.3无像素真值的真实数据

当前真实拉曼面扫更稳妥的理解是“弱标签数据”：

- `dataset/PE+淀粉/`只说明图中应有`PE`和`淀粉`
- `dataset/PP+淀粉/`只说明图中应有`PP`和`淀粉`
- `dataset/test/PP+PE+淀粉/`只说明三者都应出现

这类数据不能拿来吹像素级丰度精度，只能用于验证：

- 光谱重构是否靠谱
- 非目标组分是否被压低
- 空间分布是否连贯
- 泛化时是否稳定

## 6.实验设计的正确分层

后续代码实验必须拆成两大块，不能混写。

### 6.1合成数据实验

目标：

- 验证`CLS/OLS`、`NNLS`、`FCLS`、`NMF`的丰度恢复能力
- 给`NNLS`提供最硬的定量证据

建议指标：

- `MAE`
- `RMSE`
- `R²`
- `SAM`
- 端元重构误差

### 6.2真实数据实验

目标：

- 验证方法在真实拉曼面扫上的可解释性和可用性
- 观察空间丰度图是否符合预期
- 验证泛化和非目标抑制能力

建议指标或现象：

- 谱重构误差
- 非目标组分平均丰度
- 空间连贯性
- 同类样本之间结果稳定性
- 泛化样本上的鲁棒性

结论口径必须统一：

- 合成数据上讲“解混精度”
- 真实数据上讲“解混有效性、可信性和泛化性”

## 7.后续代码落地建议

注意：当前仓库已收成「按职责扁平铺开」结构，主线代码全部位于顶层职责目录，不再放在 `src/demixing/` 下面。`.gitignore` 已经排除了：

- `dataset/`
- `outputs/`

因此，后续正式落代码时，建议沿用现有的扁平结构，不要另起炉灶。

### 7.1建议新增或扩展的代码位置

#### 数据与端元管理

- `preprocessing/preprocess.py`
  当前已支持 `als_l2`、`als_max`、`none_l2` 三种协议
- `preprocessing/endmembers.py`
  端元谱加载，可按预处理协议构建端元矩阵，支持多淀粉来源

#### 经典解混方法

- `unmixing/unmix.py`
  集中实现 `CLS/OLS`、`NNLS`、`FCLS` 与盲 `NMF`，统一输出：

  - 每像素丰度矩阵
  - 每像素重构谱
  - 每像素重构误差
  - 每张面扫图的丰度图
  - 样本级汇总表

#### 合成数据生成

- `synthetic/generator.py` + `synthetic/generate_dataset.py`
  用端元谱生成有像素级真值的二维混合数据，输出到 `outputs/synthetic_unmixing/`

#### 正式实验脚本

- `experiments/run_synthetic_method_comparison.py`
  合成真值数据上的四方法对比
- `experiments/run_real_unmixing_single.py`
  真实数据上的解混、重构和空间图输出

实验产物默认输出到 `outputs/experiments/`、`outputs/<scenario>/` 等场景目录。

#### 可视化

- `visualization/`
  按图类型分子目录（abundance / residual / reconstruction / method_comparison / preprocessing），顶层 re-export 14 个绘图函数

## 8.方法实现顺序

后续代码不要一口气全冲，按这个顺序推进最稳。

### 第一阶段：把NNLS主线立住

任务：

- 完成端元矩阵加载
- 完成`NNLS`像素级求解
- 输出丰度图和重构图
- 在`PE+淀粉`、`PP+淀粉`真实数据上先跑通

验收：

- 能稳定输出二维丰度图
- 非目标组分丰度明显低
- 重构误差可统计

### 第二阶段：补同类监督对比

任务：

- 加入`CLS/OLS`
- 加入`FCLS`
- 保持统一输入输出格式

验收：

- 三种监督解混方法可直接横向对比
- 指标、表格、图像输出格式统一

### 第三阶段：构建有真值合成实验

任务：

- 生成二维丰度真值图
- 合成二元和三元混合谱
- 对四种方法做定量评估

验收：

- 能输出`MAE/RMSE/R²/SAM`
- 能画真值丰度图与预测丰度图对比

### 第四阶段：补盲解混对比

任务：

- 加入`NMF`
- 和`NNLS`主线做对比
- 观察盲解混端元与已知纯谱的一致性

验收：

- `NMF`能输出端元和丰度
- 能和监督方法放在同一报告中比较

### 第五阶段：准备和后续深度学习衔接

任务：

- 固化`NNLS`输出格式
- 使其可作为后续深度学习监督信号、伪标签或先验约束

验收：

- `NNLS`结果可被后续模型直接读取
- 丰度图、重构误差、样本汇总格式稳定

## 9.数据与结果管理原则

这部分必须像铁律一样执行。

- 不改原始数据：`dataset/`只读
- 不覆盖原始文件：所有预处理、合成、实验输出都写到新目录
- 不提交私有数据：`dataset/`不进git
- 不提交中间大结果：`outputs/`默认不进git
- 不把“真实数据弱标签”冒充“像素级真值”

建议固定使用以下输出目录：

- `outputs/preprocessing/`
- `outputs/synthetic_unmixing/`
- `outputs/experiments/classical_unmixing/`
- `outputs/experiments/figures/`

## 10.当前项目的统一说法

为了避免后续文档、代码、实验记录互相打架，先统一一句话：

"本项目以 `PRISM`（Physics-Regularized Iterative Spectral Mixing）为主线方法，在已知端元条件下开展拉曼光谱有监督解混。PRISM 在经典 NNLS 基础上引入波段加权、L2 Tikhonov 正则化与空间 TV-Chambolle 一致性约束；NNLS 同时是 PRISM 的退化形式（`weight_mode=uniform, λ_L2=0, tv_iters=0`）。与 `CLS/OLS / FCLS / MCR-ALS / NMF` 一起构成对比基线，通过合成真值数据验证丰度恢复精度，通过真实拉曼面扫数据验证解混结果的重构能力、空间一致性、物理可解释性（absent_load 假阳性）和泛化能力，为后续 `PRISM + 深度学习` 方法提供物理先验。"

这句话后面谁再乱改，就属于自己给自己挖坑。

## 11.当前已落地代码进度

截至目前，已经完成的代码落实如下：

### 11.1端元管理

已完成文件：

- `preprocessing/endmembers.py`

已具备能力：

- 加载 `PE`、`PP` 和不同来源淀粉纯谱
- 复用现有预处理流程，生成标准波数轴上的端元谱
- 构建供 `OLS`、`NNLS`、`FCLS` 使用的端元矩阵

### 11.2经典解混核心模块

已完成文件：

- `unmixing/unmix.py`

已具备能力：

- `OLS` 求解
- `NNLS` 求解
- `FCLS` 求解
- 盲解混 `NMF`
- `NMF` 结果自动对齐到参考纯谱名
- 统一输出丰度、系数、重构谱、残差指标

### 11.3可视化模块

已完成文件：

- `visualization/abundance/abundance_maps.py`
- `visualization/residual/residual_map.py`
- `visualization/reconstruction/reconstruction.py`
- `visualization/method_comparison/method_bars.py`
- `visualization/preprocessing/spectrum.py`
- `visualization/_common.py`

顶层 `visualization/__init__.py` 一次性 re-export 全部 14 个绘图函数。

已具备能力：

- 丰度热图
- 残差热图
- 输入谱与重构谱对比图
- 方法残差柱状图与方法平均丰度柱状图
- 预处理协议三视图与丰度网格

### 11.4实验脚本（截至 v15）

已完成的实验脚本（按维度分组）：

**真实面扫图主线**

- `experiments/run_real_unmixing_single.py` — 单张图跑`OLS / NNLS / FCLS / NMF`
- `experiments/run_real_method_comparison.py` — 单图四方法详细对比（含图）
- `experiments/run_batch_method_comparison.py` — 多张典型图四方法批量对比

**合成真值定量**

- `experiments/run_synthetic_method_comparison.py` — 合成真值上四方法定量对比

**预处理协议对比**

- `experiments/run_real_preprocessing_comparison.py` — 单图三协议对比
- `experiments/run_batch_preprocessing_comparison.py` — 多图三协议批量

**泛化稳定性**

- `experiments/run_generalization_batch.py` — 跨淀粉来源泛化批量评估

**PPT 证据补强（v13~v15）**

- `experiments/run_endmember_fingerprint_plot.py` — 端元纯谱 + 文献指纹峰标注
- `experiments/run_method_constraint_diagnostics.py` — 逐像素负丰度率 / `NMF`端元`SAM` / `NNLS`稀疏度
- `experiments/run_protocol_consistency_analysis.py` — 三协议下逐像素`CV` + 指纹峰保留率

通用能力：

- 输出每像素丰度表、端元表、摘要`json`
- 丰度热图、残差图、重构示例图、方法对比柱图
- 支持`--limit`小样本调试
- 主线产物镜像到`outputs/showcase/<场景>/`

### 11.5当前默认输出位置

当前新增脚本只写入新结果目录，不覆盖原始数据，主要输出到：

- `outputs/experiments/formal_v6_classical_unmixing_real/`
- `outputs/experiments/formal_v7_method_comparison_real/`
- `outputs/experiments/formal_v8_batch_method_comparison/`
- `outputs/experiments/formal_v9_synthetic_method_comparison/`
- `outputs/experiments/formal_v10_preprocessing_comparison_real/`
- `outputs/experiments/formal_v11_batch_preprocessing_comparison/`
- `outputs/experiments/formal_v12_generalization_batch/`
- `outputs/experiments/formal_v13_method_constraint_diagnostics/`
- `outputs/experiments/formal_v14_protocol_consistency/`
- `outputs/experiments/formal_v15_endmember_fingerprint/`
- `outputs/showcase/`（从上述目录筛出的展示型结果包）

### 11.6当前仍未完成的部分

当前`PPT`阶段（2026-05-06 起）还要补强的内容：

- 三类`v13 ~ v15`证据图的视觉小瑕疵兜底（文字交叠 / 排版细节）
- 按`PPT`选图清单从`outputs/showcase/`二次精选
- 合成真值评估指标进一步校准（归一化后丰度真值的解释口径）

## 13.补充约束：预处理与可解释性主线

这部分是后续代码和实验必须遵守的新口径，不能再按旧默认值糊过去。

### 13.1预处理主线

当前正式主线改为：

- `ALS基线校正 + L2归一化`

这意味着：

- 以后默认预处理不再以`ALS + max归一化`作为主口径
- 现有预处理代码需要改成“协议可切换”
- 后续所有解混实验都要明确写清楚使用的是哪一种预处理协议

### 13.2预处理对比要求

后续实验不能只比较解混算法，还要比较预处理协议，至少要回答：

- 为什么默认采用`ALS + L2`
- `ALS + L2`相比其他预处理协议是否更稳定
- 不同预处理协议会不会改变`NNLS/FCLS/NMF`的解混结果

建议最少纳入以下协议对比：

- `ALS + L2`
- `ALS + max`
- 无基线校正 + `L2`
- 其他可选基线/归一化组合

需要特别注意的一点：

- 一旦做了逐谱归一化，尤其是`L2`归一化，原始丰度真值与归一化后线性系数不再严格一一对应
- 所以合成真值实验里，丰度误差指标必须和重构误差、`SAM`一起解释，不能只盯着一个`MAE/RMSE`

### 13.3解混主线

解混方法的主线（2026-05-20 升级）：

- 默认解混主线：`PRISM`（NNLS + 波段加权 + L2 Tikhonov + 空间 TV）
- 退化基线：`NNLS`（PRISM 在 `weight_mode=uniform, λ_L2=0, tv_iters=0` 时的退化形式）
- 经典监督对比：`CLS/OLS`、`FCLS`、`MCR-ALS（端元锁死）`
- 盲/半盲对比：`NMF`、`MCR-ALS（半盲）`

这里的核心口径不是"谁重构误差最小就盲目选谁"，而是：

- `PRISM` 在 `NNLS` 物理可解释性基础上进一步压制噪声 / 假阳性，符合"非负 + 物理先验"双重约束
- `NMF` 与半盲 `MCR-ALS` 都出现端元漂移问题，重构 R² 看似最高但丰度跟真值零相关——反向证明"已知端元有监督"主线选择正确
- 选择 `PRISM` 是为了突出解混结果的可解释性 + 真实数据假阳性可控，而不是只追求一个黑箱指标

### 13.4后续代码落实要求

接下来必须补上的代码方向：

- 把 `preprocessing/preprocess.py` 持续保持「协议化」入口，避免硬编码
- 新增预处理协议比较脚本（`experiments/run_*_preprocessing_comparison.py` 已存在）
- 在单图、多图和合成真值实验里，把「预处理协议」和「解混算法」拆成两层对比

## 14.当前新增进展补记

在前面文档基础上，当前又新增了下面这些落实：

- 已把默认预处理主线切到 `ALS + L2`
- 旧的 `ALS + max` 没有删，继续保留作后续对比
- `preprocessing/preprocess.py` 已经支持 `als_l2`、`als_max`、`none_l2`
- `preprocessing/endmembers.py` 已经支持按预处理协议加载端元
- 已新增 `experiments/run_real_preprocessing_comparison.py`，可固定解混方法对比不同预处理协议
- 已新增 `experiments/run_batch_preprocessing_comparison.py`，可在多张典型图上批量汇总预处理对比结果
- 已新增 `synthetic/generator.py` 和 `synthetic/generate_dataset.py`，用于生成有真值的二维合成解混数据
- 已新增 `experiments/run_synthetic_method_comparison.py`，用于在合成真值数据上比较 `OLS/NNLS/FCLS/NMF`
- 合成真值评估口径已分层为 orig/proj/重构三层指标（v9 校准）
- 已新增 `experiments/run_generalization_batch.py`，用于在 `dataset/泛化/` 不同淀粉来源上做批量解混评估
- 仓库结构已整体重构：主线代码从 `src/demixing/` 与 `scripts/` 平铺到 `preprocessing/`、`synthetic/`、`unmixing/`、`visualization/`、`experiments/`、`utils/` 六个顶层目录；旧分类路线整体归档到 `archive/legacy_classification/`
- 已新增`v13 / v14 / v15`三类`PPT`证据脚本：方法约束诊断（`run_method_constraint_diagnostics.py`）、协议一致性（`run_protocol_consistency_analysis.py`）、端元指纹峰图（`run_endmember_fingerprint_plot.py`），分别对应`outputs/showcase/method_constraints / protocol_consistency / endmember_fingerprint/`三个展示子目录
- `visualization/method_comparison/`新增`constraint_diagnostics.py`、`visualization/preprocessing/`新增`endmember_fingerprint.py / protocol_consistency.py`，顶层`__init__.py`同步`re-export`，当前共`14`个绘图函数

当前还需要继续补强的重点：

- 合成真值评估指标还需要进一步校准，特别是"归一化后丰度真值"的解释口径
- 预处理对比目前先在真实数据上跑通，后续还要扩展到更多样本和更多协议

## 15.2026-05-20 PRISM 主线升级补记

在以上 NNLS 时代基础上，本仓库主线已升级为 **PRISM**（Physics-Regularized Iterative Spectral Mixing）。本节是最新进展，与第 4 / 10 / 13.3 节同时生效；前文 v6~v15 实验脚本与产物**仍然有效**作为 PRISM 的对比基线。

### 15.1新增主线方法 PRISM

- `unmixing/unmix.py::prism_unmix_spectra` — PRISM 主实现（加权 NNLS + L2 Tikhonov + 空间 TV anchor 迭代）
- `unmixing/__init__.py` — 顶层 re-export `prism_unmix_spectra` / `PrismUnmixingResult`
- v1 默认超参：`weight_mode="uniform", λ_L2=1e-2, λ_TV=0.10, tv_iters=2, λ_anchor_scale=5.0`
- 完整方法说明、参数扫描与加权策略消融见 [`prism_method.md`](prism_method.md)

### 15.2新增对比方法 MCR-ALS（拉曼社区事实标准）

- `unmixing/unmix.py::mcr_als_unmix_spectra` — 基于 `pymcr` 0.5.1 实现，支持 hard-constrained（端元锁死，等价 NNLS）与 semi-blind（端元 init 但允许微调）两种模式
- 关键发现：semi-blind MCR-ALS 在 PE/PP 谱重叠场景下端元漂移导致 Pearson r ≈ -0.015 灾难，反向证明"已知端元有监督"主线选择正确

### 15.3新增 PRISM 实验脚本（6 个）

- `experiments/run_prism_quick_check.py` — 合成数据快速验证
- `experiments/run_prism_abundance_viz.py` — 丰度图对比可视化
- `experiments/run_prism_param_sweep.py` — 34 配置超参网格扫描
- `experiments/run_prism_real_check.py` — 3 个真实样本对比
- `experiments/run_prism_absent_check.py` — absent_load 物理一致性测试（"不应有 PE" 的假阳性率）
- `experiments/run_prism_synth_std_vs_uni.py` — 加权策略消融
- `experiments/run_mcr_als_check.py` — MCR-ALS hard / semi-blind 对比

### 15.4新增 PRISM 实验产物目录

- `outputs/experiments/prism_quick_check*/` / `prism_abundance_viz*/` / `prism_param_sweep/` / `prism_real_check/` / `prism_absent_check/` / `prism_synth_std_vs_uni/`
- `outputs/experiments/mcr_als_check_formal_v1/`

### 15.5新增文档

- `docs/prism_method.md` — PRISM 完整方法说明（数学公式 + 参数扫描 + 加权策略消融 + limitation）
- `docs/thesis_abstract.md` — 论文摘要演进草稿（中 200/300 + 英 215 + 数字事实源 + WP-2/3/5 待回填清单）
- `论证总览.md` § 2 已扩为 **三个核心论点**（论点③ = PRISM 进一步降假阳性 + 空间一致性）

### 15.6 PRISM 关键数字（论证用）

- 真实数据 spatial_TV ↓ 41~65%
- 真实 PP+淀粉 假阳性 frac > 10%：**0.13% → 0.0%**（彻底消除）
- 重构 RMSE 持平 NNLS（0.0072 vs 0.0070）
- 合成 NOISY 40×40 MAE ↓ 14%（0.0515 → 0.0441），Pearson r ↑ 39%

完整数字事实源见 [`thesis_abstract.md`](thesis_abstract.md) § 5。
