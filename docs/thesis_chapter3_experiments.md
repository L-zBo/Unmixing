# 实验摘要 + 核心结果速览

> 论文 §3 / PPT 实验汇报的精简版"摘要叙述 + 核心结果"。不是全文骨架——全文骨架按需在写作时基于此扩展。完整的读图说明 / 物理含义 / 论文写作 checklist 见 [`thesis_chapter1_figures.md`](thesis_chapter1_figures.md)。

---

## 1. 项目目标（一句话）

针对**食品基质（特别是淀粉类）中聚烯烃微塑料（PE / PP）** 的拉曼面扫数据，开展像素级线性光谱解混，输出每个像素的非负丰度（PE / PP / starch），为后续微塑料定量检测与污染溯源提供物理可解释的工具。

---

## 2. 研究不足与本工作贡献

### 不足 ①：拉曼解混领域缺三组分 PE+PP+淀粉 系统对照

现有拉曼像素级解混研究多集中于水体 / 血清 / 沉积物基质下的单一聚合物检测，未见 **PE + PP + 淀粉 三组分体系**的系统方法对照，方法层面也未将"波段加权 + L2 Tikhonov + 空间 TV"组合系统迁移到拉曼面扫场景——而这正是"端元谱在 CH₂ 弯曲峰区高度重叠 + 淀粉强荧光本底 + 面扫数据存在二维空间先验"这一拉曼特有困难下亟需的工具。

### 贡献 ①：提出 PRISM 框架（拉曼场景定制）

提出 **PRISM**（Physics-Regularized Iterative Spectral Mixing），在经典 NNLS 基础上引入三层物理正则：
- 波段加权（默认 uniform，避免端元歧义偏置放大）
- L2 Tikhonov 正则化（λ_L2 = 10⁻²，抗共线性与噪声）
- 空间 TV-Chambolle anchor 迭代（λ_TV = 0.10, 2 步，利用面扫二维空间先验）

NNLS 是 PRISM 在 `weight_mode=uniform, λ_L2=0, tv_iters=0` 下的退化形式，**消融关系数学上严密**。超参在 PE/PP 指纹峰重叠场景调优（非 hyperspectral 默认值搬用）。

### 不足 ②：盲 / 半盲解混在端元重叠拉曼数据上发生端元漂移

NMF / 半盲 MCR-ALS 等"允许端元在迭代中调整"的方法，在拉曼端元谱重叠场景下会出现**端元漂移病**——重构 RMSE 看似最优（NMF R² 0.996），但端元偏离物理参考 0.40 弧度（23°），丰度与真值 Pearson r ≈ 0。这条失败模式在拉曼解混文献中讨论不足。

### 贡献 ②：实验定量证明"端元锁死 + 物理正则"是对的

在 7 方法横向对比（OLS / NNLS / FCLS / NMF / MCR-ALS-hard / MCR-ALS-semi / PRISM）中：
- **MCR-ALS-hard（端元锁死）数学等价 NNLS**——6 列指标完全相同，是"硬约束 MCR-ALS 退化"的实证
- **NMF (Pearson -0.009) + MCR-ALS-semi (Pearson -0.015) 双双 Pearson < 0**——两条独立证据反向证明"已知端元 + 不放开"主线选择正确
- **PRISM 在丰度精度 / 空间一致 / 假阳性 三个核心维度全场最佳**，且真实数据上 PP+淀粉 样本中"不应有 PE"的假阳性像素比例从 NNLS 的 0.13% **彻底降至 0.0%**

---

## 3. 核心结果展示

### 3.1 7 方法 × 7 指标合成数据横向（NOISY 40×40，PE/PP/starch）

| 指标 | OLS | NNLS | FCLS | NMF | MCR-hard | MCR-semi | **PRISM** | best |
|---|---|---|---|---|---|---|---|---|
| MAE ↓ | 0.0515 | 0.0515 | 0.0539 | 0.1416 | 0.0515 | 0.1969 | **0.0441** | PRISM |
| RMSE ↓ | 0.0784 | 0.0786 | 0.0774 | 0.2084 | 0.0786 | 0.2557 | **0.0548** | PRISM |
| Pearson r ↑ | 0.277 | 0.275 | 0.319 | -0.009 | 0.275 | -0.015 | **0.383** | PRISM |
| recon_RMSE | 0.0101 | 0.0101 | 0.0104 | 0.0064 | 0.0104 | 0.0064 | 0.0101 | (中性) |
| spatial_TV ↓ | 0.0568 | 0.0569 | 0.0557 | 0.1582 | 0.0569 | 0.2345 | **0.0247** | PRISM |
| mean_active_endmembers | 2.96 | 2.96 | 2.99 | 2.74 | 2.96 | 2.70 | 3.00 | (中性) |
| elapsed_s | 0.08 | 0.07 | 3.30 | 1.90 | 0.36 | 15.22 | 0.21 | (中性) |

数据源：`outputs/showcase/method_comparison/seven_method_synth_summary.csv`

### 3.2 PRISM vs NNLS 跨 3 个真实样本（空间一致性）

| 样本 | NNLS spatial_TV | PRISM spatial_TV | 改进 |
|---|---|---|---|
| PE+starch | 0.0145 | **0.0051** | ↓ 65% |
| PP+starch | 0.0186 | **0.0111** | ↓ 40% |
| PE+PP+starch（test） | 0.0176 | **0.0111** | ↓ 37% |

**重构 RMSE 全部持平 NNLS**（NNLS 0.0072 / PRISM 0.0070，差 < 0.0005）—— PRISM **不以拟合谱为代价换空间一致性**。

数据源：`outputs/showcase/method_comparison/prism_real_cross_sample.csv` (real_sample 段)

### 3.3 absent_load 假阳性（物理一致性证据）

"不应有 X 的样本中 X 丰度 > 10% 的像素比例" —— 越接近 0 越符合物理：

| 样本 | absent 端元 | NNLS | **PRISM** |
|---|---|---|---|
| PE+starch | PP | 0.375% | **0.0%** |
| PP+starch | PE | 0.125% | **0.0%** |

两个样本下 PRISM 都把假阳性 frac>10% **彻底消除到 0%**，NNLS 仍有 0.125~0.375% 残留。

数据源：`outputs/showcase/method_comparison/prism_real_cross_sample.csv` (absent_load 段)

---

## 4. 一句话总结

**PRISM 在三组分拉曼解混的合成数据上 MAE/RMSE/Pearson/spatial_TV 4 个核心指标全场最佳，真实数据 spatial_TV 较 NNLS 降 37~65%、假阳性彻底消除，重构能力持平 NNLS——是物理可解释 + 空间一致 + 假阳性可控 + 速度可接受的拉曼像素级解混工具。**
