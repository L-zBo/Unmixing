# 摘要草稿 — 食品基质中聚烯烃微塑料的拉曼像素级解混（PRISM）

> 本文档是论文摘要的演进草稿，所有 WP-2 / WP-5 数字已回填，WP-3 深度方法对比按"仅毕业不投期刊"口径永久留 future work。仅保留 `[TBD: 调研]` 标记待论文最终稿前补齐文献卷号。
>
> 关键词、英文版、引用文献集中在文末。

---

## 1. 中文摘要 — 200 字版（主推，PPT/开题用）

食品基质（特别是淀粉类）中聚烯烃微塑料（PE/PP）的拉曼定量检测面临**淀粉荧光本底叠加 PE/PP 在 CH₂ 弯曲峰区（1300~1450 cm⁻¹）高度重叠**的挑战，常规单像素光谱匹配方法在该场景失效。本文提出 **PRISM**（Physics-Regularized Iterative Spectral Mixing），在非负最小二乘（NNLS）基础上引入波段加权、L2 Tikhonov 正则化与空间 TV-Chambolle 一致性约束，针对拉曼面扫数据的物理先验定制；NNLS 是 PRISM 在 `λ_L2=0, tv_iters=0, weight_mode=uniform` 下的退化形式。在合成真值数据与 3 个真实拉曼面扫样本上，对比 **OLS / NNLS / FCLS / NMF / MCR-ALS-hard / MCR-ALS-semi** 共六种基线方法。PRISM 在 7 方法 4 个核心指标（MAE / RMSE / Pearson r / spatial_TV）上全场最佳；真实数据 spatial_TV 较 NNLS 降 **37~65%**，PP+淀粉 样本"不应有 PE"的假阳性像素比例（frac>10%）从 **0.125% 彻底降至 0%**，重构 RMSE 持平 NNLS（差 <0.0005）；合成 NOISY 数据 MAE 降 14%、Pearson r 升 39%。**值得注意**：半盲 MCR-ALS 与 NMF 在端元谱重叠场景下双双 Pearson r ≈ 0，反向证明"已知端元 + 物理正则"主线选择正确。本框架为食品基质微塑料定量检测提供物理可解释的解混工具。

字数：约 230 字。

---

## 2. 中文摘要 — 300 字版（论文正式版，留扩展空间）

### 2.1 背景痛点（约 60 字）

食品基质（特别是淀粉类食品与包装相关样品）中聚烯烃微塑料（PE / PP）的拉曼定量检测，是食品安全与公共健康领域的紧迫议题。然而**淀粉强荧光本底**叠加 PE 与 PP 在 CH₂ 弯曲峰区（1300~1450 cm⁻¹）的高度重叠，使常规单像素光谱匹配方法在该三组分体系下失效。

### 2.2 文献空白（约 40 字）

现有拉曼解混研究集中于水体、血清与沉积物基质（代表性工作如 Dong 2024 水体微塑料综述、Sobczyk 2023 二组分聚烯烃解混 —— 卷号待论文最终稿前文献检索补全），未见 PE + PP + 淀粉 三组分体系的系统对照与方法层面的物理约束创新。

### 2.3 本文方法（约 70 字）

本文提出 **PRISM**（Physics-Regularized Iterative Spectral Mixing）框架——在经典非负最小二乘（NNLS）基础上系统引入三层物理正则：**波段加权**（默认 uniform，规避端元重叠场景下的歧义偏置）、**L2 Tikhonov 正则化**（λ=10⁻²，抗共线性）与**空间 TV-Chambolle 一致性约束**（λ_TV=0.10，2 步迭代，利用拉曼面扫的二维空间先验）。NNLS 是 PRISM 在 `λ_L2=0, tv_iters=0, weight_mode=uniform` 下的退化形式，便于公平基线消融。

### 2.4 实验对照（约 50 字）

在合成真值数据（40×40 像素 PE/PP/淀粉 三组分线性混合，含噪与干净两套）与 3 个真实拉曼面扫样本（PE+淀粉 / PP+淀粉 / PE+PP+淀粉 test）上，对比 **OLS / NNLS / FCLS / NMF / MCR-ALS-hard / MCR-ALS-semi** 共六种基线方法（深度学习方法 SUnCNN / CyCU-Net 作为 PRISM-DL 扩展留 future work）。

### 2.5 关键结果（约 80 字）

PRISM 在真实数据上 **spatial_TV 相对 NNLS 降低 41~65%**、PP+淀粉 样本中"不应有 PE"的假阳性像素比例（frac>10%）从 NNLS 的 **0.13% 彻底降至 0.0%**，重构 RMSE 完美持平 NNLS（**0.0072 vs 0.0070**，无拟合谱代价）。合成 NOISY 40×40 数据上 PRISM 的丰度 MAE 相对 NNLS 降 14%（**0.0515 → 0.0441**）、Pearson r 升 39%（0.275 → 0.383）。本框架为食品基质微塑料定量检测提供物理可解释、空间一致、假阳性可控的解混工具，并作为后续 PRISM + 深度学习的物理先验承下。

字数：约 320 字（含小标题切分可压缩到 280）。

---

## 3. 关键词

**中文**：拉曼光谱、微塑料、聚乙烯、聚丙烯、淀粉、像素级光谱解混、非负最小二乘、物理正则化、空间 TV 约束

**English**：Raman spectroscopy; microplastics; polyethylene; polypropylene; starch matrix; pixel-level spectral unmixing; non-negative least squares; physics-regularized; spatial total variation

---

## 4. English Abstract — 200-word version

Quantitative Raman detection of polyolefin microplastics (PE / PP) in food matrices, particularly starch-based samples, is challenged by strong starch fluorescence background and severe overlap of PE/PP fingerprints in the CH₂ bending region (1300–1450 cm⁻¹), where conventional single-pixel matching approaches fail. We propose **PRISM** (Physics-Regularized Iterative Spectral Mixing), which extends classical non-negative least squares (NNLS) with three physics-motivated regularizers: band weighting, L2 Tikhonov regularization, and a spatial TV-Chambolle anchor iteration tailored to Raman hyperspectral mapping. NNLS is recovered as a degenerate case of PRISM (`λ_L2=0, tv_iters=0, weight_mode=uniform`). On synthetic ground-truth (40×40 PE/PP/starch) and three real Raman mapping samples, we compare against **six baselines: OLS / NNLS / FCLS / NMF / MCR-ALS-hard / MCR-ALS-semi**; deep learning baselines (SUnCNN / CyCU-Net) are left as future work for the PRISM-DL extension. PRISM achieves the best score on four core metrics (MAE / RMSE / Pearson r / spatial_TV) among all seven methods; on real data, the spatial total variation of abundance maps drops **37–65%** versus NNLS, the false-positive rate (fraction with PE > 10% in PP+starch samples) is eliminated from **0.125% to 0.0%**, with no reconstruction RMSE penalty (<0.0005). On noisy synthetic data, abundance MAE drops 14% and Pearson r rises 39%. Notably, semi-blind MCR-ALS and NMF both yield Pearson r ≈ 0 under fingerprint overlap, reverse-validating our supervised-endmember design. The framework provides a physically interpretable, spatially coherent, and false-positive-controlled tool for food-matrix microplastic quantification.

约 215 词。

---

## 5. 数字单一事实源（写论文/PPT 都引这几个文件）

| 数字 | 出处 |
|---|---|
| OLS 最坏 76.5% 负丰度 | `outputs/showcase/method_constraints/negative_coef_fraction_bars.png` (v13) |
| NMF 端元 SAM 0.40 rad | `outputs/showcase/method_constraints/nmf_endmember_sam_bars.png` (v13) |
| NNLS / FCLS 合成 MAE / RMSE / R² | `outputs/showcase/synthetic_truth/synthetic_method_comparison_summary.csv` (v9) |
| PRISM vs NNLS 真实 spatial_TV 41~65% | `outputs/experiments/prism_real_check/*_metrics.csv` |
| PRISM PP+淀粉 frac>10% 从 0.13% → 0% | `outputs/experiments/prism_absent_check/prism_absent_check_summary.csv` |
| PRISM 合成 NOISY MAE 0.0515 → 0.0441 | `outputs/experiments/prism_quick_check_formal_v1/prism_quick_check_summary.csv` |
| PRISM 加权策略消融 | `outputs/experiments/prism_synth_std_vs_uni/prism_synth_std_vs_uni_summary.csv` |
| 全维度 9 行 7 维度汇总 | `outputs/showcase/method_comparison/method_overall_summary.csv` |
| **7 方法 × 7 指标合成总表（论文表 1 源）** | `outputs/showcase/method_comparison/seven_method_synth_summary.csv` |
| **PRISM vs NNLS 跨真实样本+absent_load 汇总（表 2/3 源）** | `outputs/showcase/method_comparison/prism_real_cross_sample.csv` |
| PRISM tv_iters trade-off 收敛曲线 | `outputs/showcase/prism_convergence/prism_tv_iters_tradeoff.png` |
| PRISM 完整方法说明 | `docs/prism_method.md` |
| 论文级读图说明 + 物理含义 | `docs/thesis_chapter1_figures.md` |

---

## 6. 待回填清单（按工作包追溯）

| 占位标记 | 内容 | 来源 |
|---|---|---|
| ✅ ~~`[TBD: WP-2]`~~ | MCR-ALS 在合成数据上的 MAE / RMSE / spatial_TV / 假阳性等指标 | **已回填**：`outputs/showcase/method_comparison/seven_method_synth_summary.csv` 含 7 方法 7 指标 |
| 永久 future work | SUnCNN / CyCU-Net 深度方法对比（按"仅毕业"口径不做） | 留 PRISM + 深度学习大论文 v2 |
| ✅ ~~`[TBD: WP-5]`~~ | 跨真实样本 + absent_load 汇总数字 | **已回填**：`outputs/showcase/method_comparison/prism_real_cross_sample.csv` |
| `[TBD: 调研]` | Dong 2024 / Sobczyk 2023 等"水体 / 血清基质"代表性文献精确引用 | 论文最终稿前文献检索补全 |

---

## 7. 引用文献（摘要相关，与 thesis_chapter1_figures.md §8 同源）

### 解混经典方法

- Heinz, D. C., & Chang, C. I. (2001). Fully constrained least squares linear spectral mixture analysis method for material quantification in hyperspectral imagery. _IEEE TGRS_.
- Bioucas-Dias, J. M., et al. (2012). Hyperspectral unmixing overview: Geometrical, statistical, and sparse regression-based approaches. _IEEE JSTARS_.

### PRISM 方法理论基础

- Bro, R., & De Jong, S. (1997). A fast non-negativity-constrained least squares algorithm. _Journal of Chemometrics_, 11(5):393-401.
- Iordache, M.-D., Bioucas-Dias, J. M., & Plaza, A. (2011). Sparse Unmixing of Hyperspectral Data. _IEEE TGRS_, 49(6):2014-2039.
- Iordache, M.-D., Bioucas-Dias, J. M., & Plaza, A. (2012). Total Variation Spatial Regularization for Sparse Unmixing. _IEEE TGRS_, 50(11):4484-4502.
- Chambolle, A. (2004). An algorithm for total variation minimization and applications. _Journal of Mathematical Imaging and Vision_, 20:89-97.

### 拉曼预处理协议

- Eilers, P. H. C., & Boelens, H. F. M. (2005). Baseline correction with asymmetric least squares smoothing.
- Spectroscopy Online — Key Steps in the Workflow to Analyze Raman Spectra.

### 微塑料拉曼检测背景（论文最终稿前文献检索补全）

- Dong, X. et al. (2024). [待论文写作前补全卷号] — 水体微塑料拉曼综述代表作
- Sobczyk, M. et al. (2023). [待论文写作前补全卷号] — 二组分聚烯烃 PE/PP 解混代表作
- NMF 综述（含 SAD/RMSE benchmark） — PMC9319907.
