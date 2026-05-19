# PRISM — Physics-Regularized Iterative Spectral Mixing

> 为拉曼面扫数据像素级解混设计的非负线性解混框架。在经典 NNLS 基础上引入波段加权、L2 Tikhonov 正则化与空间 TV 一致性约束，针对拉曼数据的物理先验定制。

---

## 1. 方法位置

PRISM 是 NNLS 的三层扩展：

```
NNLS:     min ||Aw - y||²                                  s.t. w >= 0
                ↓ 加波段加权 W
WNNLS:    min ||W^{1/2}(Aw - y)||²                          s.t. w >= 0
                ↓ 加 L2 Tikhonov
WL2NNLS:  min ||W^{1/2}(Aw - y)||² + λ_L2 ||w||²            s.t. w >= 0
                ↓ 加空间 TV anchor 迭代
PRISM:    上式 + λ_a ||w - a||²，a = TV-Chambolle(W_map)；迭代 T 步
```

NNLS 是 PRISM 的退化形式：当 `weight_mode="uniform", λ_L2=0, tv_iters=0` 时两者等价。

---

## 2. 数学公式

像素 y_i ∈ R^P，端元库 A ∈ R^{P×K}：

### 2.1 PRISM-core（无空间约束）

```
w_i* = argmin_{w >= 0}  ||W^{1/2}(Aw - y_i)||² + λ_L2 ||w||²
```

通过增广矩阵转标准 NNLS：

```
A_aug = [W^{1/2} A ; sqrt(λ_L2) · I_K]
y_aug = [W^{1/2} y_i ; 0_K]
w_i* = nnls(A_aug, y_aug)
```

### 2.2 PRISM-full（加空间 TV anchor 迭代）

记 W_map = reshape({w_i}, H, W, K)。迭代 T 步，每步对每个端元通道做 2D TV-Chambolle 平滑得到 anchor，把 `||w - a_i||²` 作为额外二次项加入增广矩阵：

```
A_aug = [W^{1/2} A ; sqrt(λ_L2) I_K ; sqrt(λ_a) I_K]
y_aug = [W^{1/2} y_i ; 0 ; sqrt(λ_a) · a_i]
```

其中 `λ_a = λ_TV × λ_anchor_scale`（默认 5.0）。

---

## 3. 默认超参（v1）

```python
prism_unmix_spectra(
    spectra, library,
    image_shape=(H, W),
    weight_mode="uniform",
    lambda_l2=1e-2,
    lambda_tv=0.10,
    tv_iters=2,
    lambda_anchor_scale=5.0,
)
```

选型依据见 §6 参数扫描总结。

---

## 4. PRISM vs NNLS 实验结果

| 维度 | 数据集 | NNLS | PRISM | 改进 |
|---|---|---|---|---|
| 丰度 MAE | 合成 NOISY 40×40 | 0.0515 | 0.0441 | ↓14% |
| 丰度 Pearson r | 合成 NOISY | 0.275 | 0.383 | ↑39% |
| 空间 TV | 合成 NOISY | 0.0569 | 0.0247 | ↓57% |
| 重构 RMSE | 真实 PE+淀粉 | 0.0077 | 0.0077 | 持平 |
| 真实空间 TV | 3 真实样本 | 0.0145~0.0186 | 0.0051~0.0111 | ↓ 41~65% |
| 假阳性 max(PE) | 真实 PP+淀粉 | 11.75% | 9.52% | ↓19% |
| 假阳性 frac > 10% | 真实 PP+淀粉 | 0.13% | **0.0%** | 彻底消除 |

完整数据见 `outputs/experiments/prism_*`。

---

## 5. 加权策略消融

`weight_mode` 有两个候选，差异 **强烈依赖数据集端元的相互独立性**：

| 指标 | `uniform`（v1 默认）| `endmember_std` |
|---|---|---|
| 合成 NOISY MAE | 0.0441 | **0.0283** ✓ |
| 真实 PP+淀粉 mean(PE_absent) | 7.94% | 11.06% |
| 真实 PP+淀粉 frac > 10% | **0.0%** ✓ | **97.3%** ❌ |
| 真实重构 RMSE | 持平 NNLS | 升 4% |

### 物理解释

- **合成数据**：端元谱 = 真值端元，无谱重叠歧义 → `endmember_std` 给"差异波段"高权重，把端元区分得更开 → MAE 进一步降 36%
- **真实数据**：PE / PP 都是聚烯烃，CH₂ 弯曲峰 1300/1450 cm⁻¹ 高度重叠，PE 端元在差异波段基线偏高 → `endmember_std` 放大此偏置并经 TV 扩散，导致 PP+淀粉 样本 97% 像素假预测含 PE > 10%
- **`uniform`** 等权回归经典 NNLS 数据项，不放大端元歧义偏置

### 推荐
- **`uniform`**：真实拉曼数据（默认）
- **`endmember_std`**：端元谱独立的合成数据 / 受控实验

---

## 6. 参数扫描总结

合成 NOISY 40×40 上扫了 34 个配置（lambda_l2 × lambda_tv × tv_iters × weight_mode）：

- `lambda_l2`：1e-5 ~ 1e-2 区间影响小（MAE 差 4%），1e-2 一致最优
- `lambda_tv`：拐点在 0.10（noisy MAE 0.029，与 0.20 接近但不过度平滑小信号）
- `tv_iters`：单次收益最大，2 次接近平台，3+ 过度平滑（Pearson r 反降）
- `weight_mode`：见 §5

完整网格见 `outputs/experiments/prism_param_sweep/prism_param_sweep_full.csv`。

---

## 7. 局限性

1. **不修正端元谱测量偏差**：PRISM 的正则项不修复端元谱本身的测量误差
2. **端元谱重叠场景下丰度精度提升有限**：真实数据 PE/PP 谱重叠场景下 MAE 提升仅 14%，远不如端元独立场景下的 45%
3. **TV 平滑可能损失小颗粒信号**：λ_TV 过大时小于平滑核尺寸的微颗粒被平均掉——v1 的 λ_TV=0.10 是经验上不损失颗粒的上限
4. **超参与数据集相关**：v1 默认是真实场景的稳健折衷，受控实验场景下应单独调参

---

## 8. 代码使用

```python
from preprocessing.endmembers import build_default_endmember_library
from unmixing.unmix import prism_unmix_spectra

library = build_default_endmember_library(include_components=("PE", "PP", "starch"))

result = prism_unmix_spectra(
    spectra=spectra_normalized,     # (N_pixels, n_points) L2-normalized
    library=library,
    image_shape=(40, 40),           # mapping H × W; None disables TV
)
# result.abundances     (N_pixels, K) row-normalised
# result.coefficients   (N_pixels, K) raw PRISM coefs
# result.reconstructed  (N_pixels, n_points)
# result.residual_rmse  (N_pixels,) per-pixel residual
# result.config         hyperparameter dict used
```

---

## 9. 相关脚本

- `unmixing/unmix.py::prism_unmix_spectra` — 主实现
- `experiments/run_prism_quick_check.py` — 合成数据快速验证
- `experiments/run_prism_abundance_viz.py` — 丰度图对比可视化
- `experiments/run_prism_param_sweep.py` — 超参网格扫描
- `experiments/run_prism_real_check.py` — 真实样本对比
- `experiments/run_prism_absent_check.py` — absent_load 物理一致性测试
- `experiments/run_prism_synth_std_vs_uni.py` — 加权策略消融

---

## 10. 命名由来

PRISM = **P**hysics-**R**egularized **I**terative **S**pectral **M**ixing。

棱镜（prism）是光谱学的经典分光意象，对应本方法把混合光谱 **物理上分解** 为非负端元贡献的核心目标；"Physics-Regularized" 强调三个正则项（波段加权、L2 Tikhonov、空间 TV）都源自拉曼面扫数据的物理先验，而非通用数据先验。
