# Unmixing 解混模块

本目录包含项目主线的**经典解混算法实现**——已知端元条件下的有监督解混 + NMF 盲解混对比。

## 📁 文件列表

| 文件名 | 功能 | 状态 |
|--------|------|------|
| `unmix.py` | OLS / NNLS（主线）/ FCLS / NMF 全套实现 | 🚧 C4 搬入 |
| `README.md` | 本文档 | ✅ |

---

## 🎯 主线方法定位

| 方法 | 类型 | 角色 |
|------|------|------|
| `NNLS` | 有监督 + 非负 | **主线**——丰度非负、可解释、为后续 NNLS+DL 大论文铺路 |
| `OLS / CLS` | 有监督，无约束 | 同类对比基线 |
| `FCLS` | 有监督 + 非负 + 和为 1 | 同类对比（约束最强） |
| `NMF` | 盲解混 | 端元未知场景对比组 |

---

## 📊 输入输出口径

### 输入
- 预处理后的光谱矩阵 `X`：shape `(n_pixels, n_points)`
- 端元矩阵 `A`：shape `(n_points, n_endmembers)`

### 输出（统一 `UnmixingResult` 数据结构）
- `abundances`：每像素丰度 shape `(n_pixels, n_endmembers)`
- `coefficients`：解出的线性系数（FCLS 下等于 abundances，OLS 可能含负值）
- `reconstructed`：重构光谱
- `residual_rmse`：每像素重构 RMSE
- `component_names`：端元名（如 `("PE", "PP", "starch")`）

---

## ⚠️ 重要约定

**输出本质是丰度，不是分类标签**。任何"标签图"都是丰度的派生形式：
- 取最大丰度对应的主导组分
- 阈值掩膜
- 是否存在某组分的二值判断

---

## 🚧 当前状态

C3 阶段：目录骨架已建，`unmix.py` 待 C4 搬入。
