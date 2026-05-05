# Visualization 可视化模块

本目录的可视化代码**按图类型细分子目录**——每种图独立模块，方便维护和扩展。

## 📁 子目录结构（C5 阶段拆分）

| 子目录 | 用途 | 计划包含函数 |
|--------|------|------------|
| `abundance/` | 丰度热图 | `plot_abundance_maps` |
| `method_comparison/` | 方法横向对比柱图 | `plot_method_metric_bars`, `plot_method_abundance_bars` |
| `preprocessing/` | 预处理协议对比 | `plot_single_spectrum_preprocessing`, `plot_protocol_spectrum_triptych`, `plot_protocol_abundance_grid` |
| `reconstruction/` | 输入谱与重构谱对比 | `plot_reconstruction_examples` |
| `residual/` | 残差空间分布 | `plot_residual_map` |

## 📁 顶层文件

| 文件名 | 功能 | 状态 |
|--------|------|------|
| `_common.py` | 共用工具（`_ensure_parent`, `_coordinate_frame`, `_grid_from_frame`） | 🚧 C5 拆出 |
| `__init__.py` | 顶层 re-export 所有函数（保兼容） | ✅ |
| `README.md` | 本文档 | ✅ |

---

## 🎯 使用方式

```python
# 顶层入口（推荐，简洁）
from visualization import plot_abundance_maps, plot_residual_map

# 也可以从子目录直接导入（精确）
from visualization.abundance.abundance_maps import plot_abundance_maps
```

---

## 🚧 当前状态

C3 阶段：顶层目录已建，模块拆分将在 C5 完成。

## 📐 扩展规则

新增图时遵循以下规则：

1. **属于已有子目录** → 直接在对应子目录下加新 `.py`
2. **图类型全新**（如 SHAP / GradCAM / 端元偏离图）→ 新建子目录并加 `__init__.py`
3. **共用工具**（坐标解析、栅格生成）→ 加到 `_common.py`
