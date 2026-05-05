# Synthetic 合成数据模块

本目录用于生成**有像素级真值的合成拉曼面扫数据**，专门服务定量解混评估。

## 📁 文件列表

| 文件名 | 功能 | 状态 |
|--------|------|------|
| `generator.py` | 合成数据核心生成逻辑（端元线性混合 + 噪声/基线扰动） | 🚧 C4 搬入 |
| `generate_dataset.py` | 命令行入口：生成一份合成数据集到 outputs/ | 🚧 C4 搬入 |
| `README.md` | 本文档 | ✅ |

---

## 🎯 为什么需要合成数据

真实拉曼面扫只是"弱标签"——只能验证整图组分组成，**不能给出每个像素的真实丰度**。要定量评估 NNLS / OLS / FCLS / NMF 的丰度恢复精度，必须用合成真值数据。

合成方式：
1. 从纯谱库选取端元
2. 人工生成二维丰度图（高斯平滑、归一化和=1）
3. 用 `丰度 × 端元谱` 线性混合得到合成谱
4. 叠加可控噪声、基线扰动和强度缩放

---

## 📊 与主线的关系

```
preprocessing/endmembers.py (PE/PP/淀粉 纯谱)
    ↓
generator.py 生成 (abundance, spectra) 配对
    ↓
generate_dataset.py 落盘到 outputs/synthetic_datasets/<name>/
    ↓
experiments/run_synthetic_method_comparison.py 跑四方法对比
```

---

## 🚧 当前状态

C3 阶段：目录骨架已建，模块文件待 C4 搬入。Wave 1.5 阶段将重写生成器，加入更多丰度分布模式（smooth / extreme / sparse）和噪声等级。

## 🔬 合成数据集迭代记录（待 Wave 1.5 后填充）

| 数据集名 | smooth_sigma | noise_std | 丰度方差 | 关键发现 |
|---------|-------------|-----------|---------|---------|
| `smooth` | 3.0 | 0.01 | ~0.0002 | 方差太小、R² 失效；OLS=NNLS=FCLS 等价 |
