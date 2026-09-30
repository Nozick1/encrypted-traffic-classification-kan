# KAN vs MLP 流量分类对照实验

公平对比实验：**同一个 CNN 特征提取器**，仅替换分类头为 **MLP** 或 **Chebyshev-KAN**，
其余（数据、优化器、学习率、epoch、批大小、随机种子）完全一致，用于回答「KAN 分类头是否优于 MLP 分类头」。

## 环境
- Python 3.10，PyTorch 2.5.1+cu121（CUDA 可选）
- 依赖：torch、torchvision、numpy、matplotlib（无需 sklearn）

## 文件
- `chebykan.py` —— 切比雪夫-KAN 层（已修复原仓库 `cos(k·arccos(x))` 的 NaN 数值问题，改用递推式）
- `train_compare.py` —— 训练脚本（`--model mlp|kan|both`，`--seed` 复现，`--subset` 冒烟，`--resume` 续跑）
- `eval_models.py` —— 仅用已保存的最优权重做测试集评估（轻量，无训练）
- `merge_summary.py` —— 合并「MLP 完整训练」与「续跑 KAN」的结果
- `plot_final.py` —— 从 `results/final_metrics.json` 出图（混淆矩阵 / 指标对比柱状图）
- `plot_results.py` —— 从 `results/summary.json` 出图（训练曲线 / 混淆矩阵 / 对比图）

## 模型结构
- 特征提取器（两者共用）：3 层卷积（3→32→64→128）+ ReLU + MaxPool → 全局平均池化 → 128 维
- MLP 头：128→256→ReLU→Dropout(0.5)→10（总模型 ≈ 12.9 万参数）
- KAN 头：LayerNorm(128)→ChebyKAN(128→64, deg=3)→LayerNorm(64)→ChebyKAN(64→10, deg=3)（总模型 ≈ 12.9 万参数）

## 数据集
USTC-TFC2016 10 类子集（59,331 训练 / 6,593 测试），流量转 224×224 灰度图。
类别：BitTorrent / Facetime / Gmail / Miuref / MySQL / Outlook / Skype / Tinba / WorldOfWarcraft / Zeus。

## 结果（种子 0，完整 10 epoch）

| 分类头 | 测试准确率 | F1 | 精确率 | 召回率 |
|---|---|---|---|---|
| MLP | 86.06% | 84.77% | 90.91% | 86.06% |
| ChebyKAN | **86.71%** | **85.47%** | **92.06%** | **86.71%** |

难点类：BitTorrent（38.0%）、Gmail（20.9%）——二者均为 443 端口加密流量，图像模式相似，多被误判为 Outlook。
