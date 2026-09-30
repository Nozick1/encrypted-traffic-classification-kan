# 基于 Chebyshev-KAN 的加密网络流量分类

**Encrypted Traffic Classification with Chebyshev-KAN** · 本科大创项目

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 将原始网络流（PCAP）编码为灰度图像，用 CNN 提取特征，并在**完全相同**的特征下公平对比 MLP 分类头与 Chebyshev-KAN 分类头。

---

## 简介

本项目实现了一套基于「流量图像化」的加密流量分类流程：

- **流量图像化**：按 FlowPic 风格将每条网络流编码为 28×28 灰度图；
- **特征提取**：3 层卷积 + 全局平均池化，得到 128 维特征向量；
- **分类头对比**：在共享特征下，公平对比 **MLP 头** 与 **切比雪夫 KAN（ChebyKAN）头**；
- **数值稳定性修复**：修复原 ChebyKAN 实现中 `cos(k·arccos x)` 在输入饱和时的 NaN 梯度发散问题，改用切比雪夫递推式求值。

## 核心结果

在 USTC-TFC2016 数据集的 10 类子集（训练 59,331 / 测试 6,593，流量已转 28×28 灰度图）上，随机种子 0、训练 10 个 epoch：

| 分类头 | 测试准确率 | 精确率 | 召回率 | F1 |
|---|---|---|---|---|
| MLP | 86.06% | 90.91% | 86.06% | 84.77% |
| **ChebyKAN** | **86.71%** | **92.06%** | **86.71%** | **85.47%** |

ChebyKAN 头在四项指标上均以微弱幅度优于 MLP 头。混淆矩阵、训练曲线与指标对比柱状图见 `kan_experiment/results/`；逐类别分析显示错误主要集中于 BitTorrent 与 Gmail 两类 443 端口加密流量（二者因图像模式相似而与 Outlook 互相混淆）。

## 目录结构

```
encrypted-traffic-classification-kan/
├── paper/                       # 论文（Word 与 Markdown 草稿）
├── kan_experiment/
│   ├── chebykan.py              # Chebyshev-KAN 层（数值稳定版）
│   ├── train_compare.py         # 训练脚本（--model mlp|kan|both）
│   ├── eval_models.py           # 用已保存权重评估（无需训练）
│   ├── merge_summary.py         # 合并 MLP 与续跑 KAN 的完整结果
│   ├── plot_final.py            # 从 final_metrics.json 出图
│   ├── plot_results.py          # 从 summary.json 出图（训练曲线等）
│   └── results/                 # 结果 JSON + 图 + 最优权重
├── requirements.txt
└── LICENSE
```

## 快速开始

### 1. 环境

- Python 3.10，PyTorch 2.5.1（CUDA 可选，CPU 亦可）
- 依赖：torch、torchvision、numpy、matplotlib（**无需 sklearn**）

```bash
pip install -r requirements.txt
```

### 2. 数据准备

数据集采用 USTC-TFC2016 的 10 类子集，流量已转换为 28×28 灰度 PNG，并以 `ImageFolder` 格式组织（`train/` 与 `test/`，各含 0~9 十个类别文件夹）。类别依次为：

`BitTorrent / Facetime / Gmail / Miuref / MySQL / Outlook / Skype / Tinba / WorldOfWarcraft / Zeus`（其中 Miuref、Tinba、Zeus 为恶意流量）。

> 数据不在本仓库内。请自行准备后将路径传给 `--data_dir`（脚本内默认路径为本机路径，需改为你的路径）。

### 3. 训练

```bash
# 冒烟测试（3000 张子集、3 epoch，快速验证 KAN 不再 NaN）
python -u kan_experiment/train_compare.py --model both --subset 3000 --epochs 3 --workers 0 --data_dir /path/to/dataset

# 完整训练（10 epoch，约 1.5~2 小时）
python -u kan_experiment/train_compare.py --model both --epochs 10 --workers 0 --seed 0 --data_dir /path/to/dataset
```

> 低内存机器请用 `--workers 0`；默认 `--workers 4` 在 16GB 内存下可能因内存不足被系统终止。

### 4. 用已训练权重评估（无需训练）

仓库已附带种子 0 的最优权重（`kan_experiment/results/best_{mlp,kan}.pth`），可复现论文数字：

```bash
# 将 eval_models.py 中的 data_dir 改为你的数据集路径后运行
python kan_experiment/eval_models.py
```

### 5. 复现图表

```bash
python kan_experiment/plot_results.py   # 训练曲线 / 混淆矩阵 / 对比柱状图
python kan_experiment/plot_final.py     # 从 final_metrics.json 出图
```

## 方法与实现要点

- **特征提取器（两者共用）**：3 层卷积（通道 3→32→64→128，3×3）+ ReLU + 2×2 最大池化 → 全局平均池化 → 128 维。
- **MLP 头**：`128→256→ReLU→Dropout(0.5)→10`（总参数量 128,842）。
- **ChebyKAN 头**：`LayerNorm(128)→ChebyKAN(128→64, 3 阶)→LayerNorm(64)→ChebyKAN(64→10, 3 阶)`（总参数量 128,960）。
- **数值稳定性**：原实现以 `cos(k·arccos x)` 计算切比雪夫多项式，当 `tanh` 饱和至 ±1 时梯度出现 `0/0` 导致 NaN；本项目改用递推式 `T_k(x) = 2x·T_{k-1}(x) − T_{k-2}(x)`（`T_0=1, T_1=x`），并在每层 KAN 前加 LayerNorm。

更多细节见 `kan_experiment/README.md` 与 `paper/` 目录下的论文。

## 引用

若本项目对你有帮助，欢迎引用：

```bibtex
@misc{encrypted-traffic-classification-kan,
  author       = {Qin, Yuxin},
  title        = {基于 Chebyshev-KAN 的加密网络流量分类},
  year         = {2026},
  howpublished = {\url{https://github.com/Nozick1/encrypted-traffic-classification-kan}},
}
```

## 许可证

本项目基于 [MIT License](LICENSE) 开源。
