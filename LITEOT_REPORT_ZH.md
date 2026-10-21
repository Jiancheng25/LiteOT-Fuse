# LiteOT-Fuse：方法设计、实现与第一轮实验

## 1. 方法定位

LiteOT-Fuse 以 MFNet 为轻量基线，不引入 Transformer 或大模型。它只在编码器最深层
的 1/16 尺度增加两个模块：

1. **Spatially Constrained Local OT**：在非重叠 4×4 窗口内对齐 RGB 与热红外特征；
2. **Reliability-Gated Fusion**：根据特征、RGB 亮度和热红外局部对比度动态分配模态权重。

网络仍使用 MFNet 的双编码器、shortcut 和轻量解码器。完整模型参数从 736,275
增加到 744,942，仅增加 8,667（约 1.18%）。

## 2. 数学定义

在一个包含 N=16 个 token 的局部窗口内，RGB token 为 r_i，热红外 token 为 t_j，
匹配代价定义为：

```text
C_ij = 1 - cosine(r_i, t_j) + lambda_s * ||p_i - p_j||^2
```

其中第一项约束跨模态语义相似度，第二项限制空间距离。实验使用
`lambda_s=0.2`。熵正则 OT 为：

```text
P* = argmin_P <P,C> - epsilon H(P)
s.t. P 1 = a, P^T 1 = b
```

`a,b` 均为均匀分布，使用 3 次 log-domain Sinkhorn 迭代。对齐后的热红外特征：

```text
t_aligned(i) = N * sum_j P_ij t_j
F_T' = F_T + tanh(alpha) * W_o(t_aligned)
```

可靠性门控为：

```text
g = sigmoid(G([F_RGB, F_T', illumination, thermal_contrast])
            + beta * (illumination - 0.35))
F_fuse = concat(2g F_RGB, 2(1-g) F_T')
```

白天亮度较高时 g 倾向增大；夜间 g 降低，更多使用热红外。测试集统计中，联合训练
模型的平均 RGB 权重为：白天 0.534，夜间 0.429。

## 3. 实验设置

- MFNet 官方 train/val/test：1568/392/393 项；训练集的 `_flip` 项动态生成；
- 输入分辨率 480×640，像素除以 255；
- 从发布 MFNet 权重初始化共享骨干；
- SGD，momentum 0.9，weight decay 5e-4；
- 联合训练 30 epoch，初始学习率 0.001，每 epoch 乘 0.94；
- checkpoint 只按验证集论文协议 mIoU 选择，测试集在训练结束后评估；
- 所有表格沿用原论文/仓库协议，同时保留标准 mIoU、Eq.1 recall 和 overall accuracy。

## 4. 测试集结果

### 4.1 与发布 MFNet 的主要比较

| 方法 | 参数量 | Class avg. | 论文 mIoU | 标准前景 mIoU | Overall labeled acc. |
|---|---:|---:|---:|---:|---:|
| MFNet released | 736,275 | 0.6355 | 0.64915 | 0.2902 | 0.6079 |
| LiteOT-Fuse joint | 744,942 | **0.6534** | **0.64930** | 0.2764 | 0.5639 |
| LiteOT-Fuse frozen | 744,942 | 0.6404 | **0.64932** | 0.2889 | **0.6080** |

联合训练版在论文 Class avg. 上提升 0.01787（1.79 个百分点），论文 mIoU 基本持平；
夜间论文 mIoU 从 0.61895 提升到 0.62444。冻结骨干版增益较小，但 overall accuracy
不下降，更适合作为“轻量插件”的稳定结果。

### 4.2 消融实验

| 方法 | Class avg. | 论文 mIoU | Day mIoU | Night mIoU |
|---|---:|---:|---:|---:|
| MFNet released | 0.6355 | 0.64915 | **0.61076** | 0.61895 |
| Backbone fine-tune | 0.6047 | 0.64981 | 0.53759 | 0.62861 |
| OT only | 0.6049 | 0.65017 | 0.54746 | **0.62885** |
| Gate only | 0.6028 | **0.65083** | 0.54950 | 0.62749 |
| LiteOT-Fuse joint | **0.6534** | 0.64930 | 0.55831 | 0.62444 |

消融说明 OT-only 相对同训练流程的 backbone control 在夜间和全集 mIoU 上有小幅增益；
完整模型在 Class avg. 上出现明显协同增益。不过，联合训练导致标准 mIoU 和 overall
accuracy 下降，当前结果不能表述为“所有指标全面超过基线”。论文写作时必须报告该
现象，后续应通过多随机种子和更稳健的训练目标验证。

### 4.3 速度

同一张 RTX PRO 6000、batch=1、PyTorch eager、50 次预热和 300 次计时：

| 模型 | 延迟 | FPS |
|---|---:|---:|
| MFNet | 2.801 ms | 357.0 |
| LiteOT-Fuse | 3.829 ms | 261.2 |

OT 增加约 1.03 ms，但仍远高于实时要求。不同硬件上的绝对 FPS 不应直接与论文
Titan X 数值比较。

## 5. 代码与运行

核心文件：

- `model/LiteOTFuse.py`：局部 OT、Sinkhorn、可靠性门控及完整网络；
- `train_liteot.py`：训练、冻结骨干、消融和验证集选权重；
- `evaluate.py`：新旧模型统一评测；
- `benchmark.py`：同卡 batch=1 速度测试；
- `make_liteot_report.py`：生成消融 CSV 和图；
- `visualize_comparison.py`：固定样本的定性比较；
- `tests/test_liteot.py`：OT 边缘约束、输出形状和梯度测试。

联合训练：

```bash
conda activate SLOT
cd /home/wjc/Pro/MFNet-pytorch-master
python train_liteot.py --model liteot --ablation full \
  --data-dir data/MF --output-dir weights/LiteOTFuse/full \
  --init-mfnet weights/MFNet/final.pth --epochs 30 \
  --batch-size 6 --learning-rate 0.001 --device cuda:0
```

冻结骨干、只训练 8,667 个新增参数：

```bash
python train_liteot.py --model liteot --ablation full --freeze-backbone \
  --data-dir data/MF --output-dir weights/LiteOTFuse/full_frozen \
  --init-mfnet weights/MFNet/final.pth --epochs 20 \
  --batch-size 6 --learning-rate 0.01 --device cuda:0
```

评测：

```bash
python evaluate.py --model liteot --ablation full --data-dir data/MF \
  --weights weights/LiteOTFuse/full/best.pth \
  --output-dir results/liteot_full --device cuda:0
```

## 6. 下一阶段建议

1. 至少运行 3 个随机种子，报告均值和标准差；当前单次结果不能作为最终论文结论。
2. 只依据验证集尝试 `spatial_weight={0.1,0.2,0.4}`、Sinkhorn 3/5 次和 2×2/4×4 窗口。
3. 加入模态 dropout/热红外噪声训练，让 reliability gate 有明确的可靠性监督。
4. 使用前景感知但对 MFNet control 同样公平的损失，缓解 92% unlabeled 像素造成的
   决策偏置，并同时报告论文协议和标准 mIoU。
5. 在固定超参数后只测试一次，避免根据测试集选择联合版或冻结版。
