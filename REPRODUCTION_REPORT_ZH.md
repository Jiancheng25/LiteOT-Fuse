# MFNet 实验复现报告

## 1. 复现结论

使用仓库发布的 `weights/MFNet/final.pth`，在官方 393 张测试划分、480×640
分辨率、输入像素除以 255 的设置下完成了测试。论文协议下的全测试集 mIoU 为
**0.6491**：按 Table II 的三位小数与论文 **0.649** 相同；按 Table III 的四位数
0.6486 比较，绝对差为 +0.00055。

| 测试部分 | 论文 Class avg. | 实测 Class avg. | 差值 | 论文 mIoU | 实测 mIoU | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| 全部 both | 0.5910 | 0.6355 | +0.0445 | 0.6490 | 0.6491 | +0.0001 |
| 白天 day | 0.4770 | 0.4901 | +0.0131 | 0.5740 | 0.6108 | +0.0368 |
| 夜间 night | 0.6350 | 0.6583 | +0.0233 | 0.6210 | 0.6189 | -0.0021 |

全测试集 mIoU 高度复现；Class avg. 和白天子集存在一定偏差。发布仓库使用的
PyTorch 权重并非论文所述 Chainer 1.17/CUDA 7.5 运行栈，仓库训练默认值与论文
也不完全相同（例如学习率衰减、batch size）。因此应把这里理解为“发布权重在
发布数据划分上的复测”，而不是原始 Chainer 模型的逐位重放。

## 2. 指标是什么意思

令 `P[i,j]` 表示真实类别为 i、预测类别为 j 的像素数。

- **Per-class score / Class avg.（论文表格协议）**：发布代码实际计算
  `P[i,i] / sum_i P[i,j]`，数学上是每类 Precision（预测成某类的像素中有多少
  是正确的），然后对包含 unlabeled 的 9 类做宏平均。数值越高越好。它容易被
  小类别和“模型很少预测该类”影响。
- **Class accuracy（论文公式 Eq. 1）**：公式文字对应
  `P[i,i] / sum_j P[i,j]`，即每类 Recall（真实属于某类的像素中有多少被找回）。
  论文公式与发布评测代码的分母方向不一致，因此 `metrics.json` 同时保留
  `paper_table_class_score` 和 `class_recall_eq1`，避免概念混淆。
- **IoU**：某一类预测区域与真实区域交集除以并集，取值 0～1；比像素准确率更
  严格，因为误报和漏报都会使它下降。
- **mIoU**：各类 IoU 的算术平均。论文排除 unlabeled，并且计算前景类并集时也
  排除了与 unlabeled 之间的交叉错误；这里称为 `paper_mean_iou`。这是与 Table II
  直接比较的主指标。
- **Standard mIoU**：常见现代语义分割实现会把“前景误分成 unlabeled”以及
  “unlabeled 误分成前景”计入并集。数据中 unlabeled 超过 92%，因此标准协议的
  数值显著低于论文协议；两者不能直接横向比较。
- **Overall pixel accuracy (labeled)**：所有真实前景像素中预测正确的比例。它会
  被像素数量大的类别主导，不能替代 mIoU。
- **FPS / ms per image**：吞吐速度和单图延迟。当前 GPU 在 batch=1、预热 50 次、
  计时 300 次时为 405.1 FPS / 2.469 ms；论文 Titan X 为 55.6 FPS。硬件、CUDA、
  PyTorch/Chainer 均不同，只能说明当前机器的部署速度，不能作为严格复现误差。

## 3. 逐类别观察

完整数值见 `results/mfnet_reproduction/paper_comparison.csv`。全测试集相对论文：

| 类别 | 论文 | 实测 | 差值 |
|---|---:|---:|---:|
| Unlabeled | 0.9680 | 0.9735 | +0.0055 |
| Car | 0.8290 | 0.7885 | -0.0405 |
| Pedestrian | 0.8520 | 0.8267 | -0.0253 |
| Bike | 0.7420 | 0.6811 | -0.0609 |
| Curve | 0.6150 | 0.6445 | +0.0295 |
| Car stop | 0.2730 | 0.3160 | +0.0430 |
| Guardrail | 0.0000 | 0.0000 | 0.0000 |
| Color cone | 0.6070 | 0.7181 | +0.1111 |
| Bump | 0.4330 | 0.7713 | +0.3383 |

Guardrail 为 0 说明该发布模型在此划分上没有形成有效的正确预测；小类别像素少，
所以 Color cone、Bump 等类别对模型版本和数据分布尤其敏感。

## 4. 环境、数据与产物

- Conda 环境：`SLOT`（Python 3.10.20、PyTorch 2.11.0+cu128）
- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition
- 测试图像：393（day 205，night 188）
- 模型参数：736,275，和论文约 0.73M 一致
- 权重 SHA-256：`48578bd812c6bcbebeca8c91455ba8644340bd763d0ed802e9b3937665529a13`
- 数据压缩包 SHA-256：`41bd75b8dee29fe37a21a6280fe2723336e22591589dedba46297d57c7d1611b`

官方 Google Drive 在本机网络下无法列目录，因此实际下载使用公开 Kaggle 镜像；
解压后的目录结构、1569 张原始数据、类别编号和 393 张测试列表均与项目说明一致。

主要产物：

- `results/mfnet_reproduction/paper_table2_full_comparison.png`：Table II 逐类别对比图
- `results/mfnet_reproduction/paper_comparison.png`：Class avg./mIoU 汇总对比图
- `results/mfnet_reproduction/paper_comparison.csv`：论文与实测原始表格
- `results/mfnet_reproduction/metrics.json`：混淆矩阵和全部新旧协议指标
- `results/mfnet_reproduction/benchmark_batch1.json`：batch=1 速度测试

## 5. 后续重复测试

```bash
conda activate SLOT
cd /home/wjc/Pro/MFNet-pytorch-master
python evaluate.py --data-dir data/MF --weights weights/MFNet/final.pth \
  --batch-size 8 --device cuda:0
```

修改网络后，只需把 `--weights` 指向新权重；如果类数或输出定义改变，需要同步修改
`evaluate.py` 中的 `CLASS_NAMES` 和模型构造部分。
