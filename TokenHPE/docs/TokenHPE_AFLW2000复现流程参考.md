# TokenHPE AFLW2000 对比实验复现流程参考

本文档用于给后续实验报告撰写提供素材，记录本次如何将 TokenHPE 作为 6DRepNet 的对比方法，在同一个 AFLW2000 测试集上完成推理评估，并复现作者论文/README 中给出的结果。

## 1. 对比方法选择

课程实验要求选择近 3 年内另一篇公开源码的人体头部姿态估计方法，与 6DRepNet 在同一测试集下进行对比实验。我们选择的对比方法是 CVPR 2023 论文 **TokenHPE: Learning Orientation Tokens for Efficient Head Pose Estimation via Transformers**。

选择 TokenHPE 的原因主要有三点。第一，它与 6DRepNet 属于同一任务，均面向 head pose estimation，并且都在 AFLW2000 上报告了结果，便于公平对比。第二，TokenHPE 作者公开了完整源码和训练权重，满足课程“作者公开全部源码”的要求。第三，TokenHPE 的方法路线与 6DRepNet 有明显差异：6DRepNet 重点在于连续 6D 旋转表示与 CNN backbone，而 TokenHPE 引入 ViT 和 orientation tokens，通过 Transformer 学习面部局部区域之间的关系，因此适合作为结构和思想上都有代表性的对比方法。

TokenHPE 论文中在 AFLW2000 上主要报告了 TokenHPE-v1 和 TokenHPE-v2 两个版本。v1 使用 9 个 orientation tokens，对应 9 个基础朝向区域；v2 使用 11 个 orientation tokens，对应更细粒度的 11 个基础朝向区域。本次本地已有作者公开的 v1 checkpoint，因此实际复现对象为 **TokenHPE-v1**。论文 Table 1 中 TokenHPE-v1 的 AFLW2000 结果为 MAE 4.85、MAEV 6.11；作者 README 中 released weight 的参考结果为 MAE 4.81、VMAE 6.09。

## 2. 代码获取与目录组织

首先获取 TokenHPE 官方代码库。若从零开始，可以在总工程目录下执行：

```bash
git clone https://github.com/nbh0819/TokenHPE.git TokenHPE
```

本实验将 TokenHPE 放置在总工程目录下：

```text
HeadPoseEstimationExplore/
├── 6DRepNet/
│   ├── datasets/AFLW2000/
│   └── ...
├── TokenHPE/
│   ├── checkpoints/
│   ├── script/
│   ├── model.py
│   ├── test.py
│   ├── datasets.py
│   └── ...
└── .venv/
```

为了保证和 6DRepNet 的对比公平，本次没有为 TokenHPE 单独复制 AFLW2000 数据，而是直接复用 6DRepNet 已经准备好的数据目录：

```text
6DRepNet/datasets/AFLW2000
```

这样可以确保两个方法使用同一批图像、同一批 `.mat` 标注文件、同一个 1969 样本测试列表，避免由于数据列表不同导致对比失真。

## 3. 环境配置

TokenHPE 官方 README 中给出的环境为 Python 3.9、PyTorch 1.10.1 及 CUDA 11.2。本次实际运行环境为本地 WSL Ubuntu + Python 虚拟环境 + CUDA GPU。由于只进行推理测试，不训练模型，因此并不严格要求完全复刻作者训练环境，只要 PyTorch、torchvision、timm、einops、OpenCV、SciPy 等依赖可用即可。

在总工程目录中激活已有虚拟环境：

```bash
source .venv/bin/activate
```

安装 TokenHPE 依赖：

```bash
pip install -r TokenHPE/requirements.txt
```

复现过程中发现，TokenHPE 原始 `requirements.txt` 没有列出 `timm`、`einops` 和 `seaborn`，但代码实际依赖这些包。其中 `model.py` 使用 `timm` 中的 `trunc_normal_`，Transformer token 处理依赖 `einops`，可视化相关代码导入了 `seaborn`。因此在本次复现中补充了：

```text
timm>=0.6.13
einops>=0.6.0
seaborn>=0.12.0
```

此外，当前环境安装到的 `timm` 版本为 1.0.27，新版 `timm` 中已经不再提供旧路径 `timm.models.layers.weight_init`。原代码：

```python
from timm.models.layers.weight_init import trunc_normal_
```

会报错：

```text
ModuleNotFoundError: No module named 'timm.models.layers.weight_init'
```

因此将 `TokenHPE/model.py` 中的导入改为兼容写法：

```python
try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_
```

这样既兼容新版 `timm`，也兼容旧版环境。

## 4. 数据集与测试列表

TokenHPE README 中说明其数据准备方式跟随 6DRepNet。AFLW2000 数据集包含 2000 张图像和对应 `.mat` 姿态标注，但作者评估时会过滤掉 yaw、pitch、roll 中任意角度绝对值超过 99° 的样本，最终使用 1969 个样本进行测试。

本次复现直接使用 6DRepNet 阶段已经生成的测试列表：

```text
6DRepNet/datasets/AFLW2000/files_filtered_99.txt
```

该列表与 6DRepNet 复现时一致，共包含 1969 个有效样本。这样做有两个好处：一是与作者论文中的常规评估协议保持一致；二是保证 6DRepNet 与 TokenHPE 的对比在同一测试集上完成。

## 5. 权重准备

TokenHPE README 中提供了两个权重链接：

1. TokenHPE 训练好的模型权重；
2. ViT-B/16 的 ImageNet-21K 预训练权重，可作为 feature extractor 初始化使用。

本次 AFLW2000 测试只需要作者已经训练好的 TokenHPE checkpoint。实际使用的权重文件为：

```text
TokenHPE/checkpoints/TokenHPEv1-ViTB-224_224-lyr3.tar
```

## 6. 评估脚本编写

TokenHPE 原始仓库中已经提供了 `test.py`，但直接运行存在几个不便之处：

1. 原脚本路径默认假设数据在 `TokenHPE/datasets/AFLW2000`，而本实验希望复用 `6DRepNet/datasets/AFLW2000`；
2. 原脚本硬编码 CUDA 和部分默认参数，缺少更清晰的日志记录；
3. 原脚本会导入图形后端相关内容，但本次批量评估不需要可视化窗口；

因此在 `TokenHPE/script/` 下新增了两个复现脚本：

```text
TokenHPE/script/eval_aflw2000.py
TokenHPE/script/run_eval_aflw2000.sh
```

其中 `eval_aflw2000.py` 保留作者原始评估逻辑，包括：

```text
Resize(250)
CenterCrop(224)
ImageNet Normalize
TokenHPE(num_ori_tokens=9, depth=3, heads=8, embedding="sine", dim=128)
Yaw/Pitch/Roll MAE
Vec1/Vec2/Vec3 VMAE
```

脚本输出的 MAE 与 VMAE 计算方式也与作者 `test.py` 保持一致。`run_eval_aflw2000.sh` 则负责统一设置路径、日志文件、batch size、GPU id 和模型权重路径。

默认运行配置如下：

```bash
DATA_DIR=6DRepNet/datasets/AFLW2000
FILENAME_LIST=6DRepNet/datasets/AFLW2000/files_filtered_99.txt
MODEL_PATH=TokenHPE/checkpoints/TokenHPEv1-ViTB-224_224-lyr3.tar
NUM_ORI_TOKENS=9
BATCH_SIZE=32
GPU=0
```

## 7. 一键运行命令

在总工程目录下执行：

```bash
source .venv/bin/activate
pip install -r TokenHPE/requirements.txt
bash TokenHPE/script/run_eval_aflw2000.sh
```

## 8. 复现过程中的主要问题与解决

第一个问题是依赖缺失。原仓库依赖文件不完整，实际运行时先后发现缺少 `timm`、`einops`、`seaborn`。其中 `seaborn` 虽然只用于可视化 attention heatmap，但因为在 `model.py` 顶层导入，所以即使只做评估也必须安装。

第二个问题是 `timm` API 变化。TokenHPE 原始代码使用旧版导入路径：

```python
from timm.models.layers.weight_init import trunc_normal_
```

新版 `timm` 中该路径不可用，导致模型导入阶段直接失败。通过改成新旧兼容导入后解决。

第三个问题是评估协议对齐。为了与 6DRepNet 对比公平，本实验没有使用 TokenHPE 目录下单独生成的 `files.txt`，而是直接复用 6DRepNet 阶段已经验证过的 `files_filtered_99.txt`。最终样本数为 1969，与作者常规 AFLW2000 测试协议一致。

## 9. 最终运行结果

最终运行日志文件为：

```text
TokenHPE/output/eval_aflw2000_20260629_110015.log
```

日志核心内容如下：

```text
[TokenHPE eval] root=/home/sunflower/HeadPoseEstimationExplore/TokenHPE
[TokenHPE eval] data_dir=/home/sunflower/HeadPoseEstimationExplore/6DRepNet/datasets/AFLW2000
[TokenHPE eval] filename_list=/home/sunflower/HeadPoseEstimationExplore/6DRepNet/datasets/AFLW2000/files_filtered_99.txt
[TokenHPE eval] checkpoint=/home/sunflower/HeadPoseEstimationExplore/TokenHPE/checkpoints/TokenHPEv1-ViTB-224_224-lyr3.tar
[TokenHPE eval] device=cuda:0
dataset path is: /home/sunflower/HeadPoseEstimationExplore/6DRepNet/datasets/AFLW2000/files_filtered_99.txt
==> Add Sine PositionEmbedding~
[TokenHPE eval] model weight loaded
Samples: 1969
Yaw: 4.4337, Pitch: 5.7752, Roll: 4.2325, MAE: 4.8138
Vec1: 6.1332, Vec2: 5.3245, Vec3: 6.8360, VMAE: 6.0979
Paper reference on AFLW2000: MAE 4.81, VMAE 6.09
```

论文 Table 1 中 TokenHPE-v1 的列顺序是 `Pitch / Yaw / Roll / MAE`，而脚本输出是 `Yaw / Pitch / Roll / MAE`。对齐后结果如下：

| 指标 | 论文 Table 1 TokenHPE-v1 | 本次复现 |
| --- | ---: | ---: |
| Pitch | 5.73 | 5.7752 |
| Yaw | 4.53 | 4.4337 |
| Roll | 4.29 | 4.2325 |
| MAE | 4.85 | 4.8138 |
| MAEV/VMAE | 6.11 | 6.0979 |

同时，作者 README 中公开权重的参考结果为：

```text
MAE 4.81, VMAE 6.09
```

本次结果 `MAE=4.8138, VMAE=6.0979` 与 README 几乎完全一致，也与论文 Table 1 中 TokenHPE-v1 的结果高度一致。因此可以认为 TokenHPE-v1 在 AFLW2000 上的测试已经成功复现。

## 10. 与 6DRepNet 对比时可使用的数据

6DRepNet 在同一 AFLW2000 filtered 1969 样本上的复现结果为：

```text
Samples: 1969
Yaw: 3.6258, Pitch: 4.9073, Roll: 3.3737, MAE: 3.9689
Paper reference on AFLW2000: Yaw 3.63, Pitch 4.91, Roll 3.37, MAE 3.97
```

TokenHPE-v1 在同一测试集上的复现结果为：

```text
Samples: 1969
Yaw: 4.4337, Pitch: 5.7752, Roll: 4.2325, MAE: 4.8138
Vec1: 6.1332, Vec2: 5.3245, Vec3: 6.8360, VMAE: 6.0979
```

因此，在本次实际复现实验中，6DRepNet 的 AFLW2000 MAE 更低，而 TokenHPE-v1 的结果也准确复现了作者公开权重的指标。报告中可以从方法角度进一步分析：6DRepNet 的优势来自连续 6D rotation representation 与较强的 CNN 姿态回归能力；TokenHPE 的亮点则在于用 orientation tokens 显式学习面部区域之间的关系，并提供了 MAEV 这一基于旋转矩阵向量的补充指标。二者的比较不仅是数值高低比较，也体现了 CNN 几何表示路线与 Transformer token-learning 路线之间的差异。
