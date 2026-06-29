# 6DRepNet 本地复现运行指南

本文档用于完成数字图像处理大作业中的“基线方法复现”部分。根据助教通知，BIWI 数据集官方下载入口已经失效，本次只需要在 AFLW2000 一个数据集上复现论文结果即可。因此后续全部实验都在本地完成，不再使用远程课题组服务器。

本次需要完成两个任务：

1. 跑通 6DRepNet 测试 demo 程序，并保存可视化结果。
2. 在 AFLW2000 数据集上跑通测试代码，得到与论文接近的指标。

项目已经在 `6DRepNet/script/` 下提供两个一键脚本：

```bash
bash 6DRepNet/script/run_demo.sh
bash 6DRepNet/script/run_eval_aflw2000.sh
```

## 1. 方法核心

6DRepNet 解决的是单张人脸图像的头部姿态估计问题，输出 yaw、pitch、roll 三个角度。传统方法常直接回归欧拉角，或者使用四元数表示旋转。论文指出，欧拉角存在万向节锁和等价角度标签问题，四元数也存在 `q` 与 `-q` 表示同一旋转的歧义，这些不连续性会让神经网络回归更难。

6DRepNet 的核心思想是：网络不直接回归欧拉角，而是回归连续的 6D 旋转表示。代码中 `sixdrepnet/model.py` 的 `SixDRepNet.forward()` 输出 6 个数后，调用 `utils.compute_rotation_matrix_from_ortho6d()`，将其转换为合法的 `3 x 3` 旋转矩阵。训练时使用 SO(3) 上的 geodesic loss，测试时再把旋转矩阵转换回欧拉角，计算 yaw、pitch、roll 的 MAE。

本次复现使用作者提供的 300W-LP 训练权重，在 AFLW2000 上测试。论文 README 中 AFLW2000 参考结果如下：

| 数据集 | Yaw | Pitch | Roll | MAE |
| --- | ---: | ---: | ---: | ---: |
| AFLW2000 | 3.63 | 4.91 | 3.37 | 3.97 |

## 2. 本地目录结构

建议保持如下目录结构：

```text
6DRepNet/
  checkpoints/
    6DRepNet_300W_LP_AFLW2000.pth
  datasets/
    AFLW2000/
      image00002.jpg
      image00002.mat
      ...
      files.txt
  logs/
  output/
    demo/
      aflw2000_demo.png
  script/
    run_demo.sh
    run_eval_aflw2000.sh
    run_eval_aflw2000_all_checkpoints.sh
    demo_image.py
    eval_aflw2000.py
```

其中 `datasets/`、`checkpoints/`、`logs/`、`output/` 都是本地运行产物或大文件目录，已经写入 `.gitignore`，不要提交到 GitHub。

## 3. 关键文件

| 路径 | 作用 |
| --- | --- |
| `script/run_demo.sh` | 任务 1 的一键脚本，默认跑 AFLW2000 单图 demo 并保存可视化结果 |
| `script/demo_image.py` | 单图 demo 辅助脚本，复用作者模型和绘图函数 |
| `script/run_eval_aflw2000.sh` | 任务 2 的一键脚本，评估 AFLW2000 |
| `script/run_eval_aflw2000_all_checkpoints.sh` | 依次测试本地 `checkpoints/*.pth` 在 AFLW2000 上的表现 |
| `script/eval_aflw2000.py` | AFLW2000 测试辅助脚本，复用作者模型、数据集和误差计算逻辑 |
| `sixdrepnet/demo.py` | 作者原始摄像头实时 demo |
| `sixdrepnet/test.py` | 作者原始评估脚本 |
| `sixdrepnet/model.py` | 6DRepNet 网络定义 |
| `sixdrepnet/utils.py` | 旋转表示转换和可视化函数 |
| `sixdrepnet/datasets.py` | AFLW2000 数据读取逻辑 |

注意：仓库根目录下的 `6DRepNet/test.py` 是 pip 包使用示例，不是本次评估入口。真正的作者评估入口是 `sixdrepnet/test.py`。本次新增的 `script/eval_aflw2000.py` 与作者测试逻辑保持一致，但去掉了 GUI 后端依赖，更适合本地一键复现。

## 4. 环境准备

推荐在仓库根目录 `HeadPoseEstimationExplore/` 下创建并使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r 6DRepNet/requirements.txt
```

如果已经进入了 `6DRepNet/` 目录，则使用：

```bash
cd 6DRepNet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

如果要使用作者原始摄像头 demo，还需要安装 RetinaFace 检测器：

```bash
pip install "git+https://github.com/elliottzheng/face-detection.git@master"
```

本次默认的单图 demo 不依赖摄像头，也不强制依赖 `face_detection`。

如果摄像头 demo 报 `No module named 'numpy.lib.function_base'`，这是原作者代码中无用的 NumPy 私有导入导致的新版 NumPy 兼容问题；本仓库已经删除该导入。重新运行 `DEMO_MODE=camera bash 6DRepNet/script/run_demo.sh` 即可。

## 5. 数据和权重

### 5.1 权重文件

从作者 README 的 fine-tuned models 下载：

```text
https://drive.google.com/drive/folders/1V1pCV0BEW3mD-B9MogGrz_P91UhTtuE_?usp=sharing
```

本次使用：

```text
6DRepNet/checkpoints/6DRepNet_300W_LP_AFLW2000.pth
```

如果权重放在其他位置，可以运行脚本时指定：

```bash
SNAPSHOT=/path/to/6DRepNet_300W_LP_AFLW2000.pth bash 6DRepNet/script/run_eval_aflw2000.sh
```

### 5.2 AFLW2000

从 3DDFA 项目页面下载 AFLW2000：

```text
http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
```

解压后让 `.jpg` 和 `.mat` 文件直接位于：

```text
6DRepNet/datasets/AFLW2000/
```

作者原仓库的 `create_filename_list.py` 会过滤掉 yaw/pitch/roll 任一角度超过 ±99° 的样本。AFLW2000 原始数据有 2000 张图，按作者过滤规则实际参与 Table 1 统计的是 1969 张左右。为了和论文 Table 1 对齐，`script/eval_aflw2000.py` 会在缺少 filtered list 时自动生成：

```text
6DRepNet/datasets/AFLW2000/files_filtered_99.txt
```

不要用全量 2000 张样本直接算 Table 1，否则 MAE 会明显偏高。

## 6. 任务 1：跑通 demo

默认推荐运行单图 demo：

```bash
bash 6DRepNet/script/run_demo.sh
```

该脚本会：

1. 加载 `checkpoints/6DRepNet_300W_LP_AFLW2000.pth`。
2. 从 AFLW2000 中选择一个样本。
3. 根据 `.mat` 标注裁剪人脸区域。
4. 预测 pitch、yaw、roll。
5. 在原图上绘制姿态立方体。
6. 保存可视化图像和日志。

输出位置：

```text
6DRepNet/output/demo/aflw2000_demo.png
6DRepNet/logs/demo_image_时间戳.log
```

可以指定某一张样本：

```bash
DEMO_SAMPLE=image00002 bash 6DRepNet/script/run_demo.sh
```

如果要跑作者原始摄像头 demo：

```bash
DEMO_MODE=camera bash 6DRepNet/script/run_demo.sh
```

摄像头模式需要本机摄像头、OpenCV GUI 窗口和 `face_detection` 包。按 Esc 退出窗口。

摄像头 demo 的常见失败点有两个：

1. NumPy 兼容问题：如果出现 `No module named 'numpy.lib.function_base'`，说明运行到了作者原始代码中的旧私有导入。本仓库已经删除该导入，请重新拉取/保存代码后再运行。
2. WSL 摄像头不可见：WSL 默认通常不能直接访问 Windows 摄像头。先确认设备是否存在：

```bash
ls /dev/video*
```

如果没有 `/dev/video0`，说明当前 WSL 没有接收到 Windows 摄像头设备。此时有两种路线：

**路线 A：最快、最稳，使用 Windows 原生 Python/Anaconda 跑摄像头 demo。**

Windows 原生 OpenCV 可以直接访问摄像头，不需要 WSL 摄像头直通。进入同一份项目目录后运行：

```powershell
cd path\to\HeadPoseEstimationExplore\6DRepNet
python -m venv .venv-win
.\.venv-win\Scripts\activate
pip install -r requirements.txt
pip install "git+https://github.com/elliottzheng/face-detection.git@master"
python sixdrepnet\demo.py --gpu 0 --cam 0 --snapshot checkpoints\6DRepNet_300W_LP_AFLW2000.pth
```

**路线 B：继续在 WSL 跑，需要先把摄像头直通进 WSL。**

在 Windows 管理员 PowerShell 中安装并使用 `usbipd-win`，把摄像头 USB 设备 attach 到 WSL；然后回到 WSL 确认 `ls /dev/video*` 能看到设备，再运行：

```bash
DEMO_MODE=camera bash 6DRepNet/script/run_demo.sh
```

## 7. 任务 2：AFLW2000 测试

运行：

```bash
bash 6DRepNet/script/run_eval_aflw2000.sh
```

默认配置：

```text
GPU_ID=0
BATCH_SIZE=64
NUM_WORKERS=2
SNAPSHOT=6DRepNet/checkpoints/6DRepNet_300W_LP_AFLW2000.pth
AFLW2000_DIR=6DRepNet/datasets/AFLW2000
AFLW2000_LIST=6DRepNet/datasets/AFLW2000/files.txt
```

当前脚本实际默认使用：

```text
AFLW2000_LIST=6DRepNet/datasets/AFLW2000/files_filtered_99.txt
```

如果显存不足，可以降低 batch size：

```bash
BATCH_SIZE=32 bash 6DRepNet/script/run_eval_aflw2000.sh
```

如果临时想用 CPU 测试脚本逻辑：

```bash
GPU_ID=-1 BATCH_SIZE=8 bash 6DRepNet/script/run_eval_aflw2000.sh
```

终端最后会输出类似：

```text
Device: cuda:0
Samples: 2000
Yaw: x.xxxx, Pitch: x.xxxx, Roll: x.xxxx, MAE: x.xxxx
Paper reference on AFLW2000: Yaw 3.63, Pitch 4.91, Roll 3.37, MAE 3.97
```

本地已验证，使用 `6DRepNet_300W_LP_AFLW2000.pth` 和作者过滤列表时可复现：

```text
Samples: 1969
Yaw: 3.6267, Pitch: 4.9066, Roll: 3.3740, MAE: 3.9691
```

日志保存到：

```text
6DRepNet/logs/eval_aflw2000_时间戳.log
```

将最后一行测试指标和日志截图/复制到实验报告中即可。

如果想检查所有本地 checkpoint：

```bash
bash 6DRepNet/script/run_eval_aflw2000_all_checkpoints.sh
```

已经验证的 AFLW2000 结果如下：

| checkpoint | 样本数 | Yaw | Pitch | Roll | MAE | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `6DRepNet_300W_LP_AFLW2000.pth` | 1969 | 3.6267 | 4.9066 | 3.3740 | 3.9691 | 对应论文 Table 1 的 AFLW2000 结果 |
| `6DRepNet_300W_LP_BIWI.pth` | 1969 | 3.8616 | 5.0512 | 3.5041 | 4.1390 | BIWI 方向权重，非 AFLW2000 最优 |
| `6DRepNet_70_30_BIWI.pth` | 1969 | 14.7785 | 9.8006 | 8.8275 | 11.1356 | 对应 BIWI 70/30 设置，不适合 AFLW2000 |

## 8. 报告记录建议

报告中建议记录：

1. 代码来源：作者 GitHub 仓库 `https://github.com/thohemp/6DRepNet`。
2. 方法摘要：6D 连续旋转表示、旋转矩阵、SO(3) geodesic loss、RepVGG-B1g2。
3. 本地环境：操作系统、GPU、CUDA、Python、PyTorch、torchvision、OpenCV 版本。
4. 权重文件：`6DRepNet_300W_LP_AFLW2000.pth`。
5. 数据集：AFLW2000，共 2000 张 `.jpg/.mat` 样本。
6. demo 结果图：`output/demo/aflw2000_demo.png`。
7. AFLW2000 测试结果：yaw、pitch、roll、MAE。
8. 与论文结果对比：列出论文值、复现值和误差差异。

## 9. 常见问题

### 9.1 `Missing checkpoint`

确认权重文件存在：

```text
6DRepNet/checkpoints/6DRepNet_300W_LP_AFLW2000.pth
```

### 9.2 `Missing AFLW2000 dataset directory`

确认数据集路径为：

```text
6DRepNet/datasets/AFLW2000/
```

其中应包含同名 `.jpg` 和 `.mat` 文件。

### 9.3 `ModuleNotFoundError`

先安装依赖：

```bash
cd 6DRepNet
pip install -r requirements.txt
```

### 9.4 CUDA 不可用

检查 PyTorch 是否识别 GPU：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

如果只是想检查脚本能否跑通，可以先用 CPU：

```bash
GPU_ID=-1 BATCH_SIZE=8 bash 6DRepNet/script/run_eval_aflw2000.sh
```
