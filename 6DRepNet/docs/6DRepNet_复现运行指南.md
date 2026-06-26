# 6DRepNet 复现运行指南

本文档用于完成数字图像处理大作业中的“基线方法复现与创新设计”部分，目标是先跑通 6DRepNet 项目的测试 demo，并在论文使用的数据集上完成性能测试。本文默认本地 WSL 用于代码整理，真正的数据集、checkpoint 下载和实验运行放在 GPU 服务器上完成。

## 1. 论文与项目核心理解

6DRepNet 关注的是单张人脸图像的头部姿态估计，输出 yaw、pitch、roll 三个角度。传统方法常直接回归欧拉角，或者使用四元数表示旋转。论文指出，这些表示存在不连续或多解问题：欧拉角会遇到万向节锁以及同一姿态对应多组角度的问题，四元数也存在 antipodal ambiguity，即 `q` 和 `-q` 可表示同一个旋转。这些不连续性会让神经网络的直接回归更难学，特别是在 unconstrained、wide-range head pose 场景下。

6DRepNet 的核心思想是：网络不直接回归欧拉角，而是回归一个连续的 6D 旋转表示。代码中 `sixdrepnet/model.py` 的 `SixDRepNet.forward()` 输出 6 个数后，调用 `utils.compute_rotation_matrix_from_ortho6d()`，将前 3 个数看作第一个方向向量，后 3 个数看作第二个方向向量，再通过归一化、叉乘等操作得到合法的 `3 x 3` 旋转矩阵。训练时不在欧拉角空间计算损失，而是在 SO(3) 旋转流形上使用 geodesic loss，代码位置是 `sixdrepnet/loss.py`。测试时再把旋转矩阵转换回欧拉角，计算 yaw、pitch、roll 的 MAE。

本项目中的主干网络是 RepVGG-B1g2。作者论文主实验使用 300W-LP 训练，在 AFLW2000 和 BIWI 上测试。本文选择 AFLW2000 与 BIWI 作为课程复现的两个数据集，对应论文 README 中的表格结果：

| 数据集 | Yaw | Pitch | Roll | MAE |
| --- | ---: | ---: | ---: | ---: |
| AFLW2000 | 3.63 | 4.91 | 3.37 | 3.97 |
| BIWI | 3.24 | 4.48 | 2.68 | 3.47 |

实际复现实验允许存在小幅波动，主要受 checkpoint 版本、BIWI 预处理方式、PyTorch/CUDA 版本和图像裁剪细节影响。

## 2. 项目关键文件

在本仓库中，6DRepNet 相关代码位于 `6DRepNet/`：

| 路径 | 作用 |
| --- | --- |
| `sixdrepnet/demo.py` | 摄像头实时 demo，使用 RetinaFace 检测人脸，再画出姿态立方体 |
| `sixdrepnet/test.py` | 真正的评估入口，支持 AFLW2000、BIWI 等数据集 |
| `test.py` | pip 安装包的简单调用示例，不是本文使用的评估入口 |
| `sixdrepnet/model.py` | SixDRepNet 网络定义，输出 6D 表示并转为旋转矩阵 |
| `sixdrepnet/utils.py` | 6D 到旋转矩阵、旋转矩阵到欧拉角、绘图等工具函数 |
| `sixdrepnet/loss.py` | geodesic loss |
| `sixdrepnet/datasets.py` | AFLW2000、BIWI、300W-LP 等数据读取逻辑 |
| `sixdrepnet/create_filename_list.py` | 为 AFLW2000/300W-LP 生成 `files.txt` |
| `run_6drepnet_reproduction.sh` | 本次新增的服务器复现脚本 |

需要特别注意：作者 README 中写了 `python test.py`，但当前仓库根目录的 `test.py` 实际是包使用示例。带 `--dataset`、`--snapshot`、`--filename_list` 参数的评估程序是 `sixdrepnet/test.py`。本文和脚本均使用后者。

## 3. 服务器目录规划

建议在服务器上保持如下目录结构：

```text
6DRepNet/
  run_6drepnet_reproduction.sh
  requirements.txt
  sixdrepnet/
  datasets/
    AFLW2000/
      image00002.jpg
      image00002.mat
      ...
      files.txt
    BIWI/
      BIWI.npz
  checkpoints/
    6DRepNet_300W_LP_AFLW2000.pth
  logs/
```

其中 `datasets/`、`checkpoints/`、`logs/`、`.venv/`、`output/` 都已经写入 `.gitignore`，不要提交到 git。

## 4. 需要下载的内容

### 4.1 作者代码库

论文作者公开代码库：

```text
https://github.com/thohemp/6DRepNet
```

当前课程仓库已经包含了该项目副本，服务器上直接使用本仓库中的 `6DRepNet/` 即可。

### 4.2 checkpoint

下载作者 README 中的 fine-tuned models：

```text
https://drive.google.com/drive/folders/1V1pCV0BEW3mD-B9MogGrz_P91UhTtuE_?usp=sharing
```

将用于 300W-LP 训练、AFLW2000/BIWI 测试的 deploy 权重放到：

```text
6DRepNet/checkpoints/6DRepNet_300W_LP_AFLW2000.pth
```

如果下载到的文件名不同，可以保留原名，但运行时需要用环境变量 `SNAPSHOT=/path/to/xxx.pth` 指定。

### 4.3 AFLW2000

从 3DDFA 项目页面下载 AFLW2000：

```text
http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
```

解压后让 `.jpg` 与 `.mat` 文件直接位于：

```text
6DRepNet/datasets/AFLW2000/
```

随后用脚本生成 `files.txt`。

### 4.4 BIWI

作者 README 给出的 BIWI 下载入口是 ETH 的旧页面：

```text
https://icu.ee.ethz.ch/research/datsets.html
```

截至 2026-06-26，这个页面在浏览器中可能返回 ETH 的 `Page not found`。这通常不是网络问题，而是 ETH 网站改版或旧页面被移动导致的；另外原始链接里的 `datsets.html` 本身也疑似是 `datasets.html` 的拼写错误。

建议按以下顺序处理：

1. 先尝试修正拼写后的页面，或在 ETH/Google 中搜索关键词：

```text
https://icu.ee.ethz.ch/research/datasets.html
BIWI Kinect Head Pose Database
BIWI head pose dataset
```

2. 如果 ETH 页面仍然不可访问，记录“官方数据集链接失效，访问日期为 2026-06-26”，并优先完成 AFLW2000 的复现。报告中可以把 BIWI 列为因官方源失效而未完成下载的数据集，同时保留本仓库中的 `eval-biwi` 脚本，说明一旦取得 `BIWI.npz` 即可复跑。

3. 如果课程必须提交两个数据集的运行结果，可向助教确认是否允许用作者代码支持的其他数据集替代，例如 AFW/AFLW，或者由课程/同学提供已下载的 BIWI 原始数据或预处理 `.npz`。不要把从非官方来源获取的大型数据文件提交到 git。

作者 README 说明 BIWI 需要先用人脸检测器裁剪并保存为 `.npz`。可参考 FSA-Net 的预处理脚本：

```text
https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi.py
https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi_70_30.py
```

裁剪图像尺寸设置为 `256`，与作者 README 保持一致。本文测试入口默认读取：

```text
6DRepNet/datasets/BIWI/BIWI.npz
```

该 `.npz` 内部需要包含 `image` 和 `pose` 两个数组，代码读取位置在 `sixdrepnet/datasets.py` 的 `BIWI` 类。

## 5. 构建环境

进入服务器上的项目目录：

```bash
cd 6DRepNet
```

运行环境构建命令：

```bash
bash run_6drepnet_reproduction.sh setup
```

该命令会执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install "git+https://github.com/elliottzheng/face-detection.git@master"
```

如果服务器已有统一的 PyTorch/CUDA 环境，也可以不使用 `.venv`，只需要确保安装了 `requirements.txt` 中的依赖和 `face-detection` 包。demo 必须安装 `face_detection.RetinaFace`，否则 `sixdrepnet/demo.py` 无法运行。

之后每次运行前激活环境：

```bash
source .venv/bin/activate
```

## 6. 检查下载文件

放好 checkpoint 和数据集后运行：

```bash
bash run_6drepnet_reproduction.sh check
```

如果有文件缺失，脚本会输出缺少的路径。也可以先查看下载清单：

```bash
bash run_6drepnet_reproduction.sh downloads
```

## 7. 生成 AFLW2000 文件列表

AFLW2000 评估需要 `files.txt`，每一行是样本相对路径，不带 `.jpg` 和 `.mat` 后缀。运行：

```bash
bash run_6drepnet_reproduction.sh make-aflw-list
```

生成后应得到：

```text
6DRepNet/datasets/AFLW2000/files.txt
```

如果数据集路径不同，可以这样指定：

```bash
AFLW2000_DIR=/data/AFLW2000 \
AFLW2000_LIST=/data/AFLW2000/files.txt \
bash run_6drepnet_reproduction.sh make-aflw-list
```

## 8. 跑通摄像头 demo

确认服务器能访问摄像头或视频采集设备后运行：

```bash
bash run_6drepnet_reproduction.sh demo
```

默认使用 `GPU_ID=0`、`CAM_ID=0` 和：

```text
checkpoints/6DRepNet_300W_LP_AFLW2000.pth
```

如果需要指定其他 GPU、摄像头或 checkpoint：

```bash
GPU_ID=1 \
CAM_ID=0 \
SNAPSHOT=checkpoints/6DRepNet_300W_LP_AFLW2000.pth \
bash run_6drepnet_reproduction.sh demo
```

运行成功时，窗口中会显示摄像头画面；检测到人脸后，会在人脸区域绘制 3D 姿态立方体，并在终端输出类似：

```text
Head pose estimation: xx.xxxxxx ms
```

demo 日志会保存到：

```text
6DRepNet/logs/demo_时间戳.log
```

如果服务器没有图形界面或摄像头，demo 可能因为 OpenCV 窗口或摄像头打开失败而退出。课程报告中可记录该限制；真正的性能复现以第 9 节测试结果为准。

## 9. 跑通性能测试

### 9.1 AFLW2000 测试

运行：

```bash
bash run_6drepnet_reproduction.sh eval-aflw2000
```

等价的核心命令是：

```bash
python sixdrepnet/test.py \
  --gpu 0 \
  --batch_size 64 \
  --dataset AFLW2000 \
  --data_dir datasets/AFLW2000 \
  --filename_list datasets/AFLW2000/files.txt \
  --snapshot checkpoints/6DRepNet_300W_LP_AFLW2000.pth \
  --show_viz False
```

终端最后会输出：

```text
Yaw: x.xxxx, Pitch: x.xxxx, Roll: x.xxxx, MAE: x.xxxx
```

将该行截图或复制到实验报告。参考论文目标结果为：

```text
Yaw: 3.63, Pitch: 4.91, Roll: 3.37, MAE: 3.97
```

日志保存到：

```text
6DRepNet/logs/eval_aflw2000_时间戳.log
```

### 9.2 BIWI 测试

运行：

```bash
bash run_6drepnet_reproduction.sh eval-biwi
```

等价的核心命令是：

```bash
python sixdrepnet/test.py \
  --gpu 0 \
  --batch_size 64 \
  --dataset BIWI \
  --data_dir datasets/BIWI \
  --filename_list datasets/BIWI/BIWI.npz \
  --snapshot checkpoints/6DRepNet_300W_LP_AFLW2000.pth \
  --show_viz False
```

参考论文目标结果为：

```text
Yaw: 3.24, Pitch: 4.48, Roll: 2.68, MAE: 3.47
```

日志保存到：

```text
6DRepNet/logs/eval_biwi_时间戳.log
```

### 9.3 一次运行两个测试

```bash
bash run_6drepnet_reproduction.sh all-eval
```

## 10. 实验报告记录建议

报告中建议记录以下内容：

1. 代码来源：作者 GitHub 仓库 `https://github.com/thohemp/6DRepNet`，本仓库保存课程复现副本。
2. 论文方法摘要：6D 连续旋转表示、SO(3) geodesic loss、RepVGG-B1g2 主干。
3. 环境信息：操作系统、GPU、CUDA、Python、PyTorch、torchvision、OpenCV 版本。
4. 权重文件：`6DRepNet_300W_LP_AFLW2000.pth`，来自作者 fine-tuned models。
5. 数据集：AFLW2000 原始 `.jpg/.mat`，BIWI 预处理 `.npz`，说明 BIWI 裁剪尺寸为 256。
6. demo 运行截图：显示人脸姿态立方体。
7. AFLW2000 与 BIWI 的终端输出结果：包含 yaw、pitch、roll、MAE。
8. 与论文结果对比：用表格列出论文值、复现值和差值。

## 11. 常见问题

### 11.1 `ModuleNotFoundError: No module named 'face_detection'`

说明没有安装 RetinaFace 检测器。运行：

```bash
pip install "git+https://github.com/elliottzheng/face-detection.git@master"
```

### 11.2 `Cannot open webcam`

说明服务器没有可用摄像头，或 `CAM_ID` 不正确。可以改用：

```bash
CAM_ID=1 bash run_6drepnet_reproduction.sh demo
```

如果服务器没有图形界面和摄像头，先完成性能测试；报告中说明 demo 在有摄像头的机器上运行。

### 11.3 `Missing checkpoint`

把 checkpoint 放到：

```text
6DRepNet/checkpoints/6DRepNet_300W_LP_AFLW2000.pth
```

或者运行时指定：

```bash
SNAPSHOT=/absolute/path/to/model.pth bash run_6drepnet_reproduction.sh eval-aflw2000
```

### 11.4 AFLW2000 找不到 `.mat`

确认 `files.txt` 的每一行不带后缀，并且同名 `.jpg` 和 `.mat` 都在 `AFLW2000_DIR` 下。必要时重新运行：

```bash
bash run_6drepnet_reproduction.sh make-aflw-list
```

### 11.5 BIWI `.npz` 键名不匹配

当前代码要求 `.npz` 内有：

```text
image
pose
```

如果预处理脚本输出的键名不同，需要在生成 `.npz` 时改名，或修改 `sixdrepnet/datasets.py` 中 `BIWI` 类的读取逻辑。
