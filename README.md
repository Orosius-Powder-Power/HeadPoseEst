# HeadPoseEstimationExplore

本仓库用于完成《数字图像处理》课程大作业：人体头部姿态估计方法学习、复现、对比实验与真实视频测试。项目围绕基线方法 **6DRepNet** 展开，并选取 CVPR 2023 方法 **TokenHPE** 作为对比方法，在 AFLW2000 数据集和本地采集视频上完成实验。

## 项目内容

本项目主要包含四部分工作：

1. 阅读并整理 6DRepNet 论文与代码，理解连续 6D 旋转表示在头部姿态估计中的作用。
2. 跑通 6DRepNet demo，并在 AFLW2000 数据集上复现论文 Table 1 结果。
3. 跑通 TokenHPE 在 AFLW2000 上的测试，作为 6DRepNet 的对比实验。
4. 使用本地采集的人体头部转动视频运行 6DRepNet，生成真实视频姿态估计结果。

## 目录结构

```text
.
├── 6DRepNet/
│   ├── docs/                         # 6DRepNet 论文、复现指南和报告素材
│   ├── script/                       # 6DRepNet 复现、评估、真实视频运行脚本
│   ├── sixdrepnet/                   # 6DRepNet 原始代码
│   ├── checkpoints/                  # 本地 checkpoint，已 gitignore
│   ├── datasets/                     # 本地数据集，已 gitignore
│   ├── input/                        # 本地真实采集视频，已 gitignore
│   └── output/                       # 输出结果，已 gitignore
├── TokenHPE/
│   ├── docs/                         # TokenHPE 复现流程参考文档
│   ├── script/                       # TokenHPE AFLW2000 测试脚本
│   ├── checkpoints/                  # 本地 checkpoint，已 gitignore
│   └── output/                       # 输出结果，已 gitignore
└── README.md
```

## 环境准备

建议在仓库根目录创建并激活虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

安装 6DRepNet 依赖：

```bash
pip install -r 6DRepNet/requirements.txt
pip install "git+https://github.com/elliottzheng/face-detection.git@master"
```

安装 TokenHPE 依赖：

```bash
pip install -r TokenHPE/requirements.txt
```

如果运行摄像头或 OpenCV GUI demo，Linux/WSL 还可能需要额外安装 `python3-tk` 以及 Qt/X11 相关系统库。

## 数据与权重

大文件不提交到 Git，需要本地自行准备：

```text
6DRepNet/checkpoints/6DRepNet_300W_LP_AFLW2000.pth
6DRepNet/datasets/AFLW2000/
TokenHPE/checkpoints/TokenHPEv1-ViTB-224_224-lyr3.tar
```

AFLW2000 测试使用过滤后的 1969 样本列表：

```text
6DRepNet/datasets/AFLW2000/files_filtered_99.txt
```

该列表遵循作者协议：过滤掉 yaw、pitch、roll 中任意角度绝对值超过 99° 的样本。

## 6DRepNet 运行

### AFLW2000 单图 demo

```bash
bash 6DRepNet/script/run_demo.sh
```

输出：

```text
6DRepNet/output/demo/aflw2000_demo.png
6DRepNet/logs/demo_image_*.log
```

### 摄像头 demo

```bash
DEMO_MODE=camera bash 6DRepNet/script/run_demo.sh
```

WSL 下需要先确认摄像头已经挂载：

```bash
ls /dev/video*
```

### AFLW2000 评估

```bash
bash 6DRepNet/script/run_eval_aflw2000.sh
```

本项目复现结果：

```text
Samples: 1969
Yaw: 3.6258, Pitch: 4.9073, Roll: 3.3737, MAE: 3.9689
Paper reference on AFLW2000: Yaw 3.63, Pitch 4.91, Roll 3.37, MAE 3.97
```

## TokenHPE 对比实验

运行 TokenHPE-v1 在 AFLW2000 上的测试：

```bash
bash TokenHPE/script/run_eval_aflw2000.sh
```

本项目复现结果：

```text
Samples: 1969
Yaw: 4.4337, Pitch: 5.7752, Roll: 4.2325, MAE: 4.8138
Vec1: 6.1332, Vec2: 5.3245, Vec3: 6.8360, VMAE: 6.0979
Paper reference on AFLW2000: MAE 4.81, VMAE 6.09
```

说明：TokenHPE 论文 Table 1 中 TokenHPE-v1 的列顺序为 `Pitch / Yaw / Roll / MAE`，脚本输出顺序为 `Yaw / Pitch / Roll / MAE`。

更详细的 TokenHPE 复现流程见：

```text
TokenHPE/docs/TokenHPE_AFLW2000复现流程参考.md
```

## 真实视频测试

将真实采集视频放入：

```text
6DRepNet/input/
```

默认脚本读取：

```text
6DRepNet/input/WIN_20260629_12_09_36_Pro.mp4
```

运行：

```bash
bash 6DRepNet/script/run_real_video.sh
```

输出：

```text
6DRepNet/output/real_video/*_6drepnet_*.mp4
6DRepNet/output/real_video/*_6drepnet_*.csv
6DRepNet/logs/real_video_*.log
```

其中 `.mp4` 是叠加姿态立方体后的结果视频，`.csv` 记录逐帧 `yaw / pitch / roll` 预测值。

可选参数：

```bash
MAX_SIZE=960 bash 6DRepNet/script/run_real_video.sh
SCORE_THRESH=0.90 bash 6DRepNet/script/run_real_video.sh
INPUT_VIDEO=6DRepNet/input/your_video.mp4 bash 6DRepNet/script/run_real_video.sh
```

## 报告材料

本仓库中保留了若干 Markdown 文档，便于整理实验报告：

```text
6DRepNet/docs/6DRepNet_复现运行指南.md
6DRepNet/docs/第七章_6DRepNet基线方法复现实验报告.md
TokenHPE/docs/TokenHPE_AFLW2000复现流程参考.md
```

## Git 忽略说明

以下内容均已写入 `.gitignore`，不会提交到仓库：

```text
6DRepNet/datasets/
6DRepNet/checkpoints/
6DRepNet/input/
6DRepNet/output/
6DRepNet/logs/
TokenHPE/checkpoints/
TokenHPE/output/
TokenHPE/logs/
*.pth
*.tar
*.mp4
*.mat
```

这样可以避免将数据集、模型权重、视频和运行输出等大文件误提交。
