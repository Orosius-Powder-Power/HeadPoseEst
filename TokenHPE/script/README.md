# TokenHPE AFLW2000 复现脚本

本目录用于复现 TokenHPE 在 AFLW2000 上的测试结果，并与前面 6DRepNet 使用同一个 AFLW2000 数据目录进行公平对比。

## 1. 安装依赖

在总工程根目录激活虚拟环境后执行：

```bash
source .venv/bin/activate
pip install -r TokenHPE/requirements.txt
```

TokenHPE 原始 `requirements.txt` 缺少 `timm` 和 `einops`，本复现已经补充。

## 2. 数据与权重

默认复用 6DRepNet 已经准备好的 AFLW2000 数据：

```text
6DRepNet/datasets/AFLW2000
6DRepNet/datasets/AFLW2000/files_filtered_99.txt
```

其中 `files_filtered_99.txt` 是与 6DRepNet/TokenHPE 作者协议一致的 1969 样本测试列表。

默认 TokenHPE 权重路径为：

```text
TokenHPE/checkpoints/TokenHPEv1-ViTB-224_224-lyr3.tar
```

如果该权重被误解压成：

```text
TokenHPE/checkpoints/TokenHPEv1-ViTB-224_224-lyr3/archive/
```

脚本会在第一次运行时自动重新打包成可被 `torch.load` 读取的 `.tar` 文件。

## 3. 一键运行

```bash
bash TokenHPE/script/run_eval_aflw2000.sh
```

运行结果会保存到：

```text
TokenHPE/output/eval_aflw2000_时间戳.log
```

论文/README 中 AFLW2000 参考结果为：

```text
MAE 4.81, VMAE 6.09
```

可以按需覆盖参数：

```bash
BATCH_SIZE=16 GPU=0 bash TokenHPE/script/run_eval_aflw2000.sh
```

## 4. TokenHPE-v2

论文中的 TokenHPE-v1 使用 9 个 orientation tokens，TokenHPE-v2 使用 11 个 orientation tokens。当前默认脚本对应 v1：

```bash
NUM_ORI_TOKENS=9 bash TokenHPE/script/run_eval_aflw2000.sh
```

如果后续下载到 v2 checkpoint，可以这样运行：

```bash
NUM_ORI_TOKENS=11 MODEL_PATH=TokenHPE/checkpoints/TokenHPEv2-ViTB-224_224-lyr3.tar bash TokenHPE/script/run_eval_aflw2000.sh
```

不能直接用 v1 checkpoint 跑 v2，因为 orientation token 数量不同，`ori_tokens`、位置编码和 MLP head 的参数形状都不同。
