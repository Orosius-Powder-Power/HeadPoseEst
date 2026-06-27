# 6DRepNet Local Reproduction Scripts

Run from either the repository root or `6DRepNet/`; every script resolves paths automatically.

## 1. Demo

Default: run a deterministic image demo on one AFLW2000 sample and save a visualization.

```bash
bash 6DRepNet/script/run_demo.sh
```

Output:

```text
6DRepNet/output/demo/aflw2000_demo.png
6DRepNet/logs/demo_image_*.log
```

Optional camera mode, matching the author's original `sixdrepnet/demo.py`:

```bash
DEMO_MODE=camera bash 6DRepNet/script/run_demo.sh
```

Useful environment variables:

```bash
GPU_ID=0
CAM_ID=0
DEMO_SAMPLE=image00002
SNAPSHOT=6DRepNet/checkpoints/6DRepNet_300W_LP_AFLW2000.pth
AFLW2000_DIR=6DRepNet/datasets/AFLW2000
```

## 2. AFLW2000 Evaluation

```bash
bash 6DRepNet/script/run_eval_aflw2000.sh
```

The script creates `datasets/AFLW2000/files.txt` automatically if it is missing.

Expected paper reference:

```text
Yaw 3.63, Pitch 4.91, Roll 3.37, MAE 3.97
```
