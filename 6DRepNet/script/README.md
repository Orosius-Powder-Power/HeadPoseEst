# 6DRepNet Local Reproduction Scripts

Run from either the repository root or `6DRepNet/`; every script resolves paths automatically.

Install dependencies from the repository root:

```bash
source .venv/bin/activate
pip install -r 6DRepNet/requirements.txt
```

If your shell is already inside `6DRepNet/`, use `pip install -r requirements.txt`.

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

Camera mode also needs:

```bash
pip install "git+https://github.com/elliottzheng/face-detection.git@master"
ls /dev/video*
```

If `ls /dev/video*` prints nothing in WSL, OpenCV cannot see the Windows camera.
Use native Windows Python for the camera demo, or pass the webcam into WSL with
`usbipd-win` first.

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

The script creates `datasets/AFLW2000/files_filtered_99.txt` automatically if it is missing.
This follows the author's `create_filename_list.py` rule: samples with any yaw,
pitch, or roll angle outside `[-99, 99]` degrees are excluded.

Expected paper reference:

```text
Yaw 3.63, Pitch 4.91, Roll 3.37, MAE 3.97
```

To compare every local checkpoint on AFLW2000:

```bash
bash 6DRepNet/script/run_eval_aflw2000_all_checkpoints.sh
```
