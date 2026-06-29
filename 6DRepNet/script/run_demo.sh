#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SNAPSHOT="${SNAPSHOT:-${ROOT_DIR}/checkpoints/6DRepNet_300W_LP_AFLW2000.pth}"
AFLW2000_DIR="${AFLW2000_DIR:-${ROOT_DIR}/datasets/AFLW2000}"
GPU_ID="${GPU_ID:-0}"
CAM_ID="${CAM_ID:-0}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
CAM_FPS="${CAM_FPS:-30}"
CAM_FOURCC="${CAM_FOURCC:-MJPG}"
DEMO_MODE="${DEMO_MODE:-image}"
DEMO_SAMPLE="${DEMO_SAMPLE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/demo}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[6DRepNet demo] root=${ROOT_DIR}"
echo "[6DRepNet demo] mode=${DEMO_MODE}"
echo "[6DRepNet demo] checkpoint=${SNAPSHOT}"

if [[ ! -f "${SNAPSHOT}" ]]; then
  echo "Missing checkpoint: ${SNAPSHOT}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import cv2
import numpy
import scipy
import torch
import torchvision
from PIL import Image
PY
then
  echo "Missing Python dependencies. Please run:" >&2
  echo "  cd ${ROOT_DIR}" >&2
  echo "  pip install -r requirements.txt" >&2
  echo "For camera mode also run:" >&2
  echo "  pip install \"git+https://github.com/elliottzheng/face-detection.git@master\"" >&2
  exit 1
fi

if [[ "${DEMO_MODE}" == "camera" ]]; then
  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import face_detection
PY
  then
    echo "Missing camera-demo dependency: face_detection" >&2
    echo "Please run:" >&2
    echo "  pip install \"git+https://github.com/elliottzheng/face-detection.git@master\"" >&2
    exit 1
  fi

  if [[ -n "${WSL_DISTRO_NAME:-}" ]] && ! ls /dev/video* >/dev/null 2>&1; then
    echo "No camera device was found in WSL: /dev/video* does not exist." >&2
    echo "OpenCV cannot access the Windows webcam from this WSL session yet." >&2
    echo "Use one of these options:" >&2
    echo "  1) Run the camera demo in native Windows Python/Anaconda." >&2
    echo "  2) Pass the webcam into WSL with usbipd-win, then confirm: ls /dev/video*" >&2
    exit 1
  fi
fi

cd "${ROOT_DIR}"

if [[ "${DEMO_MODE}" == "camera" ]]; then
  echo "[6DRepNet demo] running original camera demo. Press Esc in the OpenCV window to exit."
  "${PYTHON_BIN}" sixdrepnet/demo.py \
    --gpu "${GPU_ID}" \
    --cam "${CAM_ID}" \
    --cam_width "${CAM_WIDTH}" \
    --cam_height "${CAM_HEIGHT}" \
    --cam_fps "${CAM_FPS}" \
    --cam_fourcc "${CAM_FOURCC}" \
    --snapshot "${SNAPSHOT}" \
    2>&1 | tee "${LOG_DIR}/demo_camera_$(date +%Y%m%d_%H%M%S).log"
else
  echo "[6DRepNet demo] running deterministic image demo on AFLW2000."
  "${PYTHON_BIN}" script/demo_image.py \
    --gpu "${GPU_ID}" \
    --snapshot "${SNAPSHOT}" \
    --data_dir "${AFLW2000_DIR}" \
    --sample "${DEMO_SAMPLE}" \
    --output "${OUTPUT_DIR}/aflw2000_demo.png" \
    2>&1 | tee "${LOG_DIR}/demo_image_$(date +%Y%m%d_%H%M%S).log"
fi
