#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SNAPSHOT="${SNAPSHOT:-${ROOT_DIR}/checkpoints/6DRepNet_300W_LP_AFLW2000.pth}"
AFLW2000_DIR="${AFLW2000_DIR:-${ROOT_DIR}/datasets/AFLW2000}"
GPU_ID="${GPU_ID:-0}"
CAM_ID="${CAM_ID:-0}"
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

cd "${ROOT_DIR}"

if [[ "${DEMO_MODE}" == "camera" ]]; then
  echo "[6DRepNet demo] running original camera demo. Press Esc in the OpenCV window to exit."
  "${PYTHON_BIN}" sixdrepnet/demo.py \
    --gpu "${GPU_ID}" \
    --cam "${CAM_ID}" \
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
