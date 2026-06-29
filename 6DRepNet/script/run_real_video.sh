#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SNAPSHOT="${SNAPSHOT:-${ROOT_DIR}/checkpoints/6DRepNet_300W_LP_AFLW2000.pth}"
INPUT_VIDEO="${INPUT_VIDEO:-${ROOT_DIR}/input/WIN_20260629_12_09_36_Pro.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/real_video}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
GPU_ID="${GPU_ID:-0}"
SCORE_THRESH="${SCORE_THRESH:-0.95}"
MAX_SIZE="${MAX_SIZE:-1280}"
BASENAME="$(basename "${INPUT_VIDEO}")"
BASENAME="${BASENAME%.*}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-${OUTPUT_DIR}/${BASENAME}_6drepnet_${STAMP}.mp4}"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/${BASENAME}_6drepnet_${STAMP}.csv}"
LOG_FILE="${LOG_DIR}/real_video_${STAMP}.log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[6DRepNet real video] root=${ROOT_DIR}"
echo "[6DRepNet real video] input=${INPUT_VIDEO}"
echo "[6DRepNet real video] output_video=${OUTPUT_VIDEO}"
echo "[6DRepNet real video] output_csv=${OUTPUT_CSV}"
echo "[6DRepNet real video] checkpoint=${SNAPSHOT}"
echo "[6DRepNet real video] log=${LOG_FILE}"

if [[ ! -f "${INPUT_VIDEO}" ]]; then
  echo "Missing input video: ${INPUT_VIDEO}" >&2
  exit 1
fi

if [[ ! -f "${SNAPSHOT}" ]]; then
  echo "Missing checkpoint: ${SNAPSHOT}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import cv2
import face_detection
import numpy
import torch
import torchvision
from PIL import Image
PY
then
  echo "Missing Python dependencies. Please run:" >&2
  echo "  source .venv/bin/activate" >&2
  echo "  pip install -r 6DRepNet/requirements.txt" >&2
  echo "  pip install \"git+https://github.com/elliottzheng/face-detection.git@master\"" >&2
  exit 1
fi

"${PYTHON_BIN}" "${ROOT_DIR}/script/demo_video.py" \
  --input "${INPUT_VIDEO}" \
  --output "${OUTPUT_VIDEO}" \
  --csv "${OUTPUT_CSV}" \
  --snapshot "${SNAPSHOT}" \
  --gpu "${GPU_ID}" \
  --score_thresh "${SCORE_THRESH}" \
  --max_size "${MAX_SIZE}" \
  2>&1 | tee "${LOG_FILE}"
