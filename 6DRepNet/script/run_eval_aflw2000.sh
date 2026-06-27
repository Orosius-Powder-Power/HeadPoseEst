#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SNAPSHOT="${SNAPSHOT:-${ROOT_DIR}/checkpoints/6DRepNet_300W_LP_AFLW2000.pth}"
AFLW2000_DIR="${AFLW2000_DIR:-${ROOT_DIR}/datasets/AFLW2000}"
AFLW2000_LIST="${AFLW2000_LIST:-${AFLW2000_DIR}/files.txt}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"

mkdir -p "${LOG_DIR}"

echo "[6DRepNet AFLW2000 eval] root=${ROOT_DIR}"
echo "[6DRepNet AFLW2000 eval] checkpoint=${SNAPSHOT}"
echo "[6DRepNet AFLW2000 eval] data=${AFLW2000_DIR}"
echo "[6DRepNet AFLW2000 eval] filename_list=${AFLW2000_LIST}"

if [[ ! -f "${SNAPSHOT}" ]]; then
  echo "Missing checkpoint: ${SNAPSHOT}" >&2
  exit 1
fi
if [[ ! -d "${AFLW2000_DIR}" ]]; then
  echo "Missing AFLW2000 dataset directory: ${AFLW2000_DIR}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import cv2
import numpy
import pandas
import scipy
import torch
import torchvision
from PIL import Image
PY
then
  echo "Missing Python dependencies. Please run:" >&2
  echo "  cd ${ROOT_DIR}" >&2
  echo "  pip install -r requirements.txt" >&2
  exit 1
fi

cd "${ROOT_DIR}"

"${PYTHON_BIN}" script/eval_aflw2000.py \
  --gpu "${GPU_ID}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --snapshot "${SNAPSHOT}" \
  --data_dir "${AFLW2000_DIR}" \
  --filename_list "${AFLW2000_LIST}" \
  2>&1 | tee "${LOG_DIR}/eval_aflw2000_$(date +%Y%m%d_%H%M%S).log"
