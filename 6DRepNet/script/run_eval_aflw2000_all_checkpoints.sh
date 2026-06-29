#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AFLW2000_DIR="${AFLW2000_DIR:-${ROOT_DIR}/datasets/AFLW2000}"
AFLW2000_LIST="${AFLW2000_LIST:-${AFLW2000_DIR}/files_filtered_99.txt}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"

mkdir -p "${LOG_DIR}"

cd "${ROOT_DIR}"

for snapshot in "${ROOT_DIR}"/checkpoints/*.pth; do
  echo "===== $(basename "${snapshot}") ====="
  "${PYTHON_BIN}" script/eval_aflw2000.py \
    --gpu "${GPU_ID}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --snapshot "${snapshot}" \
    --data_dir "${AFLW2000_DIR}" \
    --filename_list "${AFLW2000_LIST}"
done 2>&1 | tee "${LOG_DIR}/eval_aflw2000_all_checkpoints_$(date +%Y%m%d_%H%M%S).log"
