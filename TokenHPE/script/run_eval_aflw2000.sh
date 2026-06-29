#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${REPO_DIR}/6DRepNet/datasets/AFLW2000}"
FILENAME_LIST="${FILENAME_LIST:-${DATA_DIR}/files_filtered_99.txt}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/checkpoints/TokenHPEv1-ViTB-224_224-lyr3.tar}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GPU="${GPU:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_ORI_TOKENS="${NUM_ORI_TOKENS:-9}"
ALLOW_REPACK="${ALLOW_REPACK:-1}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/output}"

mkdir -p "${LOG_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${LOG_DIR}/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/eval_aflw2000_${STAMP}.log"

if [[ ! -f "${MODEL_PATH}" && -d "${ROOT_DIR}/checkpoints/TokenHPEv1-ViTB-224_224-lyr3" ]]; then
  MODEL_PATH="${ROOT_DIR}/checkpoints/TokenHPEv1-ViTB-224_224-lyr3"
fi

echo "[TokenHPE eval] root=${ROOT_DIR}"
echo "[TokenHPE eval] data_dir=${DATA_DIR}"
echo "[TokenHPE eval] filename_list=${FILENAME_LIST}"
echo "[TokenHPE eval] model_path=${MODEL_PATH}"
echo "[TokenHPE eval] num_ori_tokens=${NUM_ORI_TOKENS}"
echo "[TokenHPE eval] log=${LOG_FILE}"

ARGS=(
  "${SCRIPT_DIR}/eval_aflw2000.py"
  --data_dir "${DATA_DIR}"
  --filename_list "${FILENAME_LIST}"
  --model_path "${MODEL_PATH}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --num_ori_tokens "${NUM_ORI_TOKENS}"
  --gpu "${GPU}"
)

if [[ "${ALLOW_REPACK}" == "1" ]]; then
  ARGS+=(--allow_repack)
fi

python "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
