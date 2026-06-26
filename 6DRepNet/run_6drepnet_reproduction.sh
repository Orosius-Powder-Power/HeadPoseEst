#!/usr/bin/env bash
set -euo pipefail

# 6DRepNet reproduction helper for the Digital Image Processing assignment.
# Run this script from the 6DRepNet directory on the server after placing
# datasets and checkpoints under the paths described by:
#   bash run_6drepnet_reproduction.sh downloads
#
# Recommended layout:
#   6DRepNet/
#     datasets/
#       AFLW2000/
#         image00002.jpg
#         image00002.mat
#         files.txt
#       BIWI/
#         BIWI.npz
#     checkpoints/
#       6DRepNet_300W_LP_AFLW2000.pth
#     logs/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/datasets}"
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"

DEFAULT_SNAPSHOT="${CKPT_DIR}/6DRepNet_300W_LP_AFLW2000.pth"
AFLW2000_DIR="${AFLW2000_DIR:-${DATA_ROOT}/AFLW2000}"
AFLW2000_LIST="${AFLW2000_LIST:-${AFLW2000_DIR}/files.txt}"
BIWI_NPZ="${BIWI_NPZ:-${DATA_ROOT}/BIWI/BIWI.npz}"
SNAPSHOT="${SNAPSHOT:-${DEFAULT_SNAPSHOT}}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CAM_ID="${CAM_ID:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash run_6drepnet_reproduction.sh <command>

Commands:
  paper-summary       Print the method and expected paper results.
  downloads           Print every dataset/checkpoint that must be downloaded.
  setup               Create venv and install Python dependencies.
  check               Check required local files for demo/evaluation.
  make-aflw-list      Generate datasets/AFLW2000/files.txt.
  demo                Run the real-time camera demo.
  eval-aflw2000       Evaluate on AFLW2000.
  eval-biwi           Evaluate on BIWI .npz.
  all-eval            Run eval-aflw2000 and eval-biwi.

Environment variables:
  PYTHON_BIN          Python executable, default: python3
  VENV_DIR            Virtualenv path, default: 6DRepNet/.venv
  DATA_ROOT           Dataset root, default: 6DRepNet/datasets
  CKPT_DIR            Checkpoint root, default: 6DRepNet/checkpoints
  SNAPSHOT            Model weight path, default: checkpoints/6DRepNet_300W_LP_AFLW2000.pth
  GPU_ID              CUDA device id, default: 0
  BATCH_SIZE          Evaluation batch size, default: 64
  CAM_ID              Camera id for demo, default: 0
  AFLW2000_DIR        AFLW2000 folder, default: datasets/AFLW2000
  AFLW2000_LIST       AFLW2000 file list, default: datasets/AFLW2000/files.txt
  BIWI_NPZ            BIWI preprocessed npz, default: datasets/BIWI/BIWI.npz

Typical server workflow:
  cd 6DRepNet
  bash run_6drepnet_reproduction.sh downloads
  bash run_6drepnet_reproduction.sh setup
  # Place datasets/checkpoints as printed by downloads.
  source .venv/bin/activate
  bash run_6drepnet_reproduction.sh make-aflw-list
  bash run_6drepnet_reproduction.sh demo
  bash run_6drepnet_reproduction.sh all-eval
EOF
}

paper_summary() {
  cat <<'EOF'
6DRepNet paper summary:
  Core problem:
    Euler angles have gimbal-lock and equivalent-label ambiguity. Quaternions
    remove gimbal-lock but still have antipodal ambiguity. These discontinuities
    are harmful for direct neural-network regression, especially for wide-range
    pose.

  Core method:
    1. Use RepVGG-B1g2 as the image backbone.
    2. Regress 6 continuous numbers instead of yaw/pitch/roll.
    3. Interpret the 6 numbers as the first two 3D vectors of a rotation matrix.
    4. Map them to SO(3) with a Gram-Schmidt style orthogonalization:
       normalize the first vector, remove its projection from the second vector,
       normalize the second vector, and recover the third vector by cross product.
    5. Train with geodesic distance on SO(3):
       acos((trace(R_pred * R_gt^T) - 1) / 2).
    6. Convert the predicted rotation matrix back to Euler angles only for
       reporting yaw/pitch/roll MAE.

  Paper baseline to reproduce:
    Train source: 300W-LP.
    Test datasets selected for this assignment: AFLW2000 and BIWI.
    Reported AFLW2000 result: Yaw 3.63, Pitch 4.91, Roll 3.37, MAE 3.97.
    Reported BIWI result:     Yaw 3.24, Pitch 4.48, Roll 2.68, MAE 3.47.

  Note:
    This repository's README says `python test.py`, but the root test.py is a
    small pip-usage example. The real evaluation entry is sixdrepnet/test.py.
EOF
}

downloads() {
  cat <<EOF
Download and place these files on the server:

1. Code
   Official repository:
     https://github.com/thohemp/6DRepNet
   This course workspace already contains a copy under:
     ${ROOT_DIR}

2. Checkpoint for demo and table-1 evaluation
   Download the fine-tuned model folder from the author's README:
     https://drive.google.com/drive/folders/1V1pCV0BEW3mD-B9MogGrz_P91UhTtuE_?usp=sharing
   Put the 300W-LP/AFLW2000 deploy checkpoint here:
     ${DEFAULT_SNAPSHOT}
   The README demo command names this file:
     6DRepNet_300W_LP_AFLW2000.pth

3. AFLW2000
   Download AFLW2000 from the 3DDFA project page:
     http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
   Unpack so that jpg/mat pairs are directly under:
     ${AFLW2000_DIR}
   Then run:
     bash run_6drepnet_reproduction.sh make-aflw-list

4. BIWI
   The author README points to an old ETH page:
     https://icu.ee.ethz.ch/research/datsets.html
   As of 2026-06-26 this page may return "Page not found". This is usually an
   upstream ETH page move/removal, not a local download mistake. Also try:
     https://icu.ee.ethz.ch/research/datasets.html
   Search terms:
     BIWI Kinect Head Pose Database
     BIWI head pose dataset

   Preprocess it to cropped face images in npz format. The author README points
   to FSA-Net preprocessing scripts:
     https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi.py
     https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi_70_30.py
   Use crop size 256, consistent with the README. Place the table-1 BIWI test
   npz at:
     ${BIWI_NPZ}

5. Optional training backbone
   Only needed if training from scratch, not for demo/evaluation:
     RepVGG-B1g2-train.pth
   Author link:
     https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq
   Put it under:
     ${ROOT_DIR}/RepVGG-B1g2-train.pth

Large files are intentionally ignored by git. Keep datasets, checkpoints,
virtualenvs, outputs, and logs on the server filesystem.
EOF
}

setup_env() {
  cd "${ROOT_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
  pip install "git+https://github.com/elliottzheng/face-detection.git@master"
  mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${DATA_ROOT}"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    return 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    return 1
  fi
}

check_files() {
  local ok=0
  require_file "${SNAPSHOT}" "checkpoint" || ok=1
  require_dir "${AFLW2000_DIR}" "AFLW2000 directory" || ok=1
  require_file "${AFLW2000_LIST}" "AFLW2000 filename list" || ok=1
  require_file "${BIWI_NPZ}" "BIWI npz" || ok=1
  if [[ "${ok}" -ne 0 ]]; then
    echo
    echo "Run the downloads command for the expected layout."
    exit 1
  fi
  echo "All required files are present."
}

make_aflw_list() {
  cd "${ROOT_DIR}"
  require_dir "${AFLW2000_DIR}" "AFLW2000 directory"
  python sixdrepnet/create_filename_list.py \
    --root_dir "${AFLW2000_DIR}" \
    --filename "$(basename "${AFLW2000_LIST}")"
}

run_demo() {
  cd "${ROOT_DIR}"
  mkdir -p "${LOG_DIR}"
  require_file "${SNAPSHOT}" "checkpoint"
  python sixdrepnet/demo.py \
    --gpu "${GPU_ID}" \
    --cam "${CAM_ID}" \
    --snapshot "${SNAPSHOT}" \
    2>&1 | tee "${LOG_DIR}/demo_$(date +%Y%m%d_%H%M%S).log"
}

eval_aflw2000() {
  cd "${ROOT_DIR}"
  mkdir -p "${LOG_DIR}"
  require_file "${SNAPSHOT}" "checkpoint"
  require_dir "${AFLW2000_DIR}" "AFLW2000 directory"
  require_file "${AFLW2000_LIST}" "AFLW2000 filename list"
  python sixdrepnet/test.py \
    --gpu "${GPU_ID}" \
    --batch_size "${BATCH_SIZE}" \
    --dataset AFLW2000 \
    --data_dir "${AFLW2000_DIR}" \
    --filename_list "${AFLW2000_LIST}" \
    --snapshot "${SNAPSHOT}" \
    --show_viz False \
    2>&1 | tee "${LOG_DIR}/eval_aflw2000_$(date +%Y%m%d_%H%M%S).log"
}

eval_biwi() {
  cd "${ROOT_DIR}"
  mkdir -p "${LOG_DIR}"
  require_file "${SNAPSHOT}" "checkpoint"
  require_file "${BIWI_NPZ}" "BIWI npz"
  python sixdrepnet/test.py \
    --gpu "${GPU_ID}" \
    --batch_size "${BATCH_SIZE}" \
    --dataset BIWI \
    --data_dir "${DATA_ROOT}/BIWI" \
    --filename_list "${BIWI_NPZ}" \
    --snapshot "${SNAPSHOT}" \
    --show_viz False \
    2>&1 | tee "${LOG_DIR}/eval_biwi_$(date +%Y%m%d_%H%M%S).log"
}

main() {
  local command="${1:-}"
  case "${command}" in
    paper-summary) paper_summary ;;
    downloads) downloads ;;
    setup) setup_env ;;
    check) check_files ;;
    make-aflw-list) make_aflw_list ;;
    demo) run_demo ;;
    eval-aflw2000) eval_aflw2000 ;;
    eval-biwi) eval_biwi ;;
    all-eval)
      eval_aflw2000
      eval_biwi
      ;;
    ""|-h|--help|help) usage ;;
    *)
      echo "Unknown command: ${command}" >&2
      usage
      exit 2
      ;;
  esac
}

main "$@"
