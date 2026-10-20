#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Eight H20 devices.  Set CUDA_VISIBLE_DEVICES explicitly when using a
# scheduler allocation with non-contiguous device IDs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
gpu_count="$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
if [[ "${gpu_count}" != 8 ]]; then
  echo "8-GPU TTRTC config requires eight CUDA devices, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
export WANDB_MODE="${WANDB_MODE:-offline}"

CONFIG_NAME="pi05_bund_20260917_n906_v21_new_80G_8gpu_full_ttrtc"
EXP_NAME="${OPENPI_EXP_NAME:-${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${OPENPI_PYTHON:-${ROOT_DIR}/.venv/bin/python}"

if [[ "${OPENPI_RESUME:-0}" == 1 && -z "${OPENPI_EXP_NAME:-}" ]]; then
  echo "OPENPI_RESUME=1 requires OPENPI_EXP_NAME to identify an existing run" >&2
  exit 1
fi

test -x "${PYTHON}" || {
  echo "OpenPI virtualenv Python not found: ${PYTHON}" >&2
  exit 1
}

train_flags=(--overwrite)
if [[ "${OPENPI_RESUME:-0}" == 1 ]]; then
  train_flags=(--resume)
fi

exec "${PYTHON}" scripts/train.py \
  "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  "${train_flags[@]}" \
  "$@"
