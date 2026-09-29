#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export JAX_PLATFORMS="cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
export WANDB_MODE="${WANDB_MODE:-offline}"

CONFIG_NAME="pi05_bund_demo_data_0reset_0827_v002_v21_reconstructed_lora_local4"
EXP_NAME="${OPENPI_EXP_NAME:-ur7_pi05_lora_local4}"
PYTHON="${ROOT_DIR}/.venv/bin/python"

test -x "${PYTHON}" || {
  echo "OpenPI virtualenv Python not found: ${PYTHON}" >&2
  exit 1
}

test -d "/16T-2/sudu/code/.data/bund_demo_data_0reset_0827_v002_v21_reconstructed" || {
  echo "UR7 dataset not found" >&2
  exit 1
}

test -f "/16T-2/sudu/code/.data/bund_demo_data_0reset_0827_v002_v21_reconstructed/assets/norm_stats.json" || {
  echo "UR7 norm_stats.json not found" >&2
  exit 1
}

CUDA_VISIBLE_DEVICES=1,2 \
exec "${PYTHON}" scripts/train.py \
  "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  --batch-size 16 \
  --fsdp-devices 2 \
  --num-train-steps 80000 \
  --lr-schedule.peak-lr 6.25e-6 \
  --save-interval 10000 \
  --keep-period 10000 \
  --log-interval 100 \
  --num-workers 4 \
  --checkpoint-base-dir "/16T-1/sudu/result" \
  --overwrite
