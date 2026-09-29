#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export JAX_PLATFORMS="cuda"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
export WANDB_MODE="${WANDB_MODE:-offline}"

CONFIG_NAME="pi05_bund_demo_data_0reset_0827_v002_v21_reconstructed"
EXP_NAME="${OPENPI_EXP_NAME:-bund_demo_data_0reset_0827_v002_v21_reconstructed_pi05}"

# uv run --frozen --no-build-isolation --offline scripts/train.py \
#   "${CONFIG_NAME}" \
#   --exp-name "${EXP_NAME}" \
#   --overwrite

.venv/bin/python scripts/train.py \
  "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  --overwrite
