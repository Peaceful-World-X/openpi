#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false

config_name=pi05_ebench_task26
policy_dir=/shared_disk/users/wenyao.xue/results/openpi/checkpoints/pi05_ebench_task26/20260518_ebench_task26_base19_3view_pi05/50000

# config_name=pi05_robocasa_task300
# policy_dir=/shared_disk/users/wenyao.xue/results/openpi/checkpoints/pi05_robocasa_task300/20260521_pi05_robocasa_task300/10000/


uv run --no-sync --offline scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=${config_name} \
  --policy.dir=${policy_dir}