#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY=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
wandb login
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_LEROBOT_HOME=/root/.cache/huggingface/lerobot
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE=/shared_disk/users/wenyao.xue/private_data
export WANDB_MODE=online
export WANDB_DIR=/shared_disk/users/wenyao.xue/wandb
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
date_str=$(date +%Y%m%d)

# config_name=pi05_ebench_task26
config_name="pi05_robocasa_task300"


# Smoke_Test
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync --offline scripts/train.py \
#   ${config_name} \
#   --exp-name "${date_str}_${config_name}_Smoke" \
#   --data.repo-id /shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain/atomic/AdjustToasterOvenTemperature/20250820/lerobot \
#   --batch-size 128 \
#   --num-workers 32 \
#   --num-train-steps 100_000 \
#   --save-interval 1000 \
#   --keep-period 10_000 \
#   --fsdp-devices 4 \
#   --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync --offline scripts/train.py \
  ${config_name} \
  --exp-name "${date_str}_${config_name}" \
  --overwrite

# config_name=pi05_benchmark_open_water_0501_add_move_base
# date_str=$(date +%Y%m%d)
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --offline scripts/train.py \
#     ${config_name} \
# 	--checkpoint_base_dir /shared_disk/users/can.jin/model/openpi \
#     --exp-name ${date_str}_${config_name} \
#     --overwrite
