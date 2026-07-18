#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=online
export WANDB_DIR=/shared_disk/users/wenyao.xue/logs
export WANDB_API_KEY=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
wandb login
export TOKENIZERS_PARALLELISM=false     # 避免 tokenizer 多进程 warning 或潜在 fork 问题。
export PYTHONUNBUFFERED=1               # 让 Python 输出不被缓冲，实时显示日志。
export JAX_PLATFORMS=cuda               # 强制 JAX 使用 CUDA 后端，避免在某些环境下默认使用 CPU。 
export LD_LIBRARY_PATH="/opt/conda/lib:${CONDA_PREFIX:+${CONDA_PREFIX}/lib:}${LD_LIBRARY_PATH:-}"
unset XLA_FLAGS
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

# ========================================================

time_str=$(date +%Y%m%d_%H%M)
date_str=$(date +%Y%m%d)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# config_name="pi05_h01_sort_blocks_3cam"
# config_name="pi05_h01_push_button_beside_plate_3cam_merge"
config_name="pi05_h01_task_002_push_button_5view"
# config_name="pi05_h01_task_002_push_button"
# config_name="pi05_h01_task_004_table_setting"
# config_name="pi05_h01_task_005_brush_table"

# 从头训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --offline scripts/train.py \
  ${config_name} \
  --exp-name "${date_str}_${config_name}" \
  --overwrite

# 继续训练
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --offline scripts/train.py \
#   ${config_name} \
#   --exp-name "${date_str}_${config_name}" \
#   --resume


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
