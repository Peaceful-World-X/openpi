#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export LEROBOT_IMPL=fast_convert
export LEROBOT_WORKSPACE="$ROOT_DIR/lerobot_dataset"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# Multi-Processing Data conversion
DATA_ROOT=/shared_disk/datasets/private_datasets/robot_data/agilex_data/unfold_shirt
uv run  examples/aloha_real/convert_songling_aloha_data_to_lerobot_mp.py \
    --out-dir /shared_disk/users/yang.wang/data/robotic/lerobot/songling/flatten_fold_shirt_white_c3n60_0719_unmerged \
    --raw-dirs ${DATA_ROOT}/20250717T010_unfold_shirt_zhy001_01 ${DATA_ROOT}/20250717T010_unfold_shirt_zhy001_02 ${DATA_ROOT}/20250717T010_unfold_shirt_mxh001_03 \
    --repo-id giga-brain/songling_aloha_fold_shirt \
    --task "songling_aloha_flatten_fold_shirt" \
    --num-processes 10 \
    --mode video

# Data Merge
DATA_ROOT=/shared_disk/users/yang.wang/data/robotic/lerobot/songling/flatten_fold_shirt_white_c3n60_0719_unmerged
uv run examples/aloha_real/merge_groups_dataset.py \
    --sources ${DATA_ROOT}/group_1 ${DATA_ROOT}/group_2 ${DATA_ROOT}/group_3 \
    --output /shared_disk/users/yang.wang/data_test/robotic/lerobot/flatten_fold_shirt_0719_60episodes \
    --state_max_dim 14 \
    --action_max_dim 14 \
    --fps 30
