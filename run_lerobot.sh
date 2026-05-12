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
uv run  examples/aloha_real/convert_songling_aloha_data_to_lerobot_mp.py \
    --out-dir /shared_disk/users/can.jin/public/to_wangyang/pack_box_cut_fix \
    --raw-dirs \
    /shared_disk/datasets/private_datasets/robot_data/agilex_data/WY/20260313_box_zyc005_01 \
    /shared_disk/datasets/private_datasets/robot_data/agilex_data/WY/20260316T014_box_zyc005_01 \
    /shared_disk/datasets/private_datasets/robot_data/agilex_data/WY/20260317T005_box_zyc005_01 \
    --repo-id giga-brain/songling_aloha_fold_shirt \
    --task "pack the box" \
    --num-processes 5 \
    --mode video

# Data Merge
DATA_ROOT=/shared_disk/users/can.jin/public/to_wangyang/pack_box_cut_fix
uv run examples/aloha_real/merge_groups_dataset.py \
    --sources \
    ${DATA_ROOT}/group_1 \
    ${DATA_ROOT}/group_2 \
    ${DATA_ROOT}/group_3 \
    ${DATA_ROOT}/group_4 \
    ${DATA_ROOT}/group_5 \
    --output /shared_disk/users/can.jin/public/to_wangyang/pack_box_cut_merged \
    --state_max_dim 14 \
    --action_max_dim 14 \
    --fps 30
