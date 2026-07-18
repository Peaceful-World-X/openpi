#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"

# assets_dir="/shared_disk/users/wenyao.xue/results/gigabrain/assets"
# asset_id="robocasa_pretrain_human300_raw"

# RoboCasa raw LeRobot modality order:
#   state  = [base_position, base_rotation, eef_position_relative, eef_rotation_relative, gripper_qpos]
#   action = [base_motion, control_mode, eef_position, eef_rotation, gripper_close]
# The script reads each repo's meta/modality.json and keeps this start-index order.
# uv run --no-sync python -u scripts/compute_norm_stats_robocasa.py \
#   --data-root /shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain \
#   --output-path "${assets_dir}/${asset_id}" \
#   --action-horizon 50 \
#   --action-dim 32 \
#   --num-workers 16

# Smoke test for one RoboCasa repo:
# uv run --no-sync python -u scripts/compute_norm_stats_robocasa.py \
#   --data-path /shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain/composite/AddIceCubes/20250829/lerobot \
#   --output-path "${assets_dir}/${asset_id}_debug_one" \
#   --action-horizon 50 \
#   --action-dim 32 \
#   --num-workers 1 \
#   --max-files 3


# # ******************************************************************
# assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets"
# asset_id="pi05_ebench_task26"

# uv run --no-sync python -u scripts/compute_norm_stats_fast.py \
#   --data-root /shared_disk/users/wenyao.xue/EBench-Dataset \
#   --output-path "${assets_dir}/${asset_id}" \
#   --action-horizon 20 \
#   --action-dim 32 \
#   --use-base \
#   --use-delta-joint-actions \
#   --num-workers 64

assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets"

# ************************************************************************************************************************************
# h01 pi05 & gigabrian

uv run scripts/compute_norm_stats_fast.py \
    --data-path /shared_disk/users/can.jin/dataset/h01_robot/push_button_20260718 \
    --output-path "${assets_dir}/pi05_h01_push_button_beside_plate_3cam_merge" \
    --action-horizon 50 \
    --action-dim 16 \
    --mask-list 7 -1 7 -1 \
    --num-workers 64
# uv run scripts/compute_norm_stats_fast.py \
#     --data-path /shared_disk/users/can.jin/dataset/h01_robot/push_button_20260718 \
#     --action-horizon 50 \
#     --action-dim 16 \
#     --mask-list 7 -1 7 -1 \
#     --num-workers 16

# ************************************************************************************************************************************
# agilex

# pi05
# uv run scripts/compute_norm_stats_fast.py \
#     --data-path /shared_disk/benchmark/post_train_data/brush_table_agilex  \
#     --output-path /shared_disk/users/wenyao.xue/results/openpi/assets/pi05_agilex_task_005_brush_table \
#     --action-horizon 50 \
#     --action-dim 14 \
#     --mask-list 6 -1 6 -1 \
#     --adapt-to-pi \
#     --num-workers 16

# gigabrain
# uv run scripts/compute_norm_stats_fast.py \
#     --data-path /shared_disk/benchmark/post_train_data/table_setting_agilex  \
#     --output-path /shared_disk/users/wenyao.xue/results/openpi/assets/pi05_agilex_task_004_table_setting_giga \
#     --action-horizon 50 \
#     --action-dim 14 \
#     --mask-list 6 -1 6 -1 \
#     --num-workers 16

# ************************************************************************************************************************************
# # # songling piper state dim 14 action dim 16
# uv run scripts/compute_norm_stats_fast.py \
#     --data-path /shared_disk/users/can.jin/dataset/agilex/benchmark_open_water_0501/260501190118_4464  \
#     --output-path /mnt/pfs/users/can.jin/public/norm_stats/assets/benchmark_open_water_0501 \
#     --action-horizon 50 \
#     --action-dim 14 \
#     --mask-list 6 -1 6 -1 \
#     --adapt-to-pi \
#     --num-workers 16

# # # g1 arm state dim 16 action dim 18
# uv run scripts/compute_norm_stats_fast.py \
#     --data-path /shared_disk/users/yang.wang/data-g1/flat_fold_shirt/aligned_data/lerobot_merged_204_333_428_634_n300 \
#     --output-path /mnt/pfs/users/can.jin/public/norm_stats/assets/lerobot_merged_204_333_428_634_n300 \
#     --action-horizon 50 \
#     --action-dim 16 \
#     --mask-list 7 -1 7 -1 \
#     --num-workers 16


# # # g1 arm+waist(eg g1 arm+waist(1))  state dim 20 action dim 22
# uv run scripts/compute_norm_stats_fast.py \
#     --data-path /mnt/pfs/users/can.jin/public/datasets/mix_merge/fast_3_data_carry_box_b1_fast_0316_merged \
#     --output-path /mnt/pfs/users/can.jin/public/norm_stats/assets/fast_3_data_carry_box_b1_fast_0316_merged \
#     --action-horizon 50 \
#     --action-dim 20 \
#     --mask-list 7 -1 7 -1 4 \
#     --zero-mask-list -16 3 -1 \
#     --num-workers 16

# # # h01 arm dim 22 action dim 22
# uv run scripts/compute_norm_stats_fast.py \
#     --data-path /shared_disk/users/can.jin/dataset/h01_robot/h01_benchmark/build_block_0501_merged \
#     --output-path /mnt/pfs/users/can.jin/public/norm_stats/assets/build_block_0501_merged \
#     --action-horizon 50 \
#     --action-dim 16 \
#     --mask-list 7 -1 7 -1 \
#     --num-workers 16

# # h01 arm+waist(eg h01 arm+waist(3)) arm dim 22 action dim 22
# uv run scripts/compute_norm_stats_fast.py \
#     --data-path /shared_disk/users/can.jin/dataset/h01_robot/merge_fast/fast_2_carry_subbox_b1_0509_merged \
#     --output-path /mnt/pfs/users/can.jin/public/norm_stats/assets/fast_2_carry_subbox_b1_0509_merged \
#     --action-horizon 50 \
#     --action-dim 22 \
#     --mask-list 7 -1 7 -1 6 \
#     --zero-mask-list -16 1 -3 2\
#     --num-workers 16
