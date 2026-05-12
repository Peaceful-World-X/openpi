#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# # # songling piper/aloha state dim 14 action dim 14(16)
# uv run scripts/compute_norm_stats_fast.py \
#     --data-path /shared_disk/users/can.jin/dataset/agilex/robotwin2_0512  \
#     --output-path /shared_disk/users/can.jin/dataset/agilex/robotwin2_0512 \
#     --action-horizon 50 \
#     --action-dim 14 \
#     --mask-list 6 -1 6 -1 \
#     --adapt-to-pi \
#     --num-workers 16

# # songling piper/aloha state dim 14 action dim 14(16)
# uv run scripts/compute_norm_stats_fast.py \
#     --data-path /shared_disk/users/can.jin/dataset/agilex/benchmark_open_water_0501/260501190118_4464 \
#     --output-path /mnt/pfs/users/can.jin/public/norm_stats/assets/benchmark_open_water_0501 \
#     --action-horizon 50 \
#     --action-dim 14 \
#     --mask-list 6 -1 6 -1 \
#     --adapt-to-pi \
#     --num-workers 16

# # # g1 arm
# uv run scripts/compute_norm_stats_fast.py \
#     --data-path /shared_disk/users/yang.wang/data-g1/flat_fold_shirt/aligned_data/lerobot_merged_204_333_428_634_n300 \
#     --output-path /mnt/pfs/users/can.jin/public/norm_stats/assets/lerobot_merged_204_333_428_634_n300 \
#     --action-horizon 50 \
#     --action-dim 16 \
#     --mask-list 7 -1 7 -1 \
#     --num-workers 16


# # # g1 arm+waist(eg g1 arm+waist(1))
# uv run scripts/compute_norm_stats_fast.py \
#     --data-path /mnt/pfs/users/can.jin/public/datasets/mix_merge/fast_3_data_carry_box_b1_fast_0316_merged \
#     --output-path /mnt/pfs/users/can.jin/public/norm_stats/assets/fast_3_data_carry_box_b1_fast_0316_merged \
#     --action-horizon 50 \
#     --action-dim 20 \
#     --mask-list 7 -1 7 -1 4 \
#     --zero-mask-list -16 3 -1 \
#     --num-workers 16

# # # h01 arm
# uv run scripts/compute_norm_stats_fast.py \
#     --data-path /shared_disk/users/can.jin/dataset/h01_robot/h01_benchmark/build_block_0501_merged \
#     --output-path /mnt/pfs/users/can.jin/public/norm_stats/assets/build_block_0501_merged \
#     --action-horizon 50 \
#     --action-dim 16 \
#     --mask-list 7 -1 7 -1 \
#     --num-workers 16

# h01 arm+waist(eg h01 arm+waist(3))
uv run scripts/compute_norm_stats_fast.py \
    --data-path /shared_disk/users/can.jin/dataset/h01_robot/merge_fast/fast_2_carry_subbox_b1_0509_merged \
    --output-path /mnt/pfs/users/can.jin/public/norm_stats/assets/fast_2_carry_subbox_b1_0509_merged \
    --action-horizon 50 \
    --action-dim 22 \
    --mask-list 7 -1 7 -1 6 \
    --zero-mask-list -16 1 -3 2\
    --num-workers 16
