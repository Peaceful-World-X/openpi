#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}

# # 80G * 0.12 = 9.6G
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
# # 24G * 0.40 = 9.6G
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | head -n 1 | tr -d ' ')
if [ "${GPU_MEM_MIB}" -ge 70000 ]; then
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
elif [ "${GPU_MEM_MIB}" -ge 20000 ]; then
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
else
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
fi
# ==================================================================================================================

# config_name=pi05_ebench_all
# policy_dir=/shared_disk/users/wenyao.xue/results/pi05-ebench-generalist/200000

config_name=pi05_pretrain_human300
policy_dir=/shared_disk/users/wenyao.xue/results/pi05-robocasa365_task300/75000

# config_name=pi05_ebench_task26
# policy_dir=/shared_disk/users/wenyao.xue/results/openpi/checkpoints/pi05_ebench_task26/20260518_ebench_task26_base19_3view_pi05/99999

# config_name=pi05_robocasa_task300
# policy_dir=/shared_disk/users/wenyao.xue/results/openpi/checkpoints/pi05_robocasa_task300/20260521_pi05_robocasa_task300/99999/

# config_name=pi05_benchmark_train_post_task_merged
# policy_dir=/shared_disk/users/wenyao.xue/results/openpi/checkpoints/pi05_benchmark_train_post_task_merged/20260526_pi05_benchmark_train_post_task_merged/39999

# config_name=pi05_benchmark_language_task_merged
# policy_dir=/shared_disk/users/wenyao.xue/results/openpi/checkpoints/pi05_benchmark_language_task_merged/20260527_pi05_benchmark_language_task_merged/39999

# config_name=pi05_h01_benchmark_language_task_merged
# policy_dir=/shared_disk/users/wenyao.xue/results/openpi/checkpoints/pi05_h01_benchmark_language_task_merged/20260528_pi05_h01_benchmark_language_task_merged/39999

# config_name=pi05_h01_benchmark_train_post_task_merged
# policy_dir=/shared_disk/users/wenyao.xue/results/openpi/checkpoints/pi05_h01_benchmark_train_post_task_merged/20260528_pi05_h01_benchmark_train_post_task_merged/39999

export CUDA_VISIBLE_DEVICES=0
echo "[config_name]=${config_name}"
echo "[policy_dir]=${policy_dir}"
echo
uv run --no-sync --offline scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=${config_name} \
  --policy.dir=${policy_dir}

# uv run --no-sync --offline scripts/serve_policy.py \
#   --port 8000 \
#   policy:checkpoint \
#   --policy.config=${config_name} \
#   --policy.dir=${policy_dir}