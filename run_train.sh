export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_LEROBOT_HOME=/root/.cache/huggingface/lerobot
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE=/shared_disk/users/can.jin/private_data
export WANDB_MODE=offline
export WANDB_DIR=/shared_disk/users/can.jin/wandb
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

config_name=pi05_benchmark_open_water_0501_add_move_base
date_str=$(date +%Y%m%d)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --offline scripts/train.py \
    ${config_name} \
	--checkpoint_base_dir /shared_disk/users/can.jin/model/openpi \
    --exp-name ${date_str}_${config_name} \
    --overwrite
