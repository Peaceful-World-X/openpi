export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE="offline"
export WANDB_DIR=/shared_disk/users/yang.wang/wandb

export HF_LEROBOT_HOME=/mnt/pfs/users/yang.wang/data
export HF_DATASETS_CACHE=/shared_disk/users/yang.wang/cache/hf_datasets_cache

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_aloha_piper_fold_shirt_exp1001 --exp-name piper_fold_shirt_green_and_blue_n472_gpu8_bs128 --overwrite
