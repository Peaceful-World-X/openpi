
## 环境
```bash

# 激活环境
cd openpi && source .venv/bin/activate && which python
```

## 计算 norm
```bash
asset_id="robocasa_pretrain_human300_raw"
assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets"
data_dir="/shared_disk/users/can.jin/dataset/h01_robot/push_button_20260718"

uv run scripts/compute_norm_stats_fast.py \
    --data-path "${data_dir}" \
    --output-path "${assets_dir}/${asset_id}" \
    --action-horizon 50 \
    --action-dim 16 \
    --mask-list 7 -1 7 -1 \
    --num-workers 64
    
```

## 训练
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=online                # 在线查看 loss
export TOKENIZERS_PARALLELISM=false     # 避免 tokenizer 多进程 warning 或潜在 fork 问题。
export PYTHONUNBUFFERED=1               # 让 Python 输出不被缓冲，实时显示日志。
export JAX_PLATFORMS=cuda               # 强制 JAX 使用 CUDA 后端，避免在某些环境下默认使用 CPU。 
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# ========================================================
time_str=$(date +%Y%m%d_%H%M)
config_name="pi05_droid_finetune"

# 从头训练
uv run --offline scripts/train.py \
  ${config_name} \
  --exp-name "${date_str}_${config_name}" \
  --overwrite

# 继续训练
# uv run --offline scripts/train.py \
#   ${config_name} \
#   --exp-name "${date_str}_${config_name}" \
#   --resume
```

## 推理
```bash
# 80G * 0.12 = 9.6G
# 24G * 0.40 = 9.6G
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export CUDA_VISIBLE_DEVICES=0

# ========================================================
config_name=pi05_pretrain_human300
policy_dir=/shared_disk/users/wenyao.xue/results/pi05-robocasa365_task300/75000

uv run --offline scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=${config_name} \
  --policy.dir=${policy_dir}

```

## 开环测试
```bash


```