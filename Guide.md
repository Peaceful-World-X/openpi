
## Pi0.5

### A800 云端推理
```bash
# 进入环境
docker exec -u 0 -it openpi_ymy_new /bin/bash
cd /home/yaomingyuan/Program/openpi_main

# 计算归一化统计量（均值/方差）并保存，供训练与推理使用（仅需运行一次）
uv run --group rlds scripts/compute_norm_stats.py \
  --config-name=pi05_cytoderm11_joint_arm_move \
  --max-frames=500000

# 训练（全部 GPU；若需单卡请先执行 export CUDA_VISIBLE_DEVICES=0）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_cytoderm11_joint_arm_move \
  --exp-name=my_experiment_cytoderm11_joint_007 \
  --overwrite

# 推理（降低 XLA 显存占用，为推理留出显存）
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_cytoderm11_joint_arm_move \
  --policy.dir=/home/pub_envs/openpi/checkpoints/pi05_cytoderm11_joint_arm_move/my_experiment_cytoderm11_joint_007/40000
```