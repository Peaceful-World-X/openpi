# Pi0.5

## 本地-RTX4060
### 1.1 初始化
```bash
# 初始化容器
docker run -it \
  --name openpi_ymy_new \
  --init \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0\
  --shm-size=16G \
  --network host \
  -v /home/cyto/code:/home/cyto/code \
  -v /home/cyto/models:/home/cyto/models \
  -v /home/cyto/results:/home/cyto/results \
  openpi:v3.0

# 管理员权限进入容器
docker run -it --name openpi_ymy_new openpi:v3.0 /bin/bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### 1.2 本地推理
```bash
# 进入环境
docker exec -u 0 -it openpi_ymy_new /bin/bash
cd /home/cyto/code/openpi
# 推理
# 不预占整块，按需分配，避免一开始就占满导致推理时不够
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# 允许 JAX 使用 100% 显存（在无其他进程占用的前提下）
export XLA_PYTHON_CLIENT_MEM_FRACTION=1
# 使用平台分配器，有时能提高可用显存
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_cytoderm11_joint_arm_move \
  --policy.dir=/home/cyto/results/openpi/checkpoints/pi05_cytoderm11_joint_arm_move/my_experiment_cytoderm11_joint_007/40000
  
```

## 云端-A800

### 1.2 云端推理
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

