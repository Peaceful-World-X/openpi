# Pi0.5

## 1. 本地-H100、RTX4060、RTX4080

### 初始化
```bash
# 拉取代码
mkdir -p ~/code && cd ~/code
git clone -b sduty https://gitee.com/Peaceful-World-X/openpi.git
GIT_LFS_SKIP_SMUDGE=1 git clone https://gitee.com/Peaceful-World-X/lerobot.git
# 拉取解压镜像（密码cyto）
rsync -avz --progress cyto@172.16.10.40:/home/cyto/docker/images/openpi_v3.tar ~/openpi_v3.tar
sudo apt update && sudo apt install -y pv && pv -p -t -e -r -b openpi_v3.tar | docker load
# 传模型文件
sudo rsync -avzP --mkpath --progress cyto@172.16.10.40:/home/cyto/results/ /home/results/
# 把 swap 调整为 32G
sudo swapoff -a && sudo rm -f /swapfile && \
sudo fallocate -l 32G /swapfile && sudo chmod 600 /swapfile && \
sudo mkswap /swapfile && sudo swapon /swapfile && \
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
# 初始化容器
docker run -it \
  --name openpi \
  --init \
  --gpus all \
  --shm-size=16G \
  --network host \
  -v /home/cyto/code:/home/cyto/code \
  -v /home/models:/home/models \
  -v /home/results:/home/results \
  -v $HOME/.cache/uv:/root/.cache/uv \
  openpi:v3.0
# 同步环境
apt-get update && apt-get install -y cmake
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### 推理
```bash
# 进入环境
docker start openpi && docker exec -u 0 -it openpi bash -c "cd /home/cyto/code/openpi && exec /bin/bash"

# 推理 分配 8.5G 左右
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
# 第一版
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.dir=/home/results/openpi/checkpoints/pi05_cytoderm11_joint_arm_move/my_experiment_cytoderm11_joint_007/40000

# 第二版：自动复位、但是反应迟钝
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.dir=/home/results/openpi/checkpoints/pi05_cytoderm14_joint_arm_move/my_experiment_cytoderm13_joint_007/40000/

# ---------------------------------------------------------------------------------------------------------------------
# 使用 8010 端口
uv run scripts/serve_policy.py --port=8010 policy:checkpoint \
  --policy.dir=/home/results/openpi/checkpoints/pi05_cytoderm11_joint_arm_move/my_experiment_cytoderm11_joint_007/40000

# 原始命令
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_cytoderm11_joint_arm_move \
  --policy.dir=/home/results/openpi/checkpoints/pi05_cytoderm11_joint_arm_move/my_experiment_cytoderm11_joint_007/40000

# RTX 4060（8G 显存）
export XLA_PYTHON_CLIENT_MEM_FRACTION=1
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

---
## 2. 云端-A800

### 初始化
```bash
# 初始化容器（卷必须用 :z 才能在有 SELinux 的宿主机上写入；:rw 无效）
docker run -it \
  --name openpi \
  --init \
  --gpus all \
  --shm-size=64G \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p 8000-8010:8000-8010 \
  -v /home/xuewenyao/code:/home/cyto/code:z \
  -v /home/models:/home/models:z \
  -v /home/results:/home/results:z \
  -v $HOME/.cache/uv:/root/.cache/uv:z \
  openpi:v3.0
```

### 推理
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

