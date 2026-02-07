# ========================================= 1. 微调SmolVLA =========================================
# 拉取镜像
docker pull docker.1ms.run/huggingface/lerobot-gpu:latest

# 初始化容器
docker run -it \
  --name xwy_lerobot \
  --init \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --shm-size=64G \
  --network host \
  -v /home/xuewenya/code:/home/xuewenya/code \
  -v /home/models:/home/models \
  -v /home/results:/home/results \
  -v /home/datasets:/home/datasets \
  -v /home/datasets_v2:/home/datasets_v2 \
  docker.1ms.run/huggingface/lerobot-gpu:latest

# 管理员权限进入命令行
docker exec -u 0 -it xwy_lerobot /bin/bash
# 普通用户进入命令行
docker exec -it xwy_lerobot /bin/bash

# 激活环境
cd /home/xuewenya/code/lerobot
source /lerobot/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com


# ------------------------------------------ 微调 ------------------------------------------
