# UR7 Pi05 训练执行指南

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
uv --version

```

# 云端训练

先在 `.env` 配好数据、模型、GPU 卡型/数量、训练配置和 OSS 信息，以下两个入口二选一。

```bash
# 入口 1：当前容器没有 GPU，提交 GPU 训练任务
cd /ossfs/workspace/openpi
bash ./submit.sh

# 入口 2：当前云端容器已有 GPU，直接在当前容器训练
cd /ossfs/workspace/openpi
bash ./k8s_job.sh
```

```txt
无 GPU：.env → submit.sh → main.py → 申请 GPU → 新容器运行 k8s_job.sh → 解压数据 → 安装环境 → 配置 OSS → 训练并上传

有 GPU：.env →                               当前容器运行 k8s_job.sh  → 解压数据 → 安装环境 → 配置 OSS → 训练并上传
```

# 本地训练

```bash
## 阶段 0：进入项目并检查 GPU
cd /16T-2/sudu/code/openpi
source .venv/bin/activate
nvidia-smi
python --version

## 阶段 1：计算归一化统计
cd /16T-2/sudu/code/openpi
./run_norm.sh

## 阶段 2：本地 smoke test
cd /16T-2/sudu/code/openpi
export CUDA_VISIBLE_DEVICES=3
uv run --frozen --no-build-isolation --offline scripts/train.py \
  pi05_bund_demo_data_0reset_0827_v002_v21_reconstructed \
  --exp-name local_ur7_lora_smoke \
  --batch-size 1 \
  --fsdp-devices 1 \
  --num-train-steps 5 \
  --save-interval 1 \
  --log-interval 1 \
  --num-workers 0 \
  --overwrite \
  --no-wandb-enabled

## 阶段 3：Pi05 正式训练
cd /16T-2/sudu/code/openpi
export CUDA_VISIBLE_DEVICES=0,1
./run_train.sh

## 开环验证
cd /16T-2/sudu/code/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src .venv/bin/python \
scripts/eval_ur7_openloop.py \
  --checkpoint-dir /16T-2/sudu/code/result/pi05_bund_demo_data_0reset_0827_v002_v21_reconstructed_lora_local4/ur7_pi05_lora_local4/79999 \
  --episode-index 0,50,100,200 \
  --compare-horizon 50 \
  --output-dir  /16T-2/sudu/code/.result/openloop/pi05_bund_demo_data_0reset_0827_v002_v21_reconstructed_lora_local4

```

# replay

```bash
cd /data/sudo/openpi
.venv/bin/python -m deplay.replay \
  .data/bund_demo_data_0reset_0827_v002_v21_reconstructed/data/chunk-000/episode_000000.parquet \
  --initial-wait-s 3 \
  --config deplay/client.yaml \
  --execute \
  --confirm ENABLE_UR7_MOTION
  
```

# 推理
```bash

## 启动 Policy Server
cd /16T-2/sudu/code/openpi
export CUDA_VISIBLE_DEVICES=0 
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
.venv/bin/python scripts/serve_policy.py --port 9000 policy:checkpoint \
  --policy.config=pi05_bund_demo_data_0reset_0827_v002_v21_reconstructed_lora_local4 \
  --policy.dir=/16T-2/sudu/code/result/pi05_bund_demo_data_0reset_0827_v002_v21_reconstructed_lora_local4/ur7_pi05_lora_local4/30000

## 客户端无硬件 smoke
cd /16T-2/sudu/code/openpi
./.venv/bin/python -m deplay.ur7_client \
  --server ws://127.0.0.1:9000 \
  --timeout 180 \
  --task 'pick the randomly placed housing, lift it, reorient it upright, then stand it stably on the table' \
  --steps 16 \
  --action-chunk-size 16 \
  --smoke

## UR7 真机运行
cd /data/sudo/openpi
env -u HTTP_PROXY \
    -u HTTPS_PROXY \
    -u ALL_PROXY \
    -u http_proxy \
    -u https_proxy \
    -u all_proxy \
    ./.venv/bin/python -m deplay.ur7_client \
    --config deplay/client.yaml \
    --server ws://30.229.200.131:9000 \
    --execute \
    --confirm ENABLE_UR7_MOTION 

启动后在当前终端操作：
1. 第一次按 Enter：机械臂移动到复位位姿。
2. 复位完成后，按 Space：开始推理和执行动作。
3. 运行中按 Space：立即停止机械臂。
4. 停止状态按 Enter：打开夹爪。
5. 停止状态按 h：移动到 home 位姿。
6. 停止状态再次按 Space：继续推理。
7. 按 q：退出程序。

命令行会逐次显示收到的 Enter、Space、`h`、`q` 按键、按键处理结果以及状态变化；不符合当前状态的按键会显示为“已忽略”。

```
