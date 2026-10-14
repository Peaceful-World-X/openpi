# UR7 Pi05 训练执行指南
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
uv --version


```

# 训练

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

```

# 开环验证
```bash

cd /16T-2/sudu/code/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src .venv/bin/python \
scripts/eval_ur7_openloop.py \
  --config-name pi05_bund_20260917_n906_v21_new_80G_8gpu_full \
  --checkpoint-dir /16T-2/sudu/code/result/pi05_h20_8cards_20260929_145009/20000 \
  --data-path /16T-2/sudu/code/.data/bund_20260917_n906_v21_new \
  --episode-index 0,50,100,200 \
  --compare-horizon 50 \
  --output-dir /16T-2/sudu/code/.result/openloop/pi05_h20_8cards_20260929_145009

```

# 云端训练

先在 `.env` 配好数据、模型、GPU 卡型/数量、训练配置和 OSS 信息，以下两个入口二选一。

首次提交前，把数据资产和两个本地依赖项目上传到 OSS。训练容器会在运行目录中恢复 `lerobot/`、`dlimp/`，因此不会从 GitHub 在线安装这两个项目。

```txt
无 GPU：.env → submit.sh → main.py → 申请 GPU → 新容器运行 k8s_job.sh → 解压数据 → 安装环境 → 配置 OSS → 训练并上传

有 GPU：.env →                               当前容器运行 k8s_job.sh  → 解压数据 → 安装环境 → 配置 OSS → 训练并上传
```

```bash
# 入口 1：当前容器没有 GPU，提交 GPU 训练任务
cd /ossfs/code/openpi
bash ./submit.sh

# 入口 2：当前云端容器已有 GPU，直接在当前容器训练
cd /ossfs/code/openpi
bash ./k8s_job.sh

# 上传文件
bash scripts/upload_dataset_to_oss.sh --all
# 如果 .data 已上传，只同步两个项目源码
bash scripts/upload_dataset_to_oss.sh --code-deps
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


# 推理加速
```bash

## 启动 Policy Server
cd /16T-2/sudu/code/openpi
export CUDA_VISIBLE_DEVICES=0 
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
.venv/bin/python scripts/serve_policy.py --port 9000 policy:checkpoint \
  --policy.config=pi05_bund_20260917_n906_v21_new_80G_8gpu_full \
  --policy.dir=/16T-2/sudu/code/.result/pi05_h20_8cards_20260929_145009/40000

## 客户端无硬件 smoke
cd /16T-2/sudu/code/openpi
./.venv/bin/python -m deplay.ur7_client \
  --server ws://127.0.0.1:9000 \
  --timeout 180 \
  --task 'pick the randomly placed housing, lift it, reorient it upright, then stand it stably on the table' \
  --steps 16 \
  --no-rtc --reinfer 30 --blend 16 \
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
1. 第一次按 Enter：按 reset.joints_rad 的绝对关节角执行 moveJ 复位（q6 回到 0.134157449 rad）。
2. 复位完成后，按 Space：开始推理和执行动作。
3. 运行中按 Space：立即停止机械臂。
4. 停止状态按 Enter：打开夹爪。
5. 停止状态按 h：按 home 的绝对关节角归位，默认继承 reset。
6. 停止状态再次按 Space：继续推理。
7. 按 q：退出程序。

命令行会逐次显示收到的 Enter、Space、`h`、`q` 按键、按键处理结果以及状态变化；不符合当前状态的按键会显示为“已忽略”。


```

## 推理加速

```bash
cd /16T-2/sudu/code/openpi
export CFG=pi05_bund_20260917_n906_v21_new_80G_8gpu_full
export BENCH=/16T-2/sudu/code/.result/inference_accel/pi05_40000
export JAX_CKPT=/16T-2/sudu/code/.result/pi05_h20_8cards_20260929_145009/40000
export PT_CKPT="$BENCH/pytorch_checkpoint"


> 说明：下面的 JAX/PyTorch/FlashRT 服务命令是部署用法。40000 步实验报告中的六组延迟和精度数据不是通过 WebSocket 或 REST 测得的，而是由实验后端在同一进程中直接调用模型；因此报告延迟不包含网络和服务协议开销。要复现实验原始流程，使用：
>
> ```bash
> cd /16T-2/sudu/code/openpi
> .venv/bin/python test/inference_accel/run_accuracy.py \
>   --root "$BENCH" --backends jax pytorch flashrt flashrt_bf16 tensorrt tensorrt_bf16 --gpu 3
> .venv/bin/python test/inference_accel/run_sequential_latency.py \
>   --root "$BENCH"
> ```
>
> JAX 直接加载 Orbax `40000`；其余后端使用由该权重转换得到的 `pytorch_checkpoint`，TensorRT 另外读取对应的已构建引擎目录。

## 1. 官方 JAX（WebSocket，直接用于 UR7）
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
  .venv/bin/python scripts/serve_policy.py --port 9000 policy:checkpoint \
  --policy.config="$CFG" \
  --policy.dir="$JAX_CKPT"
```

## 2. 官方 PyTorch `torch.compile`（WebSocket）

```bash
CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/serve_policy.py --port 9000 policy:checkpoint \
  --policy.config="$CFG" \
  --policy.dir="$PT_CKPT"
```

这是官方 PyTorch `max-autotune` 路径；首次启动会加载并编译模型。

## 3. FlashRT FP8 / BF16（独立 HTTP 服务或离线验证）
FlashRT 自带的是 REST 接口，输入输出协议与 OpenPI UR7 WebSocket 协议不同；它不能直接替换当前 `deplay.ur7_client`。使用 FlashRT 通用服务时：


cd /16T-2/sudu/code/FlashRT
# 首次使用且环境中没有 REST 依赖时执行一次：
/16T-2/sudu/code/openpi/test/inference_accel/flashrt_env/bin/python -m pip install fastapi uvicorn
CUDA_VISIBLE_DEVICES=0 \
  /16T-2/sudu/code/openpi/test/inference_accel/flashrt_env/bin/python \
  examples/server.py \
  --checkpoint "$PT_CKPT" --framework torch --port 9001 --num_views 2


该服务的接口是 `POST http://127.0.0.1:9001/predict`。FP8/BF16 的本次 UR7 对照适配器可分别离线验证：

cd /16T-2/sudu/code/openpi
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu \
  PYTHONPATH=test/inference_accel/deps:src:. \
  test/inference_accel/flashrt_env/bin/python test/inference_accel/run_latency.py \
  --backend backend_flashrt --checkpoint "$PT_CKPT" \
  --manifest "$BENCH/manifest.json" --output "$BENCH/latency/flashrt_fp8"

OPENPI_FLASHRT_PRECISION=bf16 CUDA_VISIBLE_DEVICES=0 \
  JAX_PLATFORMS=cpu PYTHONPATH=test/inference_accel/deps:src:. \
  test/inference_accel/flashrt_env/bin/python test/inference_accel/run_latency.py \
  --backend backend_flashrt --checkpoint "$PT_CKPT" \
  --manifest "$BENCH/manifest.json" --output "$BENCH/latency/flashrt_bf16"



## 4. TensorRT FP16 / BF16（已构建引擎，离线验证）
当前 TensorRT 是 Python 驻留的三引擎实验适配版，没有接入 `scripts/serve_policy.py` 的 WebSocket 服务，也不能直接给 UR7 客户端使用。分别指定对应引擎目录进行验证：

cd /16T-2/sudu/code/openpi
OPENPI_BENCH_TRT_ENGINES="$BENCH/tensorrt/engines" \
  JAX_PLATFORMS=cpu PYTHONPATH=test/inference_accel/deps:src:. \
  LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:${LD_LIBRARY_PATH:-} \
  CUDA_VISIBLE_DEVICES=0 test/inference_accel/envs/tensorrt/bin/python \
  test/inference_accel/run_latency.py --backend backend_tensorrt \
  --checkpoint "$PT_CKPT" --manifest "$BENCH/manifest.json" \
  --output "$BENCH/latency/tensorrt_fp16"

OPENPI_BENCH_TRT_ENGINES="$BENCH/tensorrt_bf16/engines" \
  JAX_PLATFORMS=cpu PYTHONPATH=test/inference_accel/deps:src:. \
  LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:${LD_LIBRARY_PATH:-} \
  CUDA_VISIBLE_DEVICES=0 test/inference_accel/envs/tensorrt/bin/python \
  test/inference_accel/run_latency.py --backend backend_tensorrt \
  --checkpoint "$PT_CKPT" --manifest "$BENCH/manifest.json" \
  --output "$BENCH/latency/tensorrt_bf16"
```

### 服务选择

| 架构 | 精度 | 40000 权重入口 | 当前服务方式 | 能否直接接 UR7 WebSocket |
|---|---|---|---|---|
| JAX | 主要 BF16，部分 FP32 | Orbax `40000` | `scripts/serve_policy.py` | 可以 |
| PyTorch compile | BF16/FP32 | `pytorch_checkpoint` | `scripts/serve_policy.py` | 可以 |
| FlashRT | FP8 或 BF16 | `pytorch_checkpoint` | FlashRT REST 或离线适配器 | REST 不能直接替换 |
| TensorRT | FP16 或 BF16 | `tensorrt*/engines` | 离线适配器 | 暂不可以 |

如果目标是当前 UR7 真机运行，使用上面的 JAX WebSocket 命令；其余四种结果用于性能和数值对照，不能仅通过替换权重目录切换到现有 UR7 客户端。


## RTC 动作队列（2026-10-12）

机械臂服务器 `/data/sudo/openpi/deplay/client.yaml` 使用以下 RTC 配置：

```yaml
rtc:
  enabled: true
  reinfer: 30
  blend: 16
  inference_delay_steps: 4
```

- 开启：使用服务端完整动作块。队列剩余 `reinfer` 步时，根据执行后的观测异步请求下一块，继续执行已有队列；响应就绪且达到该请求的估计切换步数后，在下个动作周期切换。
- 对齐：按请求发起后实际执行的步数丢弃新预测中过去的动作，`inference_delay_steps` 仅作为初始估计。请求完成得早时继续执行旧队列直到估计切换点；请求完成得晚时按实际已执行步数切换，不额外跳到估计的未来位置。
- 融合：先删除新预测中已经过去的动作，再对剩余新动作的前 `blend` 步与旧队列同一时刻的动作作线性权重融合，旋转使用 SE(3) 插值。如果等待时已执行 4 步，跳过新块前 4 步，然后仍融合后续 16 步；旧队列剩余不足 16 步时，仅融合实际可用的重叠区间。这里实现的是客户端轨迹融合，不是模型去噪时的前缀引导。
- 关闭：`enabled: false` 或 `--no-rtc`。每次同步推理只取前 `reinfer` 步，全部执行完后重新读取观测并推理；不预取、不丢弃开头、不融合。这种模式在两块之间有推理等待是预期行为。
- 服务端必须返回足够长的动作块：RTC 开启时 `N > reinfer`，关闭时 `N >= reinfer`；不足直接报出参数错误。
- RTC 下若旧队列全部耗尽且新响应尚未到达，报错停止，不无限重复最后一个目标。
- `blend=0` 可关闭融合但保留异步预取。旧参数和旧命令行选项已移除，传入时明确报错。

例如服务端返回 50 步：首次执行 20 步后剩余 30 步，开始预取；等待时又执行 4 步，则新块从第 5 步对齐切换，从第 5 步开始融合 16 步（初始延迟估计为 4）。无需等旧块用完才切换。

为减少伺服线程的阻塞，两路相机后台采集，控制线程只取最新帧；夹爪由单独线程串行通信，合并重复硬件目标并缓存实测开度。相机/夹爪数据超过 0.5 秒未更新或通信失败时停止使用并报错。Replay 默认仍使用原有同步夹爪接口。停止时清除尚未发送的夹爪目标，但不会撤销已经发给夹爪的命令。

日志 `RTC switch: inference_ms=... consumed=... fused=... queued=...` 用于确认推理耗时、实际跳过步数和有效融合步数。`servo command interval ... exceeds target ...` 表示命令周期超时。摄像头健康、工作空间和单步位姿限制在执行前继续检查。

无硬件回归验证（不会连接机械臂）：

```bash
cd /data/sudo/openpi
PYTHONPATH=.:packages/openpi-client/src .venv/bin/python -m unittest discover -s test -p test_rtc_runtime.py -v
```


延迟估计：初始使用 `inference_delay_steps`。只统计异步预取的完整往返耗时（含排队、网络和推理），不把首次同步推理/JAX 编译计入，也不把等待切换的时间计入。积累满 10 次后，用最近 10 次耗时的滑动平均更新：`延迟步数 = ceil(平均耗时 / 控制周期)`。之后每收到一个新结果更新一次。例如 10Hz、平均 0.26s 对应 3 步，平均 0.41s 对应 5 步。日志 `RTC delay: samples=10 mean_ms=... estimated_steps=...` 显示更新结果。

估计用于安排下一次请求的切换时机，最多等到当前旧队列耗尽；真正裁剪量始终是请求后已执行的动作数。`blend=16` 在这个裁剪点之后开始计数。若实测延迟超出 `reinfer` 提供的缓冲，会记录告警；没有新响应时仍报错停止。停止/恢复时保留已学习的估计，重启进程后重新统计。关闭 RTC 时忽略延迟估计，不跳步。
