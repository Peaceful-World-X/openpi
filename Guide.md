# openpi-RLT 使用指南

本仓库是在官方 `openpi` 基础上复现 RL Token（RLT）的实验代码。实现分成两个阶段：
根目录负责 VLA/RL-token 的训练与服务化，`rlt_online_rl` 负责actor-critic、回放和机器人运行时。

## 模块介绍

### 1. 两阶段架构

- **Stage 1：RL-token/VLA**
  - `src/openpi/models/rl_token.py`：RL-token encoder/decoder。
  - `src/openpi/models/pi0.py`：提取 pi0/pi0.5 的 VLA prefix embedding。
  - `src/openpi/training/config.py`：注册 `debug_rlt`、Agilex 和 delta-action RLT 配置。
  - `scripts/train_rlt.py`：训练 RL-token；`rlt_alpha=0` 只训练 RLT，`rlt_alpha>0` 联合训练 VLA。
  - `scripts/eval_rlt.py`：计算重建 MSE/MAE/cosine 等离线指标。
  - `scripts/serve_rlt_policy.py`：通过 WebSocket 同时提供 `ref_chunk` 和 `z_rl`。

- **Stage 2：在线 RL/机器人运行时**
  - `rlt_online_rl/src/rlt_online_rl/networks.py`：chunk actor、双 critic 和损失。
  - `rlt_online_rl/src/rlt_online_rl/replay.py`：回放 transition、追加日志和恢复。
  - `rlt_online_rl/src/rlt_online_rl/trainer.py`：learner、目标网络、checkpoint 和 actor snapshot。
  - `rlt_online_rl/src/rlt_online_rl/inference.py`：actor 服务、Machine A 客户端和通用 `EnvDriver`。
  - `rlt_online_rl/launch/`：Machine B、评估、机器人 rollout 和 fake Machine A 启动器。
  - `rlt_online_rl/train_deploy_alignment/pika_sync_ros.py`：ROS 观测、动作发布和人工信号桥接。
  - `rlt_online_rl/scripts/offline/`：从 replay journal 做离线训练和诊断。

### 2. 数据流和边界

1. Machine A 调用 Stage-1 服务，返回 `z_rl`（默认 2048 维）和 VLA 的 `ref_chunk`。
2. Machine B 的 actor 读取 `z_rl / proprio / ref_chunk`，输出可执行 action chunk；learner 从 replay 更新 actor/critic 并发布带版本号的 snapshot。
3. rollout 只在 chunk 边界切换控制来源；预热阶段执行 VLA 参考动作，在线阶段才允许 actor 细化。
4. 当前 Agilex 在线配置是 `action_dim=7`、`chunk_len=10`、`delta_chunk`；Stage-1 模型本身使用 32 维动作和 50 步 action horizon，两个接口不能混用。

### 3. 审核结论

- **可运行性**：已通过 `compileall`、入口参数解析、RLT 配置/transform 构造、dummy shape 检查，以及 CPU `debug_rlt` 单步训练并保存 checkpoint。基础 config/network/replay 测试 19 个通过；并发 GPU 检查发生过 OOM，不能据此判断模型逻辑错误。
- **算法差异（高优先级）**：论文中的 reconstruction decoder 是自回归并使用先前 token；当前 `RLTokenDecoder` 的 `CrossAttentionLayer` 设置为 `causal=False`，decoder 也没有输入先前真实 token。因此当前 `L_ro` 是可运行的并行重建目标，但不是论文原始 decoder 的严格复现。若要求严格复现，应先修正 decoder，再做独立对照实验。
- **环境边界（必须遵守）**：根 `openpi` 要求 Python 3.11；`rlt_online_rl/pyproject.toml` 要求 Python 3.10。请使用两个环境，不能把一套环境同时用于 Stage 1 和 Stage 2。
- **snapshot 就绪**：完整在线 RL 测试为 40 通过、3 失败，失败均来自 `rlt_online_rl/tests/test_inference.py` 使用固定 `sleep(0.2/0.3)` 等待 JIT/snapshot；CPU 或首次编译较慢时 actor 仍是 `version=-1` 并回退到参考动作。生产启动器会等待 actor `/version`，但测试应改为轮询版本和明确的 ready 状态。
- **配置前置条件**：`AGILEX_LEROBOT_REPO` 默认是占位符 `your_hf_username/agilex_ethernet_lerobot`；训练前必须设置为实际 LeRobot 数据集。真实机器人命令还需要 ROS 2 Humble、相机/关节 topic、reset/reward/success hook 和安全动作过滤器。

## 使用命令

以下命令中的 `/path/to/openpi`、`/path/to/checkpoint`、`rlt_agilex_experiment`
和 `192.168.1.10` 需要替换为实际值。所有相对路径都相对于对应的工作目录。

### 1. Stage 1 环境（Python 3.11）

```bash
cd /path/to/openpi
git submodule update --init --recursive
uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Agilex 数据集必须是实际的 Hugging Face repo id
export AGILEX_LEROBOT_REPO=your_hf_username/agilex_ethernet_lerobot
# 可选；默认使用 gs://openpi-assets/checkpoints/pi05_base/params
export AGILEX_PI05_BASE_CKPT=gs://openpi-assets/checkpoints/pi05_base/params
```

先做 CPU/小规模 smoke test：

```bash
uv run scripts/train_rlt.py debug_rlt \
  --exp-name debug_rlt_smoke \
  --num-train-steps 1 \
  --save-interval 1 \
  --overwrite
```

Agilex Stage-1 训练（默认联合训练 VLA，约 5k steps）：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_rlt.py rlt_pi05_agilexbag_image_delta_joint \
  --exp-name rlt_agilex_experiment \
  --overwrite
```

仅当需要冻结 VLA、只训练 RLT 时，使用不带 `_joint` 的配置
`rlt_pi05_agilexbag_image_delta`。

### 2. Stage 1 评估与 Machine A 服务

```bash
uv run scripts/eval_rlt.py \
  --config rlt_pi05_agilexbag_image_delta_joint \
  --checkpoint-dir /path/to/checkpoint \
  --output-dir /tmp/rlt_eval \
  --num-samples 32
```

在 Machine A（Python 3.11 环境）启动冻结 VLA/RLT 服务：

```bash
cd /path/to/openpi
uv run scripts/serve_rlt_policy.py \
  --config rlt_pi05_agilexbag_image_delta_joint \
  --checkpoint-dir /path/to/checkpoint \
  --port 8000
```

可选 `--shared-prefix-inference` 只优化推理时 prefix 的复用，不改变
checkpoint、训练目标或网络 payload。

### 3. Stage 2 环境（Python 3.10）

```bash
cd /path/to/openpi/rlt_online_rl
conda create -y -n rlt_online_rl310 python=3.10 pip
conda activate rlt_online_rl310
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ../packages/openpi-client
python -m pip install -e .
# 可选：W&B sidecar
python -m pip install -e '.[monitor]'
```

机器人 rollout 的 shell 还需要：

```bash
source /opt/ros/humble/setup.bash
```

### 4. 本地无机器人 smoke test

该配置使用 `DummyChunkEnv` 和 `DummyFeatureProvider`，不连接 ROS、Machine A
或真实机器人：

```bash
cd /path/to/openpi/rlt_online_rl
conda activate rlt_online_rl310
python scripts/run_online_rl.py \
  --config configs/base/runtime.local_debug.yaml \
  --num-episodes 1
```

### 5. 双机在线训练

先在 Machine B 启动 replay、learner 和 actor：

```bash
cd /path/to/openpi/rlt_online_rl
conda activate rlt_online_rl310
python launch/launch_machine_b.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml
```

再在 Machine A 启动第 2 节的服务；随后在机器人 shell 启动 rollout：

```bash
cd /path/to/openpi/rlt_online_rl
source /opt/ros/humble/setup.bash
conda activate rlt_online_rl310
python launch/launch_robot_rollout.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --machine_a_ws_url ws://192.168.1.10:8000
```

训练时的键盘/ROS 信号客户端另开终端运行：

```bash
cd /path/to/openpi/rlt_online_rl
source /opt/ros/humble/setup.bash
conda activate rlt_online_rl310
python keyboard_toggle_teleop_record_reward_isolation.py
```

没有 Machine A 时，可用 fake 服务做协议集成检查（不代表 VLA 推理正确）：

```bash
cd /path/to/openpi/rlt_online_rl
conda activate rlt_online_rl310
python launch/fake_machine_a.py
```

### 6. 仅评估 rollout

评估不启动 learner/replay；actor 服务需要已有 snapshot：

```bash
cd /path/to/openpi/rlt_online_rl
conda activate rlt_online_rl310
python scripts/run_online_rl.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --system.role actor_service \
  --system.actor-service.snapshot-path /path/to/actor_snapshot.pkl
```

评估 rollout 使用一个终端：

```bash
cd /path/to/openpi/rlt_online_rl
source /opt/ros/humble/setup.bash
conda activate rlt_online_rl310
python launch/launch_actor_eval.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --machine_a_ws_url ws://192.168.1.10:8000
```

键盘客户端需要另开终端：

```bash
cd /path/to/openpi/rlt_online_rl
source /opt/ros/humble/setup.bash
conda activate rlt_online_rl310
python keyboard_actor_eval.py
```

`launch_actor_eval.py` 会等待 actor `/version`；`--eval_actor_only` 会强制
使用 actor 均值和确定性 rollout。训练/评估键位和 `full_task`/`critical_phase`
语义见 [`rlt_online_rl/README.md`](rlt_online_rl/README.md)。

### 7. Replay 与离线分析

```bash
cd /path/to/openpi/rlt_online_rl
conda activate rlt_online_rl310

python scripts/tools/inspect_replay_journal.py \
  runs/agilex_ethernet/replay/replay_journal.pkl

python scripts/offline/offline_train_from_replay.py \
  --replay-path runs/agilex_ethernet/replay/replay_journal.pkl \
  --steps 10000 \
  --batch-size 128 \
  --output-dir runs/agilex_ethernet/offline_train_bcq

python scripts/offline/eval_action_fit.py \
  --replay-path runs/agilex_ethernet/replay/replay_journal.pkl \
  --model-dir runs/agilex_ethernet/offline_train_bcq \
  --actor-mode mean
```

离线分析、回放导出和真实机器人播放前，先阅读
[`scripts/offline/README.md`](rlt_online_rl/scripts/offline/README.md) 和
[`scripts/replay_real_robot/README.md`](rlt_online_rl/scripts/replay_real_robot/README.md)。
真实机器人播放必须先用 `--dry-run`，并以保守速度验证参考动作。
