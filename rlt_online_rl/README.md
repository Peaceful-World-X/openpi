# RLT 在线 RL 运行时

本文档介绍 `rlt_online_rl` 运行时：这是在 openpi/RLT 模型训练并完成服务化之后使用的轻量级在线 RL 系统。根目录的 [README](../README.md) 介绍项目概览、演示视频、RLT 与 openpi 的关系、贡献者和引用模板。

`rlt_online_rl` 实现面向机器人的在线学习闭环：

- 机器 A 特征/参考服务：提供 `z_rl` 和 VLA 参考动作分块
- 机器 B actor 服务、learner 服务和回放管理器
- 机器人 rollout 驱动器：连接 ROS 观测、机器 A、actor、回放、人工信号、重置和评估

当前公开的任务配置为 `configs/tasks/agilex_ethernet`，用于展示真实机器人上的以太网插入任务。

## 范围

本软件包负责在线 RL 运行时的以下部分：

- B1 `actor_service`
- B2 `learner_service`
- B3 `replay_manager`
- B4 `EnvDriver`
- ROS rollout 适配器
- 回放日志、原始 episode 持久化、actor 快照、日志和指标
- 预热、在线 rollout、人工接管、关键阶段交接以及仅评估执行

它不负责训练基础 VLA 或 RL-token 模块。这些内容位于根目录的 openpi 栈中，主要涉及 `src/openpi`、`scripts/train_rlt.py` 和 `scripts/serve_rlt_policy.py`。

## 运行时架构

机器 A 运行冻结的 openpi/RLT 策略服务器。对于每个观测，它返回：

- `z_rl`：紧凑的 RL-token 特征
- `ref_chunk`：VLA 参考动作分块

机器 B 运行：

- `actor_service`：提供当前轻量级 actor，用于低延迟细化
- `learner_service`：采样回放、训练 actor/critic 并发布 actor 快照
- `replay_manager`：负责回放缓冲区和追加写入日志

机器人 rollout 流程：

- 读取 ROS 观测
- 查询机器 A
- 执行 VLA 参考分块或 actor 细化后的分块
- 记录原始 step trace
- 在 episode 结束时构建回放 transition
- 将 transition 发送给回放管理器

## 分块执行路径

在每个分块边界执行以下步骤：

1. rollout 适配器读取当前机器人观测。
2. 将观测发送给机器 A。
3. 机器 A 返回 `z_rl` 和 `ref_chunk`。
4. rollout 根据本地观测状态推导 `proprio`。
5. 在预热阶段或完整任务的非关键前缀阶段，机器人直接执行 `ref_chunk`。
6. 在线关键阶段控制期间，机器 B 的 actor 接收 `z_rl / proprio / ref_chunk`，并返回细化后的分块。
7. 机器人执行选定的分块，持续 `chunk_exec_horizon` 个控制 tick。
8. episode 先保存实际执行的原始 step。
9. episode 结束时构建回放窗口，并回填机器 A 缺失的锚点。
10. learner 采样回放并发布 actor 快照。
11. actor 服务热加载最新快照。

## 核心模式

### 预热

预热阶段使用冻结的 VLA 参考策略收集回放，actor 尚未被允许控制机器人。当回放达到 `warmup_min_size` 后，learner 才会启动。

如果设置了 `warmup_post_collect_updates`，learner 会在允许在线 rollout 之前执行指定数量的预热更新；否则，所需的预热更新预算由 `warmup_ready_adds_total * grad_updates_per_cycle` 推导得到。

### 预热等待在线

收集到足够的预热数据后，rollout 会同时等待以下条件：

- learner 状态 `ready_for_online == true`
- actor 版本达到或超过 rollout 阈值

只有在 episode 之间才会切换到在线控制，绝不会在 episode 中途切换。

### 在线

在线 episode 可以在关键阶段使用 actor。训练期间，actor 按照 `runtime.env_driver.actor_deterministic` 的设置采用随机或确定性行为。仅评估 rollout 会强制使用确定性的 actor 均值。

### 关键阶段与完整任务

`critical_phase` 会直接从精度要求高的关键片段开始 episode。`full_task` 先使用基础策略执行关键片段之前的部分，收到人工关键阶段信号后再切换到关键阶段控制。完整任务中非关键阶段的前缀不会写入回放。

## 当前以太网默认配置

当前以太网任务配置使用：

- `action_dim: 7`
- `chunk_len: 10`
- `z_dim: 2048`
- `proprio_dim: 7`
- `action_representation: delta_chunk`
- `reference_dropout_prob: 0.5`
- `warmup_min_size: 600`
- `warmup_post_collect_updates: 20000`
- `grad_updates_per_cycle: 5`
- `step_trace_stride: 0`
- `control_frequency_hz: 20.0`

注意，`step_trace_stride: 0` 会禁用稠密 stride 回放，保留分块边界回放。这是当前以太网配置的有意设置。

## 回放语义

每个回放 transition 包含：

- `z_rl`、`proprio`
- `ref_chunk`：该 transition 观测对应的机器 A/VLA 参考动作
- `action_chunk`：机器人实际执行的动作
- `rewards`、`done`
- `next_z_rl`、`next_proprio`、`next_ref_chunk`
- `source`：分块级别的控制来源
- `source_chunk`：逐 step 的控制来源
- `collection_phase`：预热或在线
- `episode_id`、`step_id`、`success`、`intervention_flag`

`TransitionSource` 仅是控制来源标签：

- `BASE`：执行冻结的 VLA 参考动作
- `RL`：执行 actor 细化后的动作
- `HUMAN`：人工控制执行
- `MIXED`：窗口同时包含人工和策略 step

训练器使用 `source_chunk` 逐 step 选择 BC 目标：

- `HUMAN / MIXED` step 对齐到实际执行的 `action_chunk`
- `BASE / RL` step 对齐到 VLA 的 `ref_chunk`

这有意区别于用人工动作替换 `ref_chunk`。部署时 actor 仍会看到 VLA 参考动作，因此人工数据教会 actor 如何将 VLA 参考动作编辑为实际执行的修正动作。

## Learner 目标

learner 使用双 critic、固定标准差的高斯分块 actor、目标网络和参考动作 dropout。当前 actor 损失为：

```text
actor_loss = bc_weight * bc_penalty - q_weight * actor_q + delta_weight * delta_penalty
```

预热和在线阶段可以使用不同的 BC/Q 权重：

- `warmup_bc_weight`、`warmup_q_weight`
- `online_bc_weight`、`online_q_weight`

`delta_penalty` 会先把归一化的训练动作转换回可执行的绝对动作分块，然后计算 step 间的增量；当前比较的是前六个机械臂关节的 step-to-step 增量。

## 回放窗口

回放在 episode 结束时根据原始 episode trace 构建。

`step_trace_stride: 0`：

- 使用分块边界回放窗口
- 在人工控制重新交还给策略控制时添加策略重启锚点
- 可能添加一个与终止状态对齐的最终窗口
- 只回填这些窗口所需的锚点

`step_trace_stride > 0`：

- 按配置的 stride 从原始 step trace 构建稠密回放窗口
- 使用批量机器 A 特征回填缺失锚点
- 设置为 `2` 时与 RLT 论文使用的稠密回放思路一致

回放日志是追加写入的 pickle 流。回放管理器在启动时从日志恢复，并从恢复数据中最大的 `episode_id + 1` 继续编号。

## 人工控制与手动信号

ROS 适配器支持以下手动服务：

- 请求下一个 episode
- 记录成功、失败或结束
- 进入或切换关键阶段
- 选择下一个关键阶段使用 actor 还是基础策略
- 切换 teleop 接管

人工接管期间，回放会在每个控制 tick 记录最新采样到的人工动作，不会将原始 teleop 事件流直接写入回放。

## 安装

为在线 RL 运行时使用独立的 Python 3.10 环境：

```bash
cd openpi-RLT/rlt_online_rl
conda create -y -n rlt_online_rl310 python=3.10 pip
conda activate rlt_online_rl310
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ../packages/openpi-client
python -m pip install -e .
```

机器人 rollout 和键盘客户端还需要在运行它们的 shell 中加载 ROS：

```bash
source /opt/ros/humble/setup.bash
```

可选的 W&B sidecar 支持：

```bash
python -m pip install -e '.[monitor]'
```

## 启动训练

启动机器 B 服务：

```bash
cd openpi-RLT/rlt_online_rl
conda activate rlt_online_rl310
python launch/launch_machine_b.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml
```

在 RLT checkpoint 可用后，从仓库根目录启动机器 A：

```bash
cd openpi-RLT
python scripts/serve_rlt_policy.py \
  --config rlt_pi05_agilexbag_image_delta_joint \
  --checkpoint-dir <checkpoint-dir> \
  --port 8000 \
  --shared-prefix-inference
```

`--shared-prefix-inference` 是机器 A 服务器上的仅推理延迟优化选项。机器 A 同时需要 `z_rl` 和冻结 VLA 的 `ref_chunk`；旧路径会为 `z_rl` 计算一次 VLA 前缀，然后在动作采样中再次计算，以构建 KV cache。启用该选项后，服务器只计算一次前缀，并复用同一个前缀输出/KV cache 来生成两个结果。它不会改变训练、checkpoint、模型权重、归一化或在线 RL 运行时 payload。若需要严格复现，请省略该选项以保留旧的推理路径。

没有真实 VLA 服务器时，可使用以下命令进行本地集成测试：

```bash
cd openpi-RLT/rlt_online_rl
python launch/fake_machine_a.py
```

启动机器人 rollout：

```bash
cd openpi-RLT/rlt_online_rl
source /opt/ros/humble/setup.bash
conda activate rlt_online_rl310
python launch/launch_robot_rollout.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --machine_a_ws_url ws://MACHINE_A_IP:8000
```

启动训练键盘客户端：

```bash
python keyboard_toggle_teleop_record_reward_isolation.py
```

## 启动评估

评估不会启动 learner 或回放。它只运行 actor 推理和机器人 rollout。

启动 actor 服务：

```bash
cd openpi-RLT/rlt_online_rl
conda activate rlt_online_rl310
python scripts/run_online_rl.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --system.role actor_service \
  --system.actor_service.snapshot_path <actor_snapshot.pkl>
```

启动评估 rollout：

```bash
python launch/launch_actor_eval.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --machine_a_ws_url ws://MACHINE_A_IP:8000
```

`launch_actor_eval.py` 会等待 actor 服务，然后启动 `pika_sync_ros.py --eval_actor_only`。仅评估 rollout 会强制使用确定性的 actor 均值。

启动评估键盘客户端：

```bash
python keyboard_actor_eval.py
```

## 常用工具

查看回放：

```bash
python scripts/tools/inspect_replay_journal.py \
  runs/agilex_ethernet/replay/replay_journal.pkl
```

绘制 learner 指标：

```bash
python scripts/tools/plot_learner_metrics.py \
  --run_dir runs/agilex_ethernet
```

离线训练和分析工具见 [scripts/offline/README.md](scripts/offline/README.md)。

真实机器人回放导出和播放工具见 [scripts/replay_real_robot/README.md](scripts/replay_real_robot/README.md)。

## 建议的首次运行顺序

训练：

1. 启动机器 B 服务。
2. 启动机器 A。
3. 启动机器人 rollout。
4. 确认机器人已重置到起始姿态。
5. 启动训练键盘客户端。
6. 按 `o` 开始 episode。
7. 在 `full_task` 模式下，到达关键阶段边界时按 `c`。
8. 成功时按 `s`，失败时按 `f`。

评估：

1. 启动 actor 服务。
2. 启动评估 rollout。
3. 确认机器人已重置到起始姿态。
4. 启动评估键盘客户端。
5. 按 `a` 或 `b` 为下一个关键阶段选择 actor 或基础策略。
6. 按 `o` 开始 episode。
7. 在 `full_task` 模式下，到达关键阶段边界时按 `c`。
8. episode 应结束时按 `s`。

## 常见误解

- `full_task` 的前缀在关键阶段开始前不会写入回放。
- `full_task` 的前缀不会使用 actor，而是执行机器 A 的参考动作。
- 训练中，`s` 表示成功并结束 episode。
- 评估中，`s` 会结束/重置 episode，不会作为训练奖励使用。
- 仅评估 rollout 会忽略训练 rollout 的随机性设置，使用 actor 均值。
- 评估中的 `a / b` 选择下一个 episode 的关键阶段策略，不会立即在当前 episode 中途切换。
- `critical_phase` 通常不需要按 `c`，因为它已经从关键片段内部开始。
- 预热就绪不会在当前 episode 执行到一半时切换为在线控制。

## 目录结构

```text
rlt_online_rl/
|-- configs/                    # 基础和任务运行时配置
|-- launch/                     # 机器 B、rollout、评估和虚拟机器 A 启动器
|-- scripts/offline/            # 离线回放训练和分析
|-- scripts/replay_real_robot/  # 导出和播放参考/actor 关节分块
|-- scripts/tools/              # 轻量级查看和绘图工具
|-- src/rlt_online_rl/          # 核心运行时包
|-- train_deploy_alignment/     # ROS 适配器和手动信号桥接
`-- tests/                      # 运行时单元测试
```
