# OpenPI RECAP 在线强化学习指南

本文介绍当前仓库中的通用 RECAP 闭环。实现扩展现有 JAX Pi0/Pi05，RECAP 默认关闭；普通训练的数据结构、loss 和推理路径不变。PyTorch 路径尚未实现 RECAP，启用后会显式报错。

## 架构

RECAP 被拆成四个边界清晰的子系统：

1. `openpi.recap.collector` 通过硬件无关协议采集自主 rollout 和人工干预。
2. `openpi.models.value_function` 与 `openpi.recap.value_trainer` 训练 distributional value function。
3. `openpi.recap.advantage` 计算 N-step advantage，`scripts/label_recap_advantage.py` 生成标签和 LeRobot sidecar。
4. `openpi.models.pi0` 消费 advantage 条件，`openpi.recap.pipeline` 按 Algorithm 1 编排多轮训练与采集。

`ReCAPConfig` 控制 `alpha`、30% conditioning dropout、默认 `guidance_scale=2.0`、40% positive fraction、201 个 value bins、`[-1, 0]` value 范围和 50 步 lookahead。启用后，`ModelTransformFactory` 自动追加 `TokenizeReCAPAdvantage`，把 `Advantage: positive` 和 `Advantage: negative` 编成固定 8 个 token；`Pi0.embed_prefix` 将选中的 token 追加到图像和语言 prefix。

策略目标对应论文 Equation 6：

```text
L = L_unconditional + alpha * L_advantage_conditioned
```

条件分支以 `advantage_dropout_prob=0.3` 的概率屏蔽 advantage token。两条分支复用同一套 flow-matching 实现；关闭 `recap.enabled` 时不会追加 token，也不会增加额外前向。

推理默认按论文 CFG 公式组合条件与无条件 flow velocity：`v = v_uncond + beta * (v_cond - v_uncond)`，其中 `beta=guidance_scale`。`create_trained_policy` 会在机器人观测未提供 RECAP 字段时默认注入 `advantage_indicator=True, use_advantage=True`，因此可直接用于在线采集；调用方可显式覆盖这两个字段。配置中的 `guidance_scale=1` 等价于直接使用 advantage-conditioned policy，并会跳过额外的无条件前向；也可通过 `Policy(sample_kwargs={"advantage_guidance_scale": 1.5})` 在部署时动态覆盖。动态值共用双分支 JIT 路径，即使传入 `1.0` 也可能保留额外前向。对 `use_advantage=False` 的样本始终返回无条件输出。

Value model 使用 OpenPI 原生组件：SigLIP 将每个相机图像编码为视觉 token，PaliGemma tokenizer/Gemma 编码语言，机器人 state 经过线性层成为额外 token；单 expert Gemma 对所有 token 做双向深度融合，masked mean pooling 后由 LayerNorm、MLP 和 201-bin head 输出 value 分布。默认骨干是 `So400m/14 + gemma_300m`，不是轻量摘要 MLP。

Value scalar target 使用相邻 bin 的 two-hot 插值，训练目标是 cross-entropy，推理通过分布期望恢复连续 value。正式默认从当前 OpenPI 组件初始化；论文使用的预训练 value 权重和 π0.6 Gemma 3 规模不在本仓库中，需通过 `train_value.py --init-checkpoint` 或 `recap_train.py --value-base-checkpoint` 显式提供兼容 checkpoint。

默认 `pi05_recap` 的 weight loader 是官方 `pi05_base`。它可直接启动本仓库的 RECAP fine-tuning，但不等同于论文先在大规模数据上训练好的 advantage-conditioned `pi_pre`；严格复现实验时应新增一个 `TrainConfig`，把 weight loader 指向兼容的 `pi_pre`，再通过 `--policy-config` 选择。Runner 每一轮都会从所选配置的固定基础权重开始，而不是继承上一轮策略，这与论文用于避免迭代漂移的做法一致。

`ValueTrainer` 使用 `model_def + params + opt_state` 的纯 JAX state，并缓存 JIT 后的 train/eval/predict 函数。训练默认裁剪全局梯度范数到 `1.0`；`freeze_mode=none` 训练全部参数，`freeze_mode=backbones` 只训练 state projection 和 value head。eval 与 value inference 按 batch 分块，避免把整个 split 或长 episode 一次放入设备内存。`--resume-checkpoint` 完整沿用 checkpoint 配置；`--init-checkpoint` 沿用模型结构，但允许用 CLI 覆盖 learning rate、gradient clip 和 freeze mode 后重新创建 optimizer。

## 论文对应关系

| 论文公式或步骤 | 代码入口 |
| --- | --- |
| 201-bin distributional V，Equation 5 | `models/value_function.py`、`recap/value_trainer.py` |
| Monte Carlo return | `recap/rewards.py::compute_episode_returns` |
| N-step advantage | `recap/advantage.py::compute_n_step_advantage` |
| 全局 improvement threshold | `recap/advantage.py::label_advantages` |
| CFG 条件/无条件速度组合 | `models/pi0.py::sample_actions` |
| 人工纠正强制 positive | `recap/advantage.py::label_advantages` |
| Advantage 文本条件 | `transforms.py::TokenizeReCAPAdvantage`、`models/pi0.py::embed_prefix` |
| Equation 6 双路目标 | `models/pi0.py::compute_recap_loss` |
| Algorithm 1 | `recap/pipeline.py::OnlineReCAPRunner`、`scripts/recap_train.py` |

论文 value 表示归一化的剩余完成时间。默认 `paper` reward 对每个任务使用最大 episode 长度 `H`：普通 transition 为 `-1/H`，成功 terminal 为 `0`，失败 terminal 为 `-1`。Value training 和 advantage labeling 共用 `rewards.py`，避免目标不一致。已有规范化环境奖励时可显式选择 `--reward-mode environment`。

N-step 目标为：

```text
A_t = sum(r_t ... r_(t+N-1)) + V(s_(t+N)) - V(s_t)
```

遇到真正 terminal 时不 bootstrap；truncated 边界只有在底层函数收到额外的 `T+1` value 时才允许 bootstrap。CLI 当前只预测已记录的 `T` 个观测，因此 timeout 默认不 bootstrap。

阈值按本项目计划和参考标注脚本，对当前批次所有非干预 rollout advantage 统一计算全局 quantile；初始只有 demonstration 时回退到全部非干预样本以保留可审计的阈值。论文正文将 `epsilon_lang` 定义为 per-task threshold，当前通用实现没有猜测任务分组，而是采用显式的全局阈值；多任务 advantage 分布差异明显时，应按 task 分批标注或扩展 threshold mapping。标签使用 `A >= epsilon`，以匹配 percentile 脚本的边界行为。论文的下游 demonstration SFT 阶段固定 `I=True`，因此 `metadata.recap_source=demo` 的帧全部保持 positive；在线人工纠正动作也强制 positive，其余 rollout 帧才由 value advantage 阈值决定。

## Episode JSON

`ReCAPOfflineEpisode` 的顶层字段如下：

```text
episode_id, task, success, timeout, max_episode_length, metadata, frames
```

每帧包含：

```text
t, observation, action, policy_action, human_action, executed_action,
reward, success, is_human_intervention, terminated, truncated
```

`observation` 使用 OpenPI canonical schema：`image`、`image_mask`、`state`、`prompt`。schema v3 在每帧增加 `observation_array_specs`，单独记录 ndarray 的 dtype 和 shape；可读的 observation 内容仍是普通 JSON list，但加载后能准确恢复 `uint8` 图像和 `float32` state。value 编码器支持 `uint8 [0,255]`、`float [0,1]`、`float [-1,1]` 和 `float [0,255]`，会统一转成 OpenPI 的 `[-1,1]` HWC 图像；缺少图像或 prompt 会直接报错，避免 critic 静默退化。旧文件只有 `obs` 和 `action` 时仍可读取，旧图像按常见整数 `0..255` 形式兼容恢复。

采集器使用临时文件、`fsync` 和 `os.replace` 原子写入；rollout 自动写入 `metadata.recap_source=rollout`，runner 将初始数据标记为 `demo`。`episode_id` 必须是安全的单个文件名，并且每次保存的 episode id 必须唯一，否则会在写盘前失败，避免历史 JSON 被同名文件覆盖。

## 接口与采集

环境实现以下接口：

```python
class ReCAPEnvironment:
    def reset(self, *, seed: int | None = None) -> dict: ...
    def step(self, action: np.ndarray) -> ReCAPStep: ...
    def close(self) -> None: ...
```

策略实现 `infer(observation)`，可以返回单步 `[D]`、动作块 `[H, D]` 或 OpenPI 的 `{"actions": [H, D]}`。人工回调返回 `InterventionDecision(action, is_intervention)`；`is_intervention=True` 时必须给出有限 human action。

采集器负责有限值检查、可选动作限位、最终执行动作平滑、超时截断和动作块失效处理。人工动作默认绕过平滑，只有显式传入 `--smooth-interventions` 才会平滑。

```python
import numpy as np
from openpi.recap import ReCAPRolloutCollector, ReCAPStep

class FakeEnv:
    def reset(self, *, seed=None):
        self.t = 0
        return self.observation()

    def observation(self):
        return {
            "image": {"base": np.zeros((16, 16, 3), np.uint8)},
            "image_mask": {"base": np.asarray(True)},
            "state": np.zeros(2, np.float32),
            "prompt": "finish the task",
        }

    def step(self, action):
        self.t += 1
        done = self.t >= 3
        return ReCAPStep(self.observation(), 0.0, done, False, done)

    def close(self):
        pass

class FakePolicy:
    def infer(self, observation):
        return np.zeros(2, np.float32)

episodes = ReCAPRolloutCollector(FakeEnv(), FakePolicy(), max_episode_length=10).collect(1)
```

真实环境、策略和人工接管通过 `module:function` 工厂注入：

```bash
uv run scripts/recap_collect.py \
  --environment my_env:make_env \
  --policy my_policy:make_policy \
  --intervention my_human:make_callback \
  --episodes 10 \
  --action-low -1 -1 -1 \
  --action-high 1 1 1 \
  --output outputs/recap/rollout
```

## 分阶段命令

```bash
# 1. 按 episode 划分 train/eval，训练 distributional value
uv run scripts/train_value.py demos \
  --output outputs/recap/iter_000/value \
  --steps 1000 \
  --learning-rate 1e-4 \
  --gradient-clip-norm 1.0 \
  --freeze-mode backbones \
  --eval-batch-size 16 \
  --init-checkpoint checkpoints/value_pre

# 2. 使用 value checkpoint 生成标签；--use-proxy 仅用于无 checkpoint 的调试
uv run scripts/label_recap_advantage.py demos \
  --value-checkpoint outputs/recap/iter_000/value \
  --inference-batch-size 16 \
  --output outputs/recap/iter_000/labels

# 3. 使用与 JSON 完全同序的 LeRobot 数据和 sidecar 训练 Pi05
uv run scripts/train.py pi05_recap \
  --exp-name recap_iter_000 \
  --data.repo-id your-user/your-aligned-dataset \
  --data.recap-fields-path outputs/recap/iter_000/labels/lerobot_fields.npz

# 4. CPU dummy smoke，不代表正式模型吞吐或显存验证；使用独立 checkpoint 目录
JAX_PLATFORM_NAME=cpu uv run scripts/train.py debug_pi05_recap \
  --checkpoint-base-dir outputs/recap/smoke \
  --num-train-steps 1
```

dummy 配置仍复用 OpenPI SigLIP，保存完整 params/train state 需要数 GiB 空间；空间不足时训练 step 可能成功而 Orbax 保存失败。不要把 smoke 目录指向已有正式 checkpoint。

## Algorithm 1

```bash
uv run scripts/recap_train.py \
  --demo-episodes demos \
  --output-dir outputs/recap \
  --num-iterations 3 \
  --collect-episodes 10 \
  --value-base-checkpoint checkpoints/value_pre \
  --policy-config pi05_recap \
  --policy-data-factory my_data:materialize_lerobot \
  --collect-factory my_rollout:collect
```

生产 policy 阶段会把最终 Orbax step 的绝对路径写入 `policy/checkpoint_path.txt`，并把该 step 路径传给 `collect-factory`；采集代码不应自行猜测 `policy/<config>/<experiment>/<step>` 层级。

`policy-data-factory` 的签名为：

```python
def materialize_lerobot(episodes_dir, labels_dir, policy_dir, base_repo_id) -> str:
    """物化聚合 JSON，返回当前 LeRobot loader 可解析的 repo id。"""
```

`collect-factory` 的签名为 `collect(policy_checkpoint, rollout_dir, count)`，其中 `policy_checkpoint` 是可直接传给 `create_trained_policy` 的具体数值 step 目录。自定义 hook 的 `train_policy` 若返回路径，Runner 同样会校验并向采集阶段透传；返回 `None` 时保留兼容行为，采集阶段收到外层 `policy/`。

这个工厂是多轮真实闭环的必要边界，因为 camera、state、action schema 和 robot-specific transforms 不能由通用 runner 猜测。它还必须生成或发布与返回 repo id 对应的 OpenPI norm stats；runner 不会从 JSON 猜测机器人归一化语义。当前 `DataConfig` 没有独立的 LeRobot `root` 参数，因此不能直接返回任意绝对目录；本地数据应放在 LeRobot cache 的 `<repo_id>` 位置并返回该 repo id，或使用自定义 DataConfig 显式处理 root。

默认 `pi05_recap` 支持以下明确的 LeRobot 列：

```text
observation.images.base_0_rgb
observation.images.left_wrist_0_rgb
observation.images.right_wrist_0_rgb
observation.state
action
task_index
```

三路图像会映射为同名 Pi0 canonical camera key，`task_index` 由 LeRobot metadata 转成 prompt。相机数量、名称或物理语义不同的机器人必须新增自己的 `TrainConfig`/input transform，并通过 `--policy-config` 选择；不能依靠排序自动映射。已经物化为顶层 `image/image_mask/state/actions/prompt` 的数据也可由同一 transform 接受。

单轮训练可以省略工厂并传 `--policy-repo-id`，但该数据集必须对应当前聚合 JSON、满足上述列契约、具有匹配 norm stats，并与 sidecar 帧身份一致。测试阶段可用 `--hook-factory module:function` 注入 `ReCAPPipelineHooks`，不会启动真实训练。

每轮固定输出：

```text
outputs/recap/iter_000/
  episodes/
  value/
  labels/recap_labels.jsonl
  labels/lerobot_fields.npz
  policy/
    checkpoint_path.txt
    <config>/<experiment>/<step>/
  rollout/
```

为避免旧 JSON 或 checkpoint 混入新实验，runner 遇到已存在的 `iter_XXX` 会拒绝运行；重跑时请使用新的 `--output-dir`。

runner history 的 `training_episodes` 表示该轮 value/label/policy 实际消费的数量，`aggregated_episodes` 表示本轮 rollout 加入后的历史总数；兼容字段 `episodes` 与后者相同。最后一轮的 `rollout/` 是下一次实验继续聚合的数据，不会被同一轮已经完成的训练反向消费。

## 与论文和参考实现的边界

| 项目 | 当前实现 |
| --- | --- |
| 在线形式 | 与论文一致，按批次 collect、离线更新、再部署；不是实时并发 RL |
| 策略骨干 | 当前 OpenPI Pi0/Pi05，不伪装成论文 Gemma 3 4B + 860M expert 的 π0.6 |
| Value 骨干 | 当前 SigLIP + `gemma_300m`，支持兼容 value checkpoint，不包含论文 670M 预训练权重和 web co-training |
| 下游 demo | 固定 `I=True`，与论文 specialist SFT 阶段一致 |
| Rollout threshold | 按本项目计划和参考脚本使用全局非干预 rollout quantile；论文原文是 per-task `epsilon_lang` |
| 迭代初始化 | value 可用 `--value-base-checkpoint` 固定从 `V_pre` 开始；policy 每轮从所选 TrainConfig 的 weight loader 开始 |
| 数据存储 | JSON episode + 只读 LeRobot sidecar，不原地修改 parquet/video |

## Sidecar 契约

`lerobot_fields.npz` 必须包含：

| 字段 | dtype | 含义 |
| --- | --- | --- |
| `advantage_indicator` | `bool` | 正/负 advantage 条件 |
| `use_advantage` | `bool` | 是否启用条件；dropout 仍会尊重原值 |
| `is_human_intervention` | `bool` | 是否为人工纠正动作 |
| `episode_index` | integer | LeRobot episode 身份 |
| `frame_index` | integer | LeRobot episode 内帧身份 |

sidecar 还保留原 JSON 的 `episode_id` 和 `t` 供审计。数据加载器会校验字段 dtype、维度、总长度，并在每次 `__getitem__` 时核对标准 LeRobot `episode_index/frame_index`，因此只满足总帧数但顺序错误的数据会立即失败。零帧 JSON episode 不占用 LeRobot episode 编号；数据物化工厂也必须跳过它。原始 parquet 和 video 不会被修改。

## 验证边界

本实现可在 CPU 上验证数学、JSON dtype round-trip、fake environment、sidecar、RECAP prefix/双路 loss/dropout、dummy VLM JIT 梯度、Orbax checkpoint 和 runner 阶段顺序。正式 `So400m/14 + gemma_300m` value 训练、Pi0/Pi05 GPU smoke、论文规模训练和真实机器人安全仍依赖可用 GPU、数据、预训练权重、机器人环境、相机语义、动作限位与人工接管系统。CPU fake 测试通过不能等同于 GPU 或硬件 readiness。
