# OpenPI MEM 架构与使用指南

本文说明当前 OpenPI 中 MEM（Memory-Enhanced Manipulation）的实现边界、数据流和最小使用方式。实现参考论文 `.other/arXiv-mem` 与 `pi0.6` 的提交 `dcbe433`，但按当前 OpenPI 接口拆分为独立模块，MEM 默认关闭。

## 1. 支持范围

当前首版只支持：

- JAX 版 Pi0 和 Pi0.5（Pi05）。
- 可随机访问的 LeRobot dataset。
- 短期视频记忆、proprioceptive state 历史和作为低层条件输入的长期语言记忆。
- 推理时注入外部 VLM callback，或训练时读取离线 JSONL memory 标签。

当前不支持 PyTorch、RLDS MEM、多节点效果验证，也不包含可训练的高层语言生成头。标准 Pi0/Pi05 在 MEM 全部关闭时继续使用原模型结构、checkpoint 和推理策略。

## 2. 总体数据流

训练数据流为：

```text
LeRobot 当前样本
  -> MEMLeRobotDataset 按 episode/frame/FPS 随机访问历史
  -> robot input transform 统一相机和 state 字段
  -> Normalize / VideoFrameStack / TokenizeMemory
  -> Observation
  -> Pi0/Pi05 prefix: language memory -> video image -> prompt -> state history
```

推理数据流为：

```text
环境 observation
  -> FrameHistoryBuffer 构造 K-1 帧和历史 state
  -> HighLevelPolicy 在触发时更新 subtask / language memory
  -> active subtask 覆盖本次低层 prompt
  -> 原 Policy.infer()
  -> 推理成功后写入当前帧和 state
```

这里 `video_memory_frames=K` 表示送入视觉编码器的总帧数，包含当前帧，因此数据字段 `image_history` 和 `state_history` 的长度都是 `K-1`。历史顺序固定为最旧到最新，当前帧由原 `image` 字段提供。

## 3. 三类记忆

### 3.1 短期视频记忆

SigLIP 先执行原空间 self-attention，并每隔 `temporal_attn_every_n_layers` 层对相同 patch 位置执行因果时间 attention。时间分支复用同一层的 LayerNorm、QKV 和输出投影，不增加独立 temporal QKV 参数。固定 sinusoidal 时间编码以当前帧为零点，episode 起始 padding 帧由 `image_history_masks` 屏蔽。

到 `drop_history_tokens_after_layer` 后只保留当前帧 token，避免后续层和语言模型的序列长度随 K 成倍增加。`K=1` 强制使用原 scan 视觉路径。

### 3.2 状态历史

当前 state 继续走原 Pi0/Pi05 路径；`K-1` 个历史 state 使用独立 `state_history_proj` 投影为 prefix token。历史 state 使用当前 `state` 的 normalization stats，并同步 pad 到 `action_dim`。

### 3.3 长期语言记忆

语言 memory 是已经成功完成步骤的压缩文本。训练时 `TokenizeMemory` 将它 pad/truncate 到固定长度；模型只把这些 token 作为低层条件，不增加 language-memory loss。

推理时 `HighLevelPolicy` 依赖调用方注入的 VLM 和 tokenizer。VLM 必须输出：

```json
{"subtask": "pick up the bowl", "updated_memory": "I opened the cabinet."}
```

失败子任务不会推进 language memory。VLM 回调异常、非法 JSON、缺少字段、tokenizer 异常或缺少 base camera 时保留旧状态，低层动作推理仍继续。

## 4. 配置

`MEMConfig` 位于 `src/openpi/models/pi0_config.py`：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `use_video_memory` | `False` | 启用多帧 SigLIP |
| `video_memory_frames` | `6` | 总帧数 K，包含当前帧 |
| `video_frame_stride_sec` | `1.0` | 历史相邻帧的时间间隔 |
| `temporal_attn_every_n_layers` | `4` | 时间 attention 层间隔 |
| `drop_history_tokens_after_layer` | `-4` | 丢弃历史 token 的视觉层 |
| `use_state_history` | `False` | 启用历史 state token |
| `use_language_memory` | `False` | 启用语言 memory token |
| `max_memory_tokens` | `256` | 固定 memory token 长度 |

`MEMDataConfig` 位于 `src/openpi/training/config.py`：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `image_key_map` | `{}` | canonical 相机名与原始字段的显式映射 |
| `state_key` | `observation.state` | LeRobot 原始 state 字段 |
| `memory_label_path` | `None` | 离线 memory JSONL 路径 |

在 `_CONFIGS` 中新增训练配置时可使用：

```python
TrainConfig(
    name="pi05_my_robot_mem",
    model=pi0_config.Pi0Config(
        pi05=True,
        mem=pi0_config.MEMConfig(
            use_video_memory=True,
            use_state_history=True,
            use_language_memory=True,
        ),
    ),
    data=LeRobotAlohaDataConfig(
        repo_id="your_org/your_dataset",
        base_config=DataConfig(
            memory=MEMDataConfig(
                image_key_map={"base_0_rgb": "observation.images.top"},
                state_key="observation.state",
                memory_label_path="/path/to/memory_labels.jsonl",
            )
        ),
    ),
)
```

## 5. LeRobot 数据约束

dataset 必须支持 `__len__` 和按整数索引随机访问。每个样本需要提供 `episode_index`、episode 内的 `frame_index`，dataset metadata 需要给出 FPS。采样步长按 `round(video_frame_stride_sec * fps)` 转为整数且至少为 1。

episode 起始处不足的历史索引会固定到首帧，并生成 `False` mask，绝不会跨 episode 取帧。默认会探测 canonical、ALOHA、DROID 和 Libero 在 repack 前后的常见相机字段；DROID 的 `joint_position` 和 `gripper_position` 会按当前 `DroidInputs` 顺序组合为历史 state。非标准数据应显式设置 `image_key_map` 和 `state_key`。

## 6. 最小训练命令

先把上述配置加入 `src/openpi/training/config.py` 的 `_CONFIGS`，然后执行：

```bash
cd /mnt/pfs/users/wenyao.xue/code/Openpi/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_my_robot_mem
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_my_robot_mem --exp-name=mem_finetune --overwrite
```

从标准单帧 checkpoint 初始化视频 MEM 时，`mem_checkpoint.py` 会把 scan 的 `Transformer/encoderblock` 按层拆为非 scan 参数；新 `state_history_proj` 保留模型初始化值。已经是 MEM 结构的 checkpoint 使用严格匹配，其他缺失或形状错误会直接报错。

## 7. 最小推理示例

核心代码不依赖任何 VLM 供应商 SDK。调用方注入的 callback 签名固定为 `vlm_inference_fn(image, prompt, max_tokens) -> str`：

```python
from openpi.models.high_level_policy import HighLevelPolicy, HighLevelPolicyConfig
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies import policy_config


def vlm_inference_fn(image, prompt, max_tokens):
    return external_vlm.generate(image=image, prompt=prompt, max_tokens=max_tokens)


max_memory_tokens = train_config.model.mem.max_memory_tokens
high_level = HighLevelPolicy(
    vlm_inference_fn,
    PaligemmaTokenizer(max_memory_tokens),
    HighLevelPolicyConfig(max_memory_tokens=max_memory_tokens),
)
policy = policy_config.create_trained_policy(
    train_config,
    "/path/to/checkpoint",
    high_level_policy=high_level,
)
policy.reset_episode(task_goal="put the dishes in the sink")
result = policy.infer(observation)
policy.notify_subtask_result(success=True, observation=observation)
```

`notify_subtask_result()` 只记录完成信号；真正的高层更新在下一次 `infer()` 之前发生。没有主动信号时，高层策略按 `subtask_trigger_steps` 定期触发。

## 8. 离线 memory 标注

输入 `subtask_annotations.jsonl` 每行描述一个 episode：

```json
{"episode_id":"ep0001","subtasks":[{"instruction":"open cabinet","success":true,"frame_index":20},{"instruction":"pick wrong bowl","success":false,"frame_index":35}]}
```

输出 `memory_labels.jsonl` 每行对应一个训练时刻：

```json
{"episode_id":"ep0001","timestep":20,"subtask_instruction":"open cabinet","subtask_success":true,"memory_before":"","memory_after":"I opened the cabinet."}
```

`timestep` 必须对应 LeRobot 的 `frame_index`；数据加载器也兼容输出中的 `frame_index` 别名，并优先读取 `memory_before`，避免把当前步骤结果泄漏到输入。

Anthropic SDK 只由脚本延迟导入，不属于 OpenPI 核心依赖：

```bash
ANTHROPIC_API_KEY=... uv run scripts/gen_memory_labels.py \
  --input /path/to/subtask_annotations.jsonl \
  --output /path/to/memory_labels.jsonl \
  --max_episodes 100 \
  --max_memory_len 512
```
