# 机器 A 服务器批量推理指南

## 概览

服务器（`serve_rlt_policy.py`）同时支持单条和批量推理。对于稠密回放（`step_trace_stride > 0`），机器 B 使用固定大小的微批次，以便让 JAX 始终处理预热过的形状，同时减少 WebSocket 往返次数。普通分块回放（`step_trace_stride = 0`）仍按需进行单条获取，避免为较小的尾部请求填充数据。

## 机器 B（RL 客户端侧）

### 无需修改运行时代码

`inference.py` 已更新。`_build_replay_transitions` 仅在稠密回放时自动使用批量路径。可调参数为：

```yaml
env_driver:
  replay_feature_batch_size: 16
```

只需确认以下两点：

### 1. 确认服务器版本

确保服务器使用最新的 `serve_rlt_policy.py` 启动。连接后，元数据应包含：

```python
{'has_rl_token': True, 'supports_batch': True, ...}
```

### 2. 行为变化

**之前（顺序执行）：**

```
episode 结束 → _build_replay_transitions
  → 每个窗口逐个向机器 A 请求（N 次往返）
  → 总耗时：N × 70ms
```

**之后（自动进行稠密回放批处理）：**

```
episode 结束 → _build_replay_transitions
  → 如果 step_trace_stride > 0，内部调用 _prefetch_features_batch
  → 收集所有未缓存的观测
  → 以固定微批次发送给机器 A，默认大小为 16
  → 总耗时：ceil(N / 16) 次批量请求
```

当 `step_trace_stride = 0` 时，回放收尾会跳过批量预取，仅使用单条 `get_features()` 调用按需获取缺失的锚点。

### 3. 不受影响的部分

- ✅ 实时 rollout（每个分块一次 `get_features` 单条请求）：不变
- ✅ actor_service：不变
- ✅ learner_service：不变
- ✅ 回放格式：不变
- ✅ 键盘控制：不变
- ✅ 如果服务器不支持批量，会自动回退到顺序执行

### 4. 新增日志输出

当 `step_trace_stride > 0` 时，每个 episode 结束后会看到额外日志：

```
Prefetching 37 features via batch request (cached=8 micro_batch_size=16)
Batch prefetch chunk 1 size=16 done in 1430.2ms
Batch prefetch chunk 2 size=16 done in 1421.7ms
Batch prefetch chunk 3 size=5 done in 734.5ms
Batch prefetch done: 37 features in 3586.4ms (96.9ms/sample, requests=3)
```

## 协议规范

### 单条请求（不变）

```
客户端发送：
{
    "images": {
        "base_0_rgb": ndarray (224, 224, 3) uint8,
        "left_wrist_0_rgb": ndarray (224, 224, 3) uint8,
        "right_wrist_0_rgb": ndarray (224, 224, 3) uint8
    },
    "state": ndarray (7,) float32,
    "prompt": "task description"
}

服务器返回：
{
    "z_rl": ndarray (2048,) float32,
    "proprio": ndarray (7,) float32,
    "ref_chunk": ndarray (50, 7) float32,
    "policy_timing": {"infer_ms": float},
    "_raw_actions": ndarray (50, 7) float32,
    "_raw_rl_token": ndarray (1, 2048) float32
}
```

### 批量请求（新增）

```
客户端发送：
{
    "batch": [
        {"images": {...}, "state": (7,), "prompt": "..."},
        {"images": {...}, "state": (7,), "prompt": "..."},
        ...
    ]
}

服务器返回：
{
    "batch_results": [
        {"z_rl": (2048,), "proprio": (7,), "ref_chunk": (50,7), ...},
        {"z_rl": (2048,), "proprio": (7,), "ref_chunk": (50,7), ...},
        ...
    ],
    "batch_size": int,           # 实际观测数量
    "padded_size": int,          # 填充到预编译大小后的数量
    "total_infer_ms": float,     # 推理总耗时
    "per_sample_infer_ms": float # 摊销到每个样本的耗时
}
```

## 服务器侧细节

### 启动服务器

```bash
cd /path/to/mt-fvla
python scripts/serve_rlt_policy.py \
    --config rlt_pi05_agilexbag_image_delta_joint \
    --checkpoint-dir checkpoints/.../params \
    --port 8000
```

服务器启动时会：

1. 加载模型和 checkpoint
2. 为预编译的批量大小（1、2、4、6、8、10、12、16）预热 JIT 编译
3. 启动 WebSocket 服务器

### 批量大小填充

为避免未见过的批量大小触发 JIT 重新编译，服务器会将请求填充到最近的预编译大小：

| 请求大小 | 填充至 | 已进行 JIT 编译 |
|-----------|--------|----------------|
| 1         | 1      | ✅（预热） |
| 2         | 2      | ✅（预热） |
| 3         | 4      | ✅（预热） |
| 4         | 4      | ✅（预热） |
| 5-6       | 6      | ✅（预热） |
| 7-8       | 8      | ✅（预热） |
| 9-10      | 10     | ✅（预热） |
| 11-12     | 12     | ✅（预热） |
| 13-16     | 16     | ✅（预热） |
| 17+       | 原大小 | ⚠️ 首次调用会触发 JIT（约 30 秒） |

填充对客户端透明，服务器只返回请求数量对应的结果（多出来的填充结果会被丢弃）。

在稠密回放收尾阶段，机器 B 会把缺失锚点拆成固定大小的微批次，默认大小为 16，从而避免发送 `17+` 的请求。

### 预编译批量大小

默认值：`[1, 2, 4, 6, 8, 10, 12, 16]`

`serve_rlt_policy.py` 使用 `RLTPolicy.COMPILED_BATCH_SIZES` 同时控制填充和启动预热，因此该列表只有一个事实来源：

```python
COMPILED_BATCH_SIZES = [1, 2, 4, 6, 8, 10, 12, 16]
```

更大的批量大小需要更多 GPU 显存，也不一定会降低总延迟。在本地 RTX 4090 上对 `rlt_pi05_agilexbag_image_delta_joint` 的基准测试中，缓存的前向耗时约为：

- batch_size=1：113ms
- batch_size=8：733ms
- batch_size=16：1431ms
- batch_size=32：3120ms

本地实测批量大小上限为 52，53 会因 OOM 失败。对于在线回放收尾，建议优先使用 `16`；如果机器 A 的 GPU 压力较大，则降至 `8`。

### WebSocket 配置

禁用 ping 超时，以避免 JIT 编译期间断开连接：

```python
# websocket_policy_server.py
ping_interval=None,
ping_timeout=None,
```

## 性能

### 基准测试（本地 RTX 4090）

| 模式 | 样本数 N | 总耗时 | 单样本耗时 |
|------|----------|--------|------------|
| 顺序执行 | 16 | 约 1806ms | 约 113ms |
| 批量（N=8） | 8 | 约 733ms | 约 92ms |
| 批量（N=16） | 16 | 约 1431ms | 约 89ms |
| 批量（N=32） | 32 | 约 3120ms | 约 97ms |

### 回放构建场景

典型稠密 episode：100 个 step、stride=2、chunk_len=10，共约 45 个窗口；利用缓存复用后，大约有 25 个未缓存的观测锚点。

| 模式 | 耗时 |
|------|------|
| 顺序执行 | 25 × 113ms ≈ 2825ms，另加 RPC 开销 |
| 固定微批次=16 | 16 + 9 个锚点 ≈ 1431ms + 733-1431ms |
| **预期加速** | **适中，通常约 15-30%，取决于网络开销** |

稠密回放的主要收益是稳定性：回放收尾不再把 23、37 或 41 等随 episode 变化的批量大小发送给 JAX，因此避免了每个 episode 形状首次出现时反复暂停进行 JIT。分块回放（`step_trace_stride=0`）通常只有少量缺失锚点，因此继续按需单条获取，避免填充开销。

## 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `scripts/serve_rlt_policy.py` | 新增 `_infer_batch`、批量填充和 JIT 预热 |
| `src/openpi/serving/websocket_policy_server.py` | 禁用 ping 超时 |
| `rlt_online_rl/src/rlt_online_rl/inference.py` | 新增 `get_features_batch` 和微批量 `_prefetch_features_batch` |
| `rlt_online_rl/src/rlt_online_rl/config.py` | 新增 `env_driver.replay_feature_batch_size` |

## 测试

```bash
# 真实服务器测试
python scripts/test_batch_padding.py --host localhost --port 8000

# 批量大小扫描
python scripts/test_rlt_batch.py --host localhost --port 8000 --batch-size 16
```
