# Training-Time RTC 指南

本文对应 `Training-Time Action Conditioning for Efficient Real-Time Chunking`，实现支持 JAX/PyTorch 的 Pi0/Pi0.5；参考实现以 `xense-openpi` 为准。

## 核心方法

1. 每个动作 token 可接收逐 token 流时间; 前缀为 `t=0`, 后缀共享当前样本的 Beta 时间。
2. 前 `delay` 个动作使用干净真值，只有后缀加噪。
3. loss 只计算后缀，并按有效后缀数量归一化。
论文从干净端 `tau=1` 表示前缀；openpi 的流方向相反，`t=0` 才是干净端，因此代码中的前缀时间必须设为 `0.0`。
配置中 `max_delay` 是排他上界: `0 <= delay < max_delay <= action_horizon`。
直接构造模型配置时使用 `Pi0RTCConfig(max_delay=N)`；旧的 `Pi0Config(enable_training_time_rtc=True, ...)` 不再支持。

| 对比项 | 当前 RTC | xense-openpi | FASTER |
|---|---|---|---|
| 代码组织 | `Pi0RTC`/`RTCPolicy` 子类 | RTC 分支直接嵌入标准 Pi0 | 完整复制为 `Pi0Faster` |
| JAX/PyTorch | 两者都支持 | 两者都支持 | RTC/FASTER 前缀模式仅 JAX |
| RTC flow time | `Beta(1.5,1)` | Uniform | `Beta(1.5,1)` |
| delay | Uniform `[0,max_delay)` | Uniform `[0,max_delay)` | Uniform `[0,max_delay)` |
| 前缀 | 干净动作，OpenPI `t=0` | 相同 | 相同 |
| loss | 严格后缀元素平均 | 受动作维度和前缀长度影响 | 严格后缀 token 平均 |
| 前缀冻结 | 每步回填，返回前再次回填 | 每步更新后回填 | 每步更新前回填，返回前未回填 |
| 请求校验 | 严格 `ValueError` 和范围校验 | 缺失值会自动设为零 | 主要使用 `assert` |
| 异步队列 | 没有 | 完整 broker、线程、队列、延迟估计 | 异步及 streaming 工作流 |
| 特有算法 | 无 | 无 | HAS、混合 schedule、逐步输出动作 |

## 训练

```bash
# 进入 openpi 仓库
cd /mnt/pfs/users/wenyao.xue/code/Openpi/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 实验名，如 20260828_1630_pi05_libero_trtc。
time_str=$(date +%Y%m%d_%H%M)
config_name=pi05_libero_trtc
exp_name="${time_str}_${config_name}"

# 计算norm，写入 assets/pi05_libero_trtc/physical-intelligence/libero/norm_stats.json。
uv run scripts/compute_norm_stats.py --config-name "$config_name"

# 从 pi05_base 参数开始微调, 优化器步数从 0 开始; 这不是随机初始化训练。
# --offline 禁止 uv 联网解析依赖, --overwrite 仅覆盖同名实验目录。
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run --offline scripts/train.py \
  "$config_name" \
  --exp-name "$exp_name" \
  --overwrite
```

## 推理

```bash
cd /mnt/pfs/users/wenyao.xue/code/Openpi/openpi

config_name=pi05_libero_trtc
checkpoint=".../29999"

# 使用同一训练配置加载模型和归一化统计，在 8000 端口启动 WebSocket 服务。
CUDA_VISIBLE_DEVICES=0 \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config="$config_name" \
  --policy.dir="$checkpoint" \
  --port=8000
```

RTC 请求必须同时携带 `action_prefix` 和 `delay`; 首次请求也应传空/零前缀及 `delay=0`, 以使用同一 RTC 采样路径。
官方 LIBERO 客户端和 `ActionChunkBroker` 不会自动维护前缀、估算延迟或异步合并动作; 只启动上述服务不代表已建立 RTC 真机闭环。