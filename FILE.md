# openpi_ant TTRTC 改造记录

## 配置

- `Pi0RTCConfig`/`Pi0RTC` 只扩展 JAX Pi05；训练 delay 在闭区间 `[0, max_delay]` 采样，当前 UR7 配置使用 `max_delay=10`、`action_horizon=50`。
- 训练主 `loss` 仍是官方 Flow Matching MSE。TTRTC 额外记录 postfix 上的真实 10D EEF 指标，模型补齐的 22 个维度不参与这些指标。
- 标准 UR7 Pi0 配置不设置 `real_action_dim`，继续走原始 scalar loss 路径，避免增加普通训练的辅助张量和日志开销。
- 云端训练入口仍为 `cd /ossfs/code/openpi && bash ./submit.sh`。沿用原 `.env`，只将 `OPENPI_TRAIN_CONFIG` 改为 `pi05_bund_20260917_n906_v21_new_80G_8gpu_full_ttrtc`。`OPENPI_GPU_TYPE=h20`、`OPENPI_GPU_NUM=8`、数据包、归一化统计、初始参数、资源组和 OSS 设置均沿用原配置。
- 调用链为 `submit.sh → main.py → k8s_job.sh → init_cfg.sh/init_env.sh → scripts/train.py`，不经过 `run_train.sh` 或 `run_ttrtc_8gpu.sh`。GPU 容器自动准备数据、依赖、tokenizer 和权重；本地 `.venv` 或资源缺失不能用于判断该云端初始化必然失败。
- 相对父提交 `cdeeca9` 的八卡配置，保持 FSDP=8、global batch=256、50,000 steps、warmup=1,000、LR=5e-5→5e-6、workers=8、保存间隔=5,000、BF16 和 EMA=0.99。这里只增加 TTRTC 条件和真实 10D 动作日志。
- TTRTC 提交适配仅在 `main.py` 的提交前卡数校验中识别 `_ttrtc` 后缀；入口直接使用所选配置，不自动改写 `.env` 或增加训练开关。GPU 卡型、卡数、主机资源、镜像、初始化和训练计算不变。后续日志适配见下文。

## 推理协议

普通请求继续使用原来的平铺 observation。TTRTC 后续请求使用：

```python
{
    "observation": observation,
    "rtc_context": {
        "inference_delay": d,
        "prev_chunk_left_over": remaining_actions,
    },
}
```

`prev_chunk_left_over` 是客户端执行空间的绝对 UR7 10D action 序列。服务端在 `RTCPolicy` 中先注入到 `UR7Inputs`，完成绝对到相对、补齐和归一化，再传给 `Pi0RTC.sample_actions`。模型每个 Euler step 和最终输出都会重新固定前 `d` 个 prefix token。

客户端配置中 `rtc` 与 `ttrtc` 独立，二者不能同时启用。TTRTC 启动还要求服务端 metadata 明确声明 `rtc_enabled=true`：

```yaml
rtc:
  enabled: false
  reinfer: 30
  blend: 16
ttrtc:
  enabled: true
  inference_delay_steps: 3
  max_delay_steps: 10
```

TTRTC 模式默认关闭客户端 blend（可用 `--blend` 显式覆盖）；延迟使用最近 10 次实际请求耗时估计，并裁剪到服务端 metadata 宣布的训练上界。

## 日志字段

训练日志保留 `loss`，并在 TTRTC 配置下增加 `loss_real_action`、`loss_real_action_dim_00` 至 `loss_real_action_dim_09`、`ttrtc_delay_mean` 和 `ttrtc_postfix_fraction`。

## 完整训练日志保存到 OSS

- 保留原 `scripts/train.py` 的输出及 `log_interval`，`k8s_job.sh` 使用 `tee -a` 收集训练进程完整 stdout/stderr，包含 loss 变化、其他训练指标及异常 traceback。同一本地实验续训会追加已有日志。
- 本地文件为 `OPENPI_CHECKPOINT_DIR/<config>/<exp>.train.log`。正常训练结束或训练报错后，等待训练输出管线结束，最终上传一次至 `OPENPI_OSS_ROOT/<project>/<exp>/train.log`，默认 `<project>` 为配置名。这是训练进程日志；初始化和上传工具输出仍由原外层日志记录器处理。
- 日志先于最终 checkpoint 同步，使用 `ossutil cp -f` 覆盖同名对象，避免增量上传按时间戳跳过最终文件；checkpoint 上传失败不会阻断日志上传。日志不存在或上传失败会报错，训练失败时仍保留原训练退出码。本次不创建额外的指标文件。
- `scripts/k8s_job_test.py` 的 7 项隔离集成测试通过：完整 stdout/stderr/最后一行与本地文件一致、唯一最终上传、训练失败、日志/检查点上传失败、无 checkpoint、续训追加。测试执行真实 worker 脚本，训练进程和 OSS 工具由本地替身实现；没有执行真实 GPU 或 OSS 操作。shell 语法、Python 编译及 `git diff --check` 通过。

## 已做检查

- 提交回归使用本地 fake SDK 验证原 `h20` 八卡环境、配置名直通、资源请求、源码打包和 worker 环境优先级；不申请 GPU。修复 WebSocket 测试中 pytest 保留参数名 `request` 导致的收集错误。
- `pi0_rtc_test.py` 运行真实 CPU/JAX 检查：clean prefix、每样本后缀 reduction、前缀零梯度、delay 上界、AdaRMS 逐 token 调制，以及缩小网络容量后的真实 Euler 采样循环。
- `train_test.py::test_ttrtc_training_logs_checkpoint_and_resume` 使用真实 `train.main`、FakeDataset、缩小的视觉和语言网络（BF16、H50/D32），完成两步训练、checkpoint 保存和恢复后续训到第四步，检查全部训练指标为有限 scalar。这不是生产模型或八卡训练的性能测试。
- 完整 Pi05（Gemma 2B + 300M、BF16、H50/D32）与父提交的参数名称、shape、dtype 相同，51 个参数叶子，共 3,353,433,872 个参数；global batch=256 的原始训练步骤通过 CPU `jax.eval_shape` 前向、反向、AdamW 和 EMA 形状追踪。形状追踪不证明 GPU 显存和吞吐。
- 真实 WebSocket 平铺/嵌套请求往返，以及 UR7 绝对动作→相对动作→归一化→还原检查通过。普通 Pi05 loss 在固定参数、输入和 RNG 下与父提交数值一致。
- 本次分组测试结果：提交测试 5 项；模型及 WebSocket 测试 17 项；配置、UR7/RTC 策略、协议和异步客户端测试 16 项（包含上述 WebSocket 测试）；训练及真实维度指标测试 2 项。均通过。训练配置 CLI、`compileall`、shell 语法和 `git diff --check` 通过；未运行全仓库测试套件。
- 未连接云端或提交 GPU 任务。真实数据训练结果、GPU 显存/吞吐及闭环收益仍需在用户训练部署后确认。训练日志长期 `ttrtc_delay_mean≈5`、`ttrtc_postfix_fraction≈0.9` 只证明随机条件和 mask 工作，不能代替效果评估。

## 本次相关文件

```text
main.py                                      # TTRTC 卡数校验
k8s_job.sh                                   # 退出后优先上传完整 train.log
scripts/k8s_job_test.py                       # 日志完整性、上传及退出码
scripts/submission_test.py                    # 原八卡云端提交契约
scripts/train_test.py                         # TTRTC 训练、标量日志、保存和续训
src/openpi/models/pi0_rtc_test.py              # 条件、loss、AdaRMS 和逐步采样
src/openpi/serving/websocket_policy_server_test.py  # 修复测试收集
src/openpi/training/config_test.py             # 与普通配置逐字段比较
src/openpi/training/config.py                 # 仅纠正旧样本预算注释
README.md / FILE.md                           # 单配置切换与验证范围
```
