import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_rtc_config
from openpi.shared import array_typing as at


# 按论文算法 1 构造干净动作前缀、逐动作流时间和前缀掩码。
def _build_rtc_training_inputs(actions, noise, time, delay):
    action_horizon = actions.shape[-2]
    prefix_mask = jnp.arange(action_horizon) < delay[..., None]

    # 论文令前缀 tau=1 (数据端); openpi 的流方向相反, 因此这里对应 time=0 (干净数据端)。
    token_time = jnp.where(prefix_mask, 0.0, time[..., None])
    x_t = token_time[..., None] * noise + (1 - token_time[..., None]) * actions
    return x_t, token_time, prefix_mask


# 将前缀损失置零并补偿有效后缀数量, 使外层 mean 等价于只平均后缀。
def _mask_rtc_loss(loss, prefix_mask):
    postfix_mask = jnp.logical_not(prefix_mask)
    postfix_count = jnp.sum(postfix_mask)
    normalizer = jnp.asarray(loss.size, dtype=loss.dtype) / jnp.maximum(postfix_count, 1)
    return loss * postfix_mask * normalizer


class Pi0RTC(_pi0.Pi0):
    """通过干净动作前缀实现 Training-Time RTC 的 Pi0 子类。"""

    def __init__(self, config: pi0_rtc_config.Pi0RTCConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        # max_delay 只控制 RTC 的随机延迟和服务校验, 不注册参数以保持 checkpoint 参数树不变。
        self.max_delay = config.max_delay

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # RTC 比标准路径多拆一个 delay key, 并保持既有 noise、time、delay 的固定顺序。
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        return self._compute_loss_training_time_rtc(noise_rng, time_rng, delay_rng, observation, actions)

    def _compute_loss_training_time_rtc(self, noise_rng, time_rng, delay_rng, observation, actions):
        """计算 Training-Time RTC 损失, 只让动作后缀产生梯度。"""
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        # 论文允许选择延迟分布; 这里沿用当前实现的均匀整数分布 [0, max_delay)。
        delay = jax.random.randint(delay_rng, batch_shape, 0, self.max_delay)
        # 前缀保持干净, 后缀继续使用官方 Beta 时间和加噪方向。
        x_t, model_time, prefix_mask = _build_rtc_training_inputs(actions, noise, time, delay)
        loss = self._compute_loss_from_inputs(observation, x_t, noise - actions, model_time)
        # 论文第 3 节要求梯度只来自动作后缀。
        return _mask_rtc_loss(loss, prefix_mask)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        action_prefix: at.Float[at.Array, "b ah ad"] | None = None,
        delay: at.Int[at.Array, " b"] | None = None,
    ) -> _model.Actions:
        # 普通请求直接复用父类采样器, 保持标准 Pi0 的时间形状与数值路径。
        if action_prefix is None and delay is None:
            return super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)
        if action_prefix is None or delay is None:
            raise ValueError("action_prefix and delay must be provided together")

        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # 固定协议形状使逐 token 条件与动作时域一一对应, 避免 JIT 中发生静默广播。
        expected_prefix_shape = (batch_size, self.action_horizon, self.action_dim)
        if action_prefix.shape != expected_prefix_shape:
            raise ValueError(f"Expected action_prefix shape {expected_prefix_shape}, got {action_prefix.shape}")
        if delay.shape != (batch_size,):
            raise ValueError(f"Expected delay shape {(batch_size,)}, got {delay.shape}")
        action_prefix_mask = jnp.arange(self.action_horizon)[None, :] < delay[:, None]

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            # 每个 Euler 步重新注入干净前缀, 防止已执行动作被去噪更新污染。
            x_t = jnp.where(action_prefix_mask[..., None], action_prefix, x_t)
            model_time = jnp.where(
                action_prefix_mask,
                0.0,
                jnp.broadcast_to(time, (batch_size, self.action_horizon)),
            )
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, model_time
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        # 最后一次 Euler 更新仍会改变前缀, 因此返回前再次回填以严格保持冻结不变量。
        return jnp.where(action_prefix_mask[..., None], action_prefix, x_0)
