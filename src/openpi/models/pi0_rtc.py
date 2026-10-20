"""Training-time and inference-time TTRTC extensions for the JAX Pi0 model."""

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_rtc_config
from openpi.shared import array_typing as at


def _build_rtc_training_inputs(actions, noise, time, delay):
    """Build clean-prefix/noisy-postfix actions and per-token flow times."""
    action_horizon = actions.shape[-2]
    prefix_mask = jnp.arange(action_horizon) < delay[..., None]
    token_time = jnp.where(prefix_mask, 0.0, time[..., None])
    token_time_expanded = token_time[..., :, None]
    x_t = token_time_expanded * noise + (1.0 - token_time_expanded) * actions
    return x_t, token_time, prefix_mask


def _mask_rtc_loss(raw_loss, prefix_mask):
    """Reduce raw [B,H,D] MSE to [B,H] with a per-sample postfix mean."""
    postfix_mask = jnp.logical_not(prefix_mask)
    per_token_loss = jnp.mean(raw_loss, axis=-1)
    postfix_count = jnp.sum(postfix_mask, axis=-1)
    # Scaling each valid token by H / postfix_count means the outer training
    # mean over [B,H] is exactly the mean of postfix-token means per sample.
    horizon = raw_loss.shape[-2]
    scale = horizon / jnp.maximum(postfix_count, 1)
    return per_token_loss * postfix_mask * scale[..., None]


class Pi0RTC(_pi0.Pi0):
    """Pi0 with training and inference-time clean-prefix conditioning."""

    def __init__(self, config: pi0_rtc_config.Pi0RTCConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.max_delay = config.max_delay

    def _compute_loss_details(self, rng, observation, actions, *, train=False):
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        # Include max_delay in the sample range while validation guarantees a
        # non-empty postfix for every sampled delay.
        delay = jax.random.randint(delay_rng, batch_shape, 0, self.max_delay + 1)
        x_t, token_time, prefix_mask = _build_rtc_training_inputs(actions, noise, time, delay)
        raw_loss = self._compute_loss_from_inputs(observation, x_t, noise - actions, token_time)
        return _mask_rtc_loss(raw_loss, prefix_mask), {
            "raw_loss": raw_loss,
            "postfix_mask": jnp.logical_not(prefix_mask),
            "delay": delay,
        }

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
        """采样动作并冻结延迟窗口内的 clean prefix。

        策略层传入模型归一化空间中的完整 ``[batch, horizon, action_dim]``
        前缀；前 ``delay`` 个 token 作为干净条件，其余 token 由标准 Euler
        采样生成。未提供 RTC 参数时保持官方 Pi0 采样路径。
        """
        if action_prefix is None and delay is None:
            return super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)
        if action_prefix is None or delay is None:
            raise ValueError("action_prefix and delay must be provided together")
        action_prefix = jnp.asarray(action_prefix)
        delay = jnp.asarray(delay)
        if not jnp.issubdtype(delay.dtype, jnp.integer):
            raise ValueError(f"delay must have an integer dtype, got {delay.dtype}")

        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        expected_prefix_shape = (batch_size, self.action_horizon, self.action_dim)
        if action_prefix.shape != expected_prefix_shape:
            raise ValueError(f"Expected action_prefix shape {expected_prefix_shape}, got {action_prefix.shape}")
        if delay.shape != (batch_size,):
            raise ValueError(f"Expected delay shape {(batch_size,)}, got {delay.shape}")
        # 配置范围由策略层校验；这里仅构造 JIT 兼容的逐 token mask。
        action_prefix_mask = jnp.arange(self.action_horizon)[None, :] < delay[:, None]

        # 与官方采样器一致，只对图像/语言 prefix 填充一次 KV cache。
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            # 每个 Euler 步重新写入 clean prefix，避免整段动作更新时发生漂移。
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
            # 对最后一步的浮点误差保持鲁棒。
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        # 最后一次更新也会触及 prefix，返回前再次严格恢复原值。
        return jnp.where(action_prefix_mask[..., None], action_prefix, x_0)
