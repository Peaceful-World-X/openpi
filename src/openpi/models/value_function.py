from __future__ import annotations

from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import siglip as _siglip
from openpi.models.value_model_config import ValueModelConfig


def _validate_support(num_bins: int, value_min: float, value_max: float) -> None:
    """校验 distributional value support, 防止公共数学 API 静默产生 NaN。"""
    if isinstance(num_bins, bool | np.bool_) or not isinstance(num_bins, int | np.integer) or num_bins < 2:
        raise ValueError("num_bins must be an integer >= 2")
    if not np.isfinite(value_min) or not np.isfinite(value_max) or value_min >= value_max:
        raise ValueError("value_min and value_max must be finite with value_min < value_max")


def value_to_bin(values: Any, *, num_bins: int = 201, value_min: float = -1.0, value_max: float = 0.0):
    """将连续 value 映射到离散 bin, 边界外值先裁剪。"""
    _validate_support(num_bins, value_min, value_max)
    values = jnp.asarray(values, dtype=jnp.float32)
    scaled = (jnp.clip(values, value_min, value_max) - value_min) / (value_max - value_min)
    return jnp.floor(scaled * (num_bins - 1)).astype(jnp.int32)


def bin_to_value(bins: Any, *, num_bins: int = 201, value_min: float = -1.0, value_max: float = 0.0):
    _validate_support(num_bins, value_min, value_max)
    bins = jnp.asarray(bins, dtype=jnp.float32)
    return jnp.clip(bins, 0, num_bins - 1) / float(num_bins - 1) * (value_max - value_min) + value_min


def get_supports(*, num_bins: int = 201, value_min: float = -1.0, value_max: float = 0.0) -> jax.Array:
    _validate_support(num_bins, value_min, value_max)
    return jnp.linspace(value_min, value_max, num_bins)


def two_hot(values: Any, *, num_bins: int = 201, value_min: float = -1.0, value_max: float = 0.0) -> jax.Array:
    """使用相邻两个 bin 的线性插值表示 scalar value target。"""
    _validate_support(num_bins, value_min, value_max)
    values = jnp.asarray(values, dtype=jnp.float32)
    scaled = (jnp.clip(values, value_min, value_max) - value_min) / (value_max - value_min) * (num_bins - 1)
    lower = jnp.floor(scaled).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, num_bins - 1)
    upper_weight = scaled - lower.astype(jnp.float32)
    result = jax.nn.one_hot(lower, num_bins) * (1.0 - upper_weight)[..., None]
    return result + jax.nn.one_hot(upper, num_bins) * upper_weight[..., None]


def distributional_cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    logits = jnp.asarray(logits)
    targets = jnp.asarray(targets, dtype=logits.dtype)
    if logits.ndim < 1 or logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have identical shapes, got {logits.shape} and {targets.shape}")
    return -jnp.sum(targets * jax.nn.log_softmax(logits, axis=-1), axis=-1)


def dist_to_value(logits: jax.Array, *, value_min: float = -1.0, value_max: float = 0.0) -> jax.Array:
    probabilities = jax.nn.softmax(jnp.asarray(logits, dtype=jnp.float32), axis=-1)
    if probabilities.ndim < 1:
        raise ValueError("logits must have a bin dimension")
    _validate_support(int(probabilities.shape[-1]), value_min, value_max)
    supports = get_supports(num_bins=logits.shape[-1], value_min=value_min, value_max=value_max)
    return jnp.sum(probabilities * supports, axis=-1)


def normalize_value_target_shape(target: Any, *, num_bins: int = 201) -> jax.Array:
    target = jnp.asarray(target, dtype=jnp.float32)
    if target.ndim == 0:
        return target[None]
    if target.ndim == 1:
        return target
    if target.ndim == 2 and target.shape[-1] == 1:
        return jnp.squeeze(target, axis=-1)
    if target.ndim == 2 and target.shape[-1] == num_bins:
        return target
    raise ValueError(f"unsupported value target shape {target.shape}")


def _full_attention_mask(input_mask: jax.Array) -> jax.Array:
    return jnp.logical_and(input_mask[:, :, None], input_mask[:, None, :])


class DistributionalValueModel(nnx.Module):
    """SigLIP + Gemma 深度融合的 RECAP distributional value function。"""

    def __init__(self, config: ValueModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        gemma_config = _gemma.get_config(config.gemma_variant)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(configs=[gemma_config], embed_dtype=config.dtype, adarms=False)
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=gemma_config.width,
                variant=config.siglip_variant,
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        fake_image = jnp.zeros((1, *config.image_resolution, 3), dtype=jnp.float32)
        img.lazy_init(fake_image, train=False, rngs=rngs)
        # 沿用 Pi0 的命名, 便于后续从兼容 OpenPI checkpoint 选择性加载 backbone。
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.state_proj = nnx.Linear(config.state_dim, gemma_config.width, rngs=rngs)
        self.value_norm = nnx.LayerNorm(gemma_config.width, rngs=rngs)
        self.value_hidden = nnx.Linear(gemma_config.width, config.hidden_dim, rngs=rngs)
        self.value_head = nnx.Linear(config.hidden_dim, config.value_bins, rngs=rngs)

    def _observation(self, observation: _model.Observation | dict[str, Any]) -> _model.Observation:
        if isinstance(observation, _model.Observation):
            return jax.tree.map(jnp.asarray, observation)
        data = dict(observation)
        if "images" in data and "image" not in data:
            data["image"] = data.pop("images")
        if "image_masks" in data and "image_mask" not in data:
            data["image_mask"] = data.pop("image_masks")
        if "tokenized_prompt" not in data or "tokenized_prompt_mask" not in data:
            raise ValueError("value observations must contain tokenized_prompt and tokenized_prompt_mask")
        images = data.get("image", {})
        if not isinstance(images, dict):
            raise TypeError("value observation image must be a dictionary")
        data["image"] = {key: jnp.asarray(value) for key, value in images.items()}
        # 在进入 JIT 前把外部 list 转数组, 同时保留 batch 维度信息。
        data["state"] = jnp.asarray(data["state"])
        if data["state"].ndim < 2:
            raise ValueError("value observation state must have shape [batch, state_dim]")
        batch_size = data["state"].shape[0]
        data.setdefault("image_mask", {key: np.ones(batch_size, dtype=np.bool_) for key in images})
        if not isinstance(data["image_mask"], dict):
            raise TypeError("value observation image_mask must be a dictionary")
        # Value CLI 不经过训练主循环的 device placement, 需要在模型边界显式转为 JAX array。
        return jax.tree.map(jnp.asarray, _model.Observation.from_dict(data))

    def _embed_inputs(
        self, observation: _model.Observation, *, train: bool
    ) -> tuple[jax.Array, jax.Array]:
        tokens = []
        masks = []
        for name in sorted(observation.images)[: self.config.num_cameras]:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=train)
            tokens.append(image_tokens)
            masks.append(einops.repeat(observation.image_masks[name], "b -> b s", s=image_tokens.shape[1]))

        if observation.tokenized_prompt is not None:
            tokens.append(self.PaliGemma.llm(observation.tokenized_prompt, method="embed"))
            masks.append(observation.tokenized_prompt_mask)

        state = jnp.asarray(observation.state, dtype=jnp.float32)[..., : self.config.state_dim]
        state = jnp.pad(state, [(0, 0), (0, max(0, self.config.state_dim - state.shape[-1]))])
        state_token = self.state_proj(state)[:, None, :]
        tokens.append(state_token)
        masks.append(jnp.ones(state_token.shape[:2], dtype=jnp.bool_))
        return jnp.concatenate(tokens, axis=1), jnp.concatenate(masks, axis=1)

    def compute_logits(
        self,
        observation: _model.Observation | dict[str, Any],
        *,
        train: bool = False,
    ) -> jax.Array:
        observation = self._observation(observation)
        image_keys = tuple(sorted(observation.images)[: self.config.num_cameras])
        observation = _model.preprocess_observation(
            None,
            observation,
            train=False,
            image_keys=image_keys,
            image_resolution=self.config.image_resolution,
        )
        embedded, input_mask = self._embed_inputs(observation, train=train)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (encoded,), _ = self.PaliGemma.llm(
            [embedded],
            positions=positions,
            mask=_full_attention_mask(input_mask),
            deterministic=not train,
        )
        mask = input_mask[..., None].astype(encoded.dtype)
        pooled = jnp.sum(encoded * mask, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1)
        hidden = nnx.gelu(self.value_hidden(self.value_norm(pooled)))
        return self.value_head(hidden).astype(jnp.float32)

    def __call__(self, observation: _model.Observation | dict[str, Any], *, train: bool = False) -> jax.Array:
        return self.compute_logits(observation, train=train)

    def compute_value(self, observation: _model.Observation | dict[str, Any]) -> jax.Array:
        return dist_to_value(
            self.compute_logits(observation, train=False),
            value_min=self.config.value_min,
            value_max=self.config.value_max,
        )

    def compute_loss(
        self,
        observation: _model.Observation | dict[str, Any],
        targets: Any,
        *,
        train: bool = False,
    ) -> jax.Array:
        logits = self.compute_logits(observation, train=train)
        target_array = normalize_value_target_shape(targets, num_bins=self.config.value_bins)
        target_distribution = (
            target_array
            if target_array.ndim == 2 and target_array.shape[-1] == self.config.value_bins
            else two_hot(
                target_array,
                num_bins=self.config.value_bins,
                value_min=self.config.value_min,
                value_max=self.config.value_max,
            )
        )
        if target_distribution.shape != logits.shape:
            raise ValueError(
                f"value target distribution shape {target_distribution.shape} does not match logits {logits.shape}"
            )
        return jnp.mean(distributional_cross_entropy(logits, target_distribution))
