import dataclasses
import math
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class ReCAPConfig:
    """RECAP 条件训练参数; 默认关闭以保持普通 Pi0/Pi05 兼容。"""

    enabled: bool = False
    alpha: float = 1.0
    advantage_dropout_prob: float = 0.3
    # CFG 只在 RECAP 推理时生效; beta=2 对齐论文建议范围和参考实现默认值。
    guidance_scale: float = 2.0
    positive_fraction: float = 0.4
    value_bins: int = 201
    value_min: float = -1.0
    value_max: float = 0.0
    n_step_lookahead: int = 50

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("RECAP enabled must be a bool")
        if isinstance(self.alpha, bool) or not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("RECAP alpha must be non-negative")
        if (
            isinstance(self.advantage_dropout_prob, bool)
            or not math.isfinite(self.advantage_dropout_prob)
            or not 0.0 <= self.advantage_dropout_prob <= 1.0
        ):
            raise ValueError("advantage_dropout_prob must be in [0, 1]")
        if isinstance(self.guidance_scale, bool) or not math.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative")
        if (
            isinstance(self.positive_fraction, bool)
            or not math.isfinite(self.positive_fraction)
            or not 0.0 < self.positive_fraction <= 1.0
        ):
            raise ValueError("positive_fraction must be in (0, 1]")
        if isinstance(self.value_bins, bool) or not isinstance(self.value_bins, int) or self.value_bins < 2:
            raise ValueError("value_bins must be at least 2")
        if not math.isfinite(self.value_min) or not math.isfinite(self.value_max) or self.value_min >= self.value_max:
            raise ValueError("value_min must be smaller than value_max")
        if (
            isinstance(self.n_step_lookahead, bool)
            or not isinstance(self.n_step_lookahead, int)
            or self.n_step_lookahead < 1
        ):
            raise ValueError("n_step_lookahead must be >= 1")


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    recap: ReCAPConfig = dataclasses.field(default_factory=ReCAPConfig)

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                # 只在 RECAP 开启时扩展 pytree, 普通 Pi0/Pi05 的输入结构保持不变。
                advantage_indicator=(jax.ShapeDtypeStruct([batch_size], jnp.bool_) if self.recap.enabled else None),
                use_advantage=(jax.ShapeDtypeStruct([batch_size], jnp.bool_) if self.recap.enabled else None),
                is_human_intervention=(jax.ShapeDtypeStruct([batch_size], jnp.bool_) if self.recap.enabled else None),
                tokenized_advantage_positive=(
                    jax.ShapeDtypeStruct([batch_size, 8], jnp.int32) if self.recap.enabled else None
                ),
                tokenized_advantage_negative=(
                    jax.ShapeDtypeStruct([batch_size, 8], jnp.int32) if self.recap.enabled else None
                ),
                tokenized_advantage_mask=(
                    jax.ShapeDtypeStruct([batch_size, 8], jnp.bool_) if self.recap.enabled else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
