from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp

import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip


@dataclasses.dataclass(frozen=True)
class ValueModelConfig:
    """复用 OpenPI SigLIP/Gemma 的 RECAP distributional value 配置。"""

    state_dim: int = 32
    hidden_dim: int = 512
    value_bins: int = 201
    value_min: float = -1.0
    value_max: float = 0.0
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    freeze_mode: str = "none"
    num_cameras: int = 3
    max_token_len: int = 48
    dtype: str = "bfloat16"
    siglip_variant: str = "So400m/14"
    gemma_variant: _gemma.Variant = "gemma_300m"
    image_resolution: tuple[int, int] = (224, 224)

    def __post_init__(self) -> None:
        integer_fields = {
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "num_cameras": self.num_cameras,
            "max_token_len": self.max_token_len,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integer_fields.values()):
            raise ValueError(f"value model integer dimensions must be positive: {integer_fields}")
        if isinstance(self.value_bins, bool) or not isinstance(self.value_bins, int) or self.value_bins < 2:
            raise ValueError("value_bins must be at least 2")
        if not math.isfinite(self.value_min) or not math.isfinite(self.value_max) or self.value_min >= self.value_max:
            raise ValueError("value_min must be smaller than value_max")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.freeze_mode not in {"none", "backbones"}:
            raise ValueError("freeze_mode must be 'none' or 'backbones'")
        if len(self.image_resolution) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in self.image_resolution
        ):
            raise ValueError("image_resolution must contain two positive integers")
        try:
            jnp.dtype(self.dtype)
            _gemma.get_config(self.gemma_variant)
            _siglip.decode_variant(self.siglip_variant)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("unsupported value backbone or dtype configuration") from exc
