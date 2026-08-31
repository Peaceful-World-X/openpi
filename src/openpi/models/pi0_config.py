import dataclasses
import math
from typing import TYPE_CHECKING

from flax import traverse_util
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
class MEMConfig:
    """Pi0.6-MEM 可选输入配置; 所有开关默认关闭以保持旧模型兼容。"""

    use_video_memory: bool = False
    video_memory_frames: int = 6
    video_frame_stride_sec: float = 1.0
    temporal_attn_every_n_layers: int = 4
    drop_history_tokens_after_layer: int = -4
    use_state_history: bool = False
    use_language_memory: bool = False
    max_memory_tokens: int = 256

    def __post_init__(self) -> None:
        if (
            not isinstance(self.video_memory_frames, int)
            or isinstance(self.video_memory_frames, bool)
            or self.video_memory_frames < 1
        ):
            raise ValueError("video_memory_frames must be a positive integer")
        if not math.isfinite(self.video_frame_stride_sec) or self.video_frame_stride_sec <= 0:
            raise ValueError("video_frame_stride_sec must be positive")
        if (
            not isinstance(self.temporal_attn_every_n_layers, int)
            or isinstance(self.temporal_attn_every_n_layers, bool)
            or self.temporal_attn_every_n_layers < 1
        ):
            raise ValueError("temporal_attn_every_n_layers must be a positive integer")
        # Pi0 固定使用 27 层 So400m SigLIP, 丢弃位置必须能解析到真实层。
        if (
            not isinstance(self.drop_history_tokens_after_layer, int)
            or isinstance(self.drop_history_tokens_after_layer, bool)
            or not -27 <= self.drop_history_tokens_after_layer < 27
        ):
            raise ValueError("drop_history_tokens_after_layer must be an integer in [-27, 26]")
        if (
            not isinstance(self.max_memory_tokens, int)
            or isinstance(self.max_memory_tokens, bool)
            or self.max_memory_tokens < 1
        ):
            raise ValueError("max_memory_tokens must be a positive integer")


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

    pytorch_compile_mode: str | None = "max-autotune"

    # MEM 默认关闭, 放在原有字段之后以保持 Pi0Config 的位置参数兼容。
    mem: MEMConfig = dataclasses.field(default_factory=MEMConfig)

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
    def use_video_memory(self) -> bool:
        return self.mem.use_video_memory

    @property
    def use_state_history(self) -> bool:
        return self.mem.use_state_history

    @property
    def use_language_memory(self) -> bool:
        return self.mem.use_language_memory

    @property
    def video_memory_frames(self) -> int:
        return self.mem.video_memory_frames

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

    def load(self, params: at.Params, *, remove_extra_params: bool = True):
        """MEM 启用时把标准 scan 视觉参数按层迁移后再严格恢复。"""
        mem_enabled = self.mem.use_video_memory or self.mem.use_state_history or self.mem.use_language_memory
        if self.mem.use_video_memory or self.mem.use_state_history:
            from openpi.models.mem_checkpoint import migrate_scan_params

            # 大模型恢复继续使用 eval_shape, 避免仅为 checkpoint 对齐而分配一整套随机权重。
            model = nnx.eval_shape(self.create, jax.random.key(0))
            _, state = nnx.split(model)
            flat_keys = traverse_util.flatten_dict(params, sep="/")
            has_video_structure = any("encoderblock_0" in key for key in flat_keys)
            has_state_history = any("state_history_proj" in key for key in flat_keys)
            # 只有目标要求的每种结构都存在才视为可严格恢复的完整 MEM checkpoint。
            is_mem_checkpoint = (not self.mem.use_video_memory or has_video_structure) and (
                not self.mem.use_state_history or has_state_history
            )
            if not is_mem_checkpoint:
                target_params = state.to_pure_dict()
                if self.mem.use_state_history:
                    # 只实例化新增的小型线性层, 并把真实初始化数组写入目标规格;
                    # 其余大模型叶仍保持 ShapeDtypeStruct, 迁移时必须由 checkpoint 提供。
                    projection = nnx.Linear(
                        self.action_dim,
                        _gemma.get_config(self.paligemma_variant).width,
                        rngs=nnx.Rngs(jax.random.key(0)),
                    )
                    _, projection_state = nnx.split(projection)
                    target_flat = traverse_util.flatten_dict(target_params, sep="/")
                    projection_flat = traverse_util.flatten_dict(projection_state.to_pure_dict(), sep="/")
                    for key in tuple(target_flat):
                        marker = "state_history_proj/"
                        if marker in key:
                            target_flat[key] = projection_flat[key.split(marker, maxsplit=1)[1]]
                    target_params = traverse_util.unflatten_dict(target_flat, sep="/")
                params = migrate_scan_params(params, target_params)
            remove_extra_params = False
        elif mem_enabled:
            # language-only MEM 不增加参数, 但仍要求 checkpoint 与模型树严格一致。
            remove_extra_params = False
        return super().load(params, remove_extra_params=remove_extra_params)

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
                image_history=(
                    {
                        key: jax.ShapeDtypeStruct(
                            [batch_size, self.mem.video_memory_frames - 1, *_model.IMAGE_RESOLUTION, 3], jnp.float32
                        )
                        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                    }
                    if self.mem.use_video_memory
                    else None
                ),
                image_history_masks=(
                    {
                        key: jax.ShapeDtypeStruct([batch_size, self.mem.video_memory_frames - 1], jnp.bool_)
                        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                    }
                    if self.mem.use_video_memory
                    else None
                ),
                state_history=(
                    jax.ShapeDtypeStruct([batch_size, self.mem.video_memory_frames - 1, self.action_dim], jnp.float32)
                    if self.mem.use_state_history
                    else None
                ),
                tokenized_memory=(
                    jax.ShapeDtypeStruct([batch_size, self.mem.max_memory_tokens], jnp.int32)
                    if self.mem.use_language_memory
                    else None
                ),
                tokenized_memory_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.mem.max_memory_tokens], jnp.bool_)
                    if self.mem.use_language_memory
                    else None
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
