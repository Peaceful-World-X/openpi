import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    # MEM 可选字段: 历史帧、历史状态和长期语言记忆。默认 None 保持旧接口。
    # 使用 object 避免 jaxtyping 对可变 K/M 维度做跨字段推断, 具体形状在 from_dict 中校验。
    image_history: dict[str, object] | None = None
    image_history_masks: dict[str, object] | None = None
    state_history: object | None = None
    tokenized_memory: object | None = None
    tokenized_memory_mask: object | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")

        def normalize_image(value, *, torch_channel_first: bool = False):
            """将 uint8 图像(包括历史帧)转换为模型使用的 [-1, 1]。"""
            if getattr(value, "dtype", None) == np.uint8:
                return value.astype(np.float32) / 255.0 * 2.0 - 1.0
            if getattr(value, "dtype", None) == torch.uint8:
                value = value.to(torch.float32)
                if torch_channel_first and getattr(value, "ndim", 0) == 4:
                    value = value.permute(0, 3, 1, 2)
                return value / 255.0 * 2.0 - 1.0
            return value

        def bool_array(value):
            """转换 mask dtype 时保留 NumPy/JAX/PyTorch 后端, 避免数据加载时回拷到主机。"""
            if isinstance(value, torch.Tensor):
                return value.to(dtype=torch.bool)
            if hasattr(value, "astype"):
                return value.astype(np.bool_)
            return np.asarray(value, dtype=np.bool_)

        def ones_mask_like(frames, shape):
            """缺省历史 mask 与帧数组保持相同设备和数组后端。"""
            if isinstance(frames, torch.Tensor):
                return torch.ones(shape, dtype=torch.bool, device=frames.device)
            if isinstance(frames, jax.Array):
                return jnp.ones(shape, dtype=jnp.bool_)
            return np.ones(shape, dtype=np.bool_)

        images = {key: normalize_image(value, torch_channel_first=True) for key, value in data["image"].items()}
        image_history = None
        image_history_masks = data.get("image_history_masks", data.get("image_history_mask"))
        if data.get("image_history") is None and image_history_masks is not None:
            raise ValueError("image_history_masks cannot be provided without image_history")
        if data.get("image_history") is not None:
            image_history = {key: normalize_image(value) for key, value in data["image_history"].items()}
            if not set(image_history).issubset(images):
                raise ValueError("image_history camera keys must be a subset of image keys")
            if image_history_masks is None:
                image_history_masks = {
                    key: ones_mask_like(frames, frames.shape[:-3]) for key, frames in image_history.items()
                }
            elif set(image_history_masks) != set(image_history):
                raise ValueError("image_history and image_history_masks must contain the same camera keys")
            else:
                image_history_masks = {key: bool_array(mask) for key, mask in image_history_masks.items()}
            history_prefix = None
            batch_prefix = data["state"].shape[:-1]
            for key, frames in image_history.items():
                if getattr(frames, "ndim", 0) < 4:
                    raise ValueError(f"image_history[{key!r}] must have shape (*b, k, h, w, c), got {frames.shape}")
                if frames.shape[-1] != 3:
                    raise ValueError(
                        f"image_history[{key!r}] must have RGB channels in the last dimension, got {frames.shape}"
                    )
                if frames.shape[:-4] != batch_prefix:
                    raise ValueError(
                        f"image_history[{key!r}] batch prefix {frames.shape[:-4]} does not match state prefix "
                        f"{batch_prefix}"
                    )
                if history_prefix is None:
                    history_prefix = frames.shape[:-3]
                elif frames.shape[:-3] != history_prefix:
                    raise ValueError(
                        f"all image_history cameras must share prefix shape {history_prefix}, got {frames.shape[:-3]}"
                    )
                if image_history_masks[key].shape != frames.shape[:-3]:
                    raise ValueError(
                        f"image_history_masks[{key!r}] shape {image_history_masks[key].shape} "
                        f"does not match history prefix {frames.shape[:-3]}"
                    )
        state_history = data.get("state_history")
        if state_history is not None:
            if getattr(state_history, "ndim", 0) < 2:
                raise ValueError(f"state_history must have shape (*b, k, s), got {state_history.shape}")
            state_prefix = data["state"].shape[:-1]
            if state_history.shape[:-2] != state_prefix:
                raise ValueError(
                    f"state_history batch prefix {state_history.shape[:-2]} does not match state prefix {state_prefix}"
                )
            if state_history.shape[-1] != data["state"].shape[-1]:
                raise ValueError(
                    f"state_history state dimension {state_history.shape[-1]} does not match current state "
                    f"dimension {data['state'].shape[-1]}"
                )
            if image_history_masks:
                expected_prefix = next(iter(image_history_masks.values())).shape
                if state_history.shape[:-1] != expected_prefix:
                    raise ValueError(
                        f"state_history shape {state_history.shape} does not match history prefix {expected_prefix}"
                    )
        if (data.get("tokenized_memory") is None) != (data.get("tokenized_memory_mask") is None):
            raise ValueError("tokenized_memory and tokenized_memory_mask must be provided together")
        if data.get("tokenized_memory") is not None:
            memory = data["tokenized_memory"]
            memory_mask = data["tokenized_memory_mask"]
            if getattr(memory, "ndim", 0) < 1 or memory.shape != memory_mask.shape:
                raise ValueError(
                    "tokenized_memory and tokenized_memory_mask must have identical shape, "
                    f"got {memory.shape} and {memory_mask.shape}"
                )
            if memory.shape[:-1] != data["state"].shape[:-1]:
                raise ValueError(
                    f"tokenized_memory batch prefix {memory.shape[:-1]} does not match state prefix "
                    f"{data['state'].shape[:-1]}"
                )
        return cls(
            images=images,
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            image_history=image_history,
            image_history_masks=image_history_masks,
            state_history=state_history,
            tokenized_memory=data.get("tokenized_memory"),
            tokenized_memory_mask=data.get("tokenized_memory_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    # 历史帧只做批量 resize, 不做随机增强, 避免同一时间窗口内产生不一致视图。
    out_image_history = None
    out_image_history_masks = observation.image_history_masks
    if observation.image_history is not None:
        out_image_history = {}
        for key in image_keys:
            if key not in observation.image_history:
                continue
            history = observation.image_history[key]
            if history.shape[-3:-1] != image_resolution:
                original_shape = history.shape
                flat = history.reshape((-1, *history.shape[-3:]))
                flat = image_tools.resize_with_pad(flat, *image_resolution)
                history = flat.reshape((*original_shape[:-3], *image_resolution, original_shape[-1]))
            out_image_history[key] = history

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        image_history=out_image_history,
        image_history_masks=out_image_history_masks,
        state_history=observation.state_history,
        tokenized_memory=observation.tokenized_memory,
        tokenized_memory_mask=observation.tokenized_memory_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
