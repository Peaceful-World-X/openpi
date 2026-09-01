from collections.abc import Callable, Mapping, Sequence
import dataclasses
import functools
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats

# RECAP sidecar 字段不是机器人观测, 但必须穿过各硬件适配器进入模型 token transform。
RECAP_FIELDS = ("advantage_indicator", "use_advantage", "is_human_intervention")
RECAP_CANONICAL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def copy_recap_fields(source: Mapping[str, object], target: dict) -> dict:
    """将可选 RECAP sidecar 字段透传到适配器输出, 保持普通数据 schema 不变。"""
    for key in RECAP_FIELDS:
        if key in source:
            target[key] = source[key]
    return target


def canonicalize_recap_image(image: object, *, key: str) -> np.ndarray:
    """统一 LeRobot 图像的 CHW/HWC 和数值范围, 防止 SigLIP 收到错误布局。"""
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"RECAP image {key!r} must be a 3-D array, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"RECAP image {key!r} must be HWC/CHW RGB, got {image.shape}")
    if image.dtype == np.uint8:
        return image
    if not np.issubdtype(image.dtype, np.floating) or not np.all(np.isfinite(image)):
        raise ValueError(f"RECAP image {key!r} must contain finite uint8 or floating-point values")
    minimum, maximum = float(np.min(image)), float(np.max(image))
    if minimum >= -1.001 and maximum <= 1.001:
        # LeRobot 常见的 float 图像是 [0, 1], 而 canonical OpenPI 图像要求 [-1, 1]。
        return (image * 2.0 - 1.0 if minimum >= -1e-6 else image).astype(np.float32)
    if minimum >= 0.0 and maximum <= 255.0:
        return (image / 255.0 * 2.0 - 1.0).astype(np.float32)
    raise ValueError(f"RECAP image {key!r} must be in [0, 1], [-1, 1], or [0, 255], got [{minimum}, {maximum}]")


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            source_recap = (
                {key: data[key] for key in RECAP_FIELDS if key in data}
                if isinstance(data, Mapping)
                else {}
            )
            data = transform(data)
            if isinstance(data, dict):
                # 任意机器人适配器都可能重建 dict; 在统一组合边界保留可选 sidecar。
                copy_recap_fields(source_recap, data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class ReCAPLeRobotInputs(DataTransformFn):
    """将明确约定的 LeRobot 列转换为 Pi0/Pi05 canonical 输入。"""

    image_keys: tuple[str, ...] = RECAP_CANONICAL_IMAGE_KEYS
    image_column_prefix: str = "observation.images."
    state_key: str = "observation.state"
    action_key: str = "action"

    def __call__(self, data: DataDict) -> DataDict:
        canonical_images = data.get("image")
        if isinstance(canonical_images, Mapping):
            images = {key: canonical_images[key] for key in self.image_keys if key in canonical_images}
        else:
            images = {
                key: data[f"{self.image_column_prefix}{key}"]
                for key in self.image_keys
                if f"{self.image_column_prefix}{key}" in data
            }
        missing_images = [key for key in self.image_keys if key not in images]
        if missing_images:
            raise ValueError(
                "RECAP LeRobot data is missing canonical camera columns for "
                f"{missing_images}; provide a robot-specific TrainConfig instead of guessing camera semantics"
            )
        state = data.get("state", data.get(self.state_key))
        actions = data.get("actions", data.get(self.action_key))
        if state is None:
            raise ValueError(f"RECAP data requires state or {self.state_key!r}")
        raw_masks = data.get("image_mask", {})
        if raw_masks is None:
            raw_masks = {}
        if not isinstance(raw_masks, Mapping):
            raise TypeError("canonical image_mask must be a mapping")
        result = {
            "image": {key: canonicalize_recap_image(images[key], key=key) for key in self.image_keys},
            "image_mask": {
                key: np.asarray(raw_masks.get(key, True), dtype=np.bool_) for key in self.image_keys
            },
            "state": state,
        }
        # 推理 observation 没有 action; 训练样本存在时再透传。
        if actions is not None:
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return copy_recap_fields(data, result)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class InjectReCAPInferenceCondition(DataTransformFn):
    """为已训练 RECAP 策略注入默认 positive 条件, 不参与训练数据处理。"""

    def __call__(self, data: DataDict) -> DataDict:
        result = dict(data)
        # 部署默认采样改进策略; 调用方仍可显式传 False 获取 negative 或无条件输出。
        result.setdefault("advantage_indicator", np.ones((), dtype=np.bool_))
        result.setdefault("use_advantage", np.ones((), dtype=np.bool_))
        result.setdefault("is_human_intervention", np.zeros((), dtype=np.bool_))
        return result


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeReCAPAdvantage(DataTransformFn):
    """把固定正负优势描述编码为 token, 供 Pi0/Pi05 的 prefix 使用。"""

    tokenizer: _tokenizer.PaligemmaTokenizer
    max_len: int = 8

    def __post_init__(self) -> None:
        if self.max_len < 1:
            raise ValueError("RECAP advantage token length must be positive")

    @functools.cached_property
    def _condition_tokens(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """固定文本每个 worker 只编码一次, 避免逐帧重复调用 tokenizer。"""
        positive, positive_mask = self.tokenizer.tokenize("Advantage: positive")
        negative, negative_mask = self.tokenizer.tokenize("Advantage: negative")

        def fixed(tokens, mask):
            tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)[: self.max_len]
            mask = np.asarray(mask, dtype=np.bool_).reshape(-1)[: self.max_len]
            if len(tokens) < self.max_len:
                pad = self.max_len - len(tokens)
                tokens = np.pad(tokens, (0, pad))
                mask = np.pad(mask, (0, pad), constant_values=False)
            return tokens, mask

        positive, positive_mask = fixed(positive, positive_mask)
        negative, negative_mask = fixed(negative, negative_mask)
        if not np.array_equal(positive_mask, negative_mask):
            # Observation 契约只有一个条件 mask; 长度不一致时必须失败, 不能 OR 后放开 padding。
            raise ValueError("positive and negative RECAP conditions must have identical token masks")
        return positive, negative, positive_mask

    def __call__(self, data: DataDict) -> DataDict:
        if "advantage_indicator" not in data:
            return data
        positive, negative, advantage_mask = self._condition_tokens
        result = dict(data)
        result["tokenized_advantage_positive"] = positive
        result["tokenized_advantage_negative"] = negative
        result["tokenized_advantage_mask"] = advantage_mask
        if "use_advantage" not in result:
            result["use_advantage"] = np.ones((), dtype=np.bool_)
        if "is_human_intervention" not in result:
            result["is_human_intervention"] = np.zeros((), dtype=np.bool_)
        return result


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
