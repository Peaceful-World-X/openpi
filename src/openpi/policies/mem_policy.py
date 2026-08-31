"""JAX Pi0/Pi05 MEM 推理策略和 episode 历史缓冲。"""

from collections import deque
from collections.abc import Mapping
import logging
from typing import Any

import numpy as np
from typing_extensions import override

from openpi.models.high_level_policy import HighLevelPolicy
from openpi.policies.policy import Policy

logger = logging.getLogger(__name__)


_CAMERA_ALIASES = {
    "base_0_rgb": (
        "base_0_rgb",
        "cam_high",
        "observation.images.top",
        "observation.images.cam_high",
        "observation.images.base",
        "observation.images.exterior_image_1_left",
        "observation/exterior_image_1_left",
        "observation/image",
        "images/cam_high",
        "exterior_image_1_left",
    ),
    "left_wrist_0_rgb": (
        "left_wrist_0_rgb",
        "cam_left_wrist",
        "observation.images.left_wrist",
        "observation.images.cam_left_wrist",
        "observation.images.wrist_image_left",
        "observation/wrist_image_left",
        "observation/wrist_image",
        "images/cam_left_wrist",
        "wrist_image_left",
        "wrist_image",
    ),
    "right_wrist_0_rgb": (
        "right_wrist_0_rgb",
        "cam_right_wrist",
        "observation.images.right_wrist",
        "observation.images.cam_right_wrist",
        "observation.images.wrist_image_right",
        "observation/wrist_image_right",
        "images/cam_right_wrist",
        "wrist_image_right",
    ),
}


class FrameHistoryBuffer:
    """按相机维护 K-1 个历史帧, 并生成 episode 起始 padding mask。"""

    def __init__(
        self,
        num_frames: int,
        image_key_map: Mapping[str, str] | None = None,
        state_key: str = "observation.state",
    ) -> None:
        if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames < 1:
            raise ValueError("num_frames must be a positive integer")
        self.num_frames = num_frames
        self.history_length = num_frames - 1
        self.image_key_map = self._canonicalize_key_map(image_key_map or {})
        self.state_key = state_key
        maxlen = max(1, self.history_length)
        self._frames = {key: deque(maxlen=maxlen) for key in _CAMERA_ALIASES}
        self._states: deque[np.ndarray] = deque(maxlen=maxlen)

    def reset(self) -> None:
        for buffer in self._frames.values():
            buffer.clear()
        self._states.clear()

    def extract_images(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """把 canonical、ALOHA、DROID 和 Libero 相机键统一为模型相机名。"""
        images: dict[str, np.ndarray] = {}
        for canonical, aliases in _CAMERA_ALIASES.items():
            custom = self.image_key_map.get(canonical)
            value = self._find_value(observation, (custom, *aliases) if custom else aliases)
            if value is not None:
                images[canonical] = np.asarray(value)
        return images

    def extract_state(self, observation: Mapping[str, Any]) -> np.ndarray | None:
        value = self._find_value(observation, (self.state_key, "state", "observation.state", "observation/state"))
        if value is not None:
            return np.asarray(value)
        joints = self._find_value(
            observation,
            ("observation/joint_position", "observation.joint_position", "joint_position"),
        )
        gripper = self._find_value(
            observation,
            ("observation/gripper_position", "observation.gripper_position", "gripper_position"),
        )
        if joints is None or gripper is None:
            return None
        return np.concatenate([np.asarray(joints).reshape(-1), np.asarray(gripper).reshape(-1)])

    def build(
        self,
        current_images: Mapping[str, np.ndarray],
        current_state: np.ndarray | None,
        *,
        include_images: bool,
        include_state: bool,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray | None]:
        """历史顺序固定为最旧到最新, padding 复制最早可用帧但 mask=False。"""
        if self.history_length == 0:
            empty_images = {
                key: np.empty((0, *np.asarray(value).shape), dtype=np.asarray(value).dtype)
                for key, value in current_images.items()
            }
            empty_masks = {key: np.empty((0,), dtype=np.bool_) for key in current_images}
            state_history = None
            if include_state and current_state is not None:
                state_history = np.empty((0, *current_state.shape), dtype=current_state.dtype)
            return empty_images if include_images else {}, empty_masks if include_images else {}, state_history

        image_history: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.ndarray] = {}
        if include_images:
            for key, current in current_images.items():
                values = list(self._frames[key])
                valid_count = len(values)
                padding_value = values[0] if values else current
                values = [padding_value] * (self.history_length - valid_count) + values
                image_history[key] = np.stack(values)
                image_masks[key] = np.asarray(
                    [False] * (self.history_length - valid_count) + [True] * valid_count,
                    dtype=np.bool_,
                )

        state_history = None
        if include_state and current_state is not None:
            values = list(self._states)
            padding_value = values[0] if values else current_state
            values = [padding_value] * (self.history_length - len(values)) + values
            state_history = np.stack(values)
        return image_history, image_masks, state_history

    def append(self, images: Mapping[str, np.ndarray], state: np.ndarray | None) -> None:
        """复制 observation, 避免环境下一步复用数组时污染历史。"""
        if self.history_length == 0:
            return
        for key, image in images.items():
            self._frames[key].append(np.array(image, copy=True))
        if state is not None:
            self._states.append(np.array(state, copy=True))

    @staticmethod
    def _canonicalize_key_map(image_key_map: Mapping[str, str]) -> dict[str, str]:
        result = {}
        for first, second in image_key_map.items():
            if first in _CAMERA_ALIASES:
                result[first] = second
            elif second in _CAMERA_ALIASES:
                result[second] = first
            else:
                raise ValueError("image_key_map entries must include a canonical MEM camera name")
        return result

    @staticmethod
    def _find_value(observation: Mapping[str, Any], keys: tuple[str | None, ...]) -> Any | None:
        for key in keys:
            if not key:
                continue
            if key in observation:
                return observation[key]
            for container_name in ("image", "images"):
                container = observation.get(container_name)
                if isinstance(container, Mapping) and key in container:
                    return container[key]
            # 自定义环境常用嵌套 observation; 直接键优先, 再解析 slash/dot 路径。
            for separator in ("/", "."):
                current: Any = observation
                for part in key.split(separator):
                    if not isinstance(current, Mapping) or part not in current:
                        break
                    current = current[part]
                else:
                    return current
        return None


class MEMPolicy(Policy):
    """在不改变 BasePolicy.infer 接口的前提下管理 MEM 推理状态。"""

    def __init__(
        self,
        *args: Any,
        num_video_frames: int = 6,
        use_video_memory: bool = False,
        use_state_history: bool = False,
        use_language_memory: bool = False,
        max_memory_tokens: int = 256,
        image_key_map: Mapping[str, str] | None = None,
        state_key: str = "observation.state",
        high_level_policy: HighLevelPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._use_video_memory = use_video_memory
        self._use_state_history = use_state_history
        self._use_language_memory = use_language_memory
        self._max_memory_tokens = max_memory_tokens
        self._history = FrameHistoryBuffer(num_video_frames, image_key_map, state_key)
        self._high_level_policy = high_level_policy
        self._pending_result: tuple[bool, Mapping[str, Any] | None] | None = None
        self._memory_tokens = np.zeros(max_memory_tokens, dtype=np.int32)
        self._memory_mask = np.zeros(max_memory_tokens, dtype=np.bool_)

    def reset_episode(self, task_goal: str = "") -> None:
        """开始新 episode, 并清空所有短期和长期状态。"""
        self._history.reset()
        self._pending_result = None
        self._memory_tokens.fill(0)
        self._memory_mask.fill(0)
        if self._high_level_policy is not None:
            self._high_level_policy.reset(task_goal)

    @override
    def reset(self) -> None:
        """兼容 runtime 的 BasePolicy 生命周期。"""
        self.reset_episode()

    def notify_subtask_result(
        self,
        success: bool,  # noqa: FBT001
        observation: Mapping[str, Any] | None = None,
    ) -> None:
        """缓存完成信号, 高层策略会在下一次低层推理前处理。"""
        self._pending_result = (bool(success), observation)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        current_images = self._history.extract_images(obs)
        current_state = self._history.extract_state(obs)
        image_history, image_masks, state_history = self._history.build(
            current_images,
            current_state,
            include_images=self._use_video_memory,
            include_state=self._use_state_history,
        )
        augmented = dict(obs)
        if self._use_video_memory:
            augmented["image_history"] = image_history
            augmented["image_history_masks"] = image_masks
        if self._use_state_history and state_history is not None:
            augmented["state_history"] = state_history

        self._update_high_level_before_inference(obs, current_images)
        if self._high_level_policy is not None and self._high_level_policy.current_subtask:
            augmented["prompt"] = self._high_level_policy.current_subtask
        if self._use_language_memory and self._high_level_policy is not None:
            self._refresh_memory_tokens()
            augmented["tokenized_memory"] = self._memory_tokens.copy()
            augmented["tokenized_memory_mask"] = self._memory_mask.copy()

        result = super().infer(augmented, noise=noise)
        # 低层推理失败时不能推进历史, 否则下一次输入对应的是未执行动作。
        self._history.append(current_images, current_state)
        return result

    def _update_high_level_before_inference(
        self,
        observation: Mapping[str, Any],
        current_images: Mapping[str, np.ndarray],
    ) -> None:
        if self._high_level_policy is None:
            return
        pending = self._pending_result
        if not self._high_level_policy.should_update(subtask_completed=pending is not None):
            return
        success, signal_observation = pending if pending is not None else (True, None)
        source = signal_observation or observation
        images = self._history.extract_images(source) if signal_observation is not None else current_images
        base_image = images.get("base_0_rgb")
        if base_image is None:
            logger.warning("MEM high-level update skipped because base camera is missing")
            return
        try:
            self._high_level_policy.update(self._to_hwc_uint8(base_image), subtask_success=success)
        except Exception:
            # 自定义高层实现也必须隔离失败, 不能中断低层动作推理。
            logger.warning("MEM high-level callback failed; keeping previous state", exc_info=True)
        finally:
            self._pending_result = None

    def _refresh_memory_tokens(self) -> None:
        if self._high_level_policy is None:
            return
        try:
            tokens, mask = self._high_level_policy.tokenize_memory()
            tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)
            mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
            if tokens.shape != (self._max_memory_tokens,) or mask.shape != tokens.shape:
                raise ValueError(
                    f"memory tokenizer must return ({self._max_memory_tokens},), got {tokens.shape} and {mask.shape}"
                )
            self._memory_tokens, self._memory_mask = tokens, mask
        except Exception:
            logger.warning("MEM tokenization failed; reusing previous memory tokens", exc_info=True)

    @staticmethod
    def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
        value = np.asarray(image)
        if value.ndim != 3:
            raise ValueError(f"base camera must be rank 3, got {value.shape}")
        if value.shape[0] == 3 and value.shape[-1] != 3:
            value = np.moveaxis(value, 0, -1)
        if value.shape[-1] != 3:
            raise ValueError(f"base camera must have 3 channels, got {value.shape}")
        if np.issubdtype(value.dtype, np.floating):
            value = np.nan_to_num(value, nan=0.0, posinf=255.0, neginf=0.0)
            if value.size and value.min() < 0:
                value = (value + 1.0) / 2.0 * 255.0
            elif not value.size or value.max() <= 1.0:
                value = value * 255.0
            value = np.clip(value, 0, 255)
        return value.astype(np.uint8, copy=False)
