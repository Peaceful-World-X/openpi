"""LeRobot 多帧历史采样适配器, 仅用于 JAX Pi0/Pi05 MEM。"""

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np


class MEMLeRobotDataset:
    """包装随机访问 LeRobot dataset, 补充历史帧、状态和语言记忆。"""

    def __init__(
        self,
        dataset,
        *,
        num_frames: int = 6,
        frame_stride_sec: float = 1.0,
        image_key_map: Mapping[str, str] | None = None,
        state_key: str = "observation.state",
        memory_label_path: str | Path | None = None,
        include_video_history: bool = True,
        include_state_history: bool = True,
        include_language_memory: bool = True,
    ) -> None:
        if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames < 1:
            raise ValueError("num_frames must be positive")
        if not np.isfinite(frame_stride_sec) or frame_stride_sec <= 0:
            raise ValueError("frame_stride_sec must be positive")
        self.dataset = dataset
        self.num_frames = num_frames
        self.state_key = state_key
        self.include_video_history = include_video_history
        self.include_state_history = include_state_history
        self.include_language_memory = include_language_memory
        meta = getattr(dataset, "meta", None)
        meta_fps = meta.get("fps") if isinstance(meta, Mapping) else getattr(meta, "fps", None)
        fps_value = meta_fps if meta_fps is not None else getattr(dataset, "fps", None)
        if fps_value is None:
            raise ValueError("MEM LeRobot dataset metadata must provide fps")
        self.fps = float(fps_value)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("dataset fps must be positive")
        self.stride = max(1, round(frame_stride_sec * self.fps))
        self.image_key_map = dict(image_key_map or {})
        self._labels = self._load_labels(memory_label_path)
        # LeRobot 不同版本把 episode_data_index 放在 dataset 或 meta 上。
        self._episode_ranges = self._read_episode_ranges(dataset, meta)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        episode_value = sample.get("episode_index", sample.get("episode_id"))
        frame_value = sample.get("frame_index", sample.get("timestep"))
        if episode_value is None or frame_value is None:
            raise ValueError("MEM LeRobot samples must contain episode_index/episode_id and frame_index/timestep")
        episode_id = self._python_scalar(episode_value)
        frame_index = self._scalar(frame_value)
        episode_index = self._optional_int(episode_id)
        start, end = self._episode_bounds(episode_index, frame_index, int(index))
        history: dict[str, list[np.ndarray]] = {}
        masks: dict[str, list[bool]] = {}
        state_history: list[np.ndarray] = []
        image_keys = self._resolve_image_keys(sample) if self.include_video_history else {}
        if self.include_video_history and not image_keys:
            raise ValueError(
                "MEM video history could not find any supported image field; set MEMDataConfig.image_key_map"
            )
        current_state = self._extract_state(sample) if self.include_state_history else None
        for offset in range(self.num_frames - 1, 0, -1):
            wanted = frame_index - offset * self.stride
            valid = wanted >= 0 and start <= start + wanted < end
            clipped = max(0, min(end - start - 1, wanted))
            hist_sample = self.dataset[start + clipped]
            for canonical, raw_key in image_keys.items():
                if raw_key in hist_sample or raw_key in sample:
                    source = hist_sample.get(raw_key, sample.get(raw_key))
                    history.setdefault(canonical, []).append(self._to_numpy(source))
                    masks.setdefault(canonical, []).append(bool(valid and raw_key in hist_sample))
            if self.include_state_history:
                historical_state = self._extract_state(hist_sample)
                if historical_state is not None:
                    state_history.append(historical_state)
                elif current_state is not None:
                    state_history.append(current_state)

        result = dict(sample)
        if self.include_video_history:
            if self.num_frames > 1:
                result["image_history"] = {key: np.stack(values) for key, values in history.items()}
                result["image_history_masks"] = {
                    key: np.asarray(values, dtype=np.bool_) for key, values in masks.items()
                }
            else:
                result["image_history"] = {
                    canonical: np.empty(
                        (0, *self._to_numpy(sample[raw_key]).shape), dtype=self._to_numpy(sample[raw_key]).dtype
                    )
                    for canonical, raw_key in image_keys.items()
                }
                result["image_history_masks"] = {canonical: np.empty((0,), dtype=np.bool_) for canonical in image_keys}
        if self.include_state_history and current_state is not None:
            if state_history:
                result["state_history"] = np.stack(state_history)
            else:
                result["state_history"] = np.empty((0, *current_state.shape), dtype=current_state.dtype)
        if self.include_language_memory:
            result["language_memory"] = self._labels.get((str(episode_id), frame_index), "")
        return result

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _scalar(value: Any) -> int:
        if hasattr(value, "item"):
            value = value.item()
        return int(value)

    @staticmethod
    def _python_scalar(value: Any) -> Any:
        """保留字符串 episode id, 同时解包 NumPy/PyTorch 标量。"""
        if hasattr(value, "item"):
            value = value.item()
        return value

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _resolve_image_keys(self, sample: Mapping[str, Any]) -> dict[str, str]:
        if self.image_key_map:
            resolved = {}
            canonical_names = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
            for first, second in self.image_key_map.items():
                if first in canonical_names:
                    canonical, raw_key = first, second
                elif second in canonical_names:
                    canonical, raw_key = second, first
                else:
                    raise ValueError(
                        "MEMDataConfig.image_key_map entries must map a raw field to a canonical camera name"
                    )
                resolved[canonical] = raw_key
            return resolved
        candidates = {
            "base_0_rgb": (
                "base_0_rgb",
                "observation.images.top",
                "observation.images.cam_high",
                "observation.images.base",
                "observation.images.exterior_image_1_left",
                "observation/exterior_image_1_left",
                "observation/image",
                "exterior_image_1_left",
                "image",
            ),
            "left_wrist_0_rgb": (
                "left_wrist_0_rgb",
                "observation.images.left_wrist",
                "observation.images.cam_left_wrist",
                "observation.images.wrist_image_left",
                "observation/wrist_image_left",
                "observation/wrist_image",
                "wrist_image_left",
                "wrist_image",
            ),
            "right_wrist_0_rgb": (
                "right_wrist_0_rgb",
                "observation.images.right_wrist",
                "observation.images.cam_right_wrist",
                "observation.images.wrist_image_right",
                "observation/wrist_image_right",
                "wrist_image_right",
            ),
        }
        return {
            canonical: next(key for key in keys if key in sample)
            for canonical, keys in candidates.items()
            if any(key in sample for key in keys)
        }

    def _episode_bounds(self, episode_id: int | None, frame_index: int, absolute_index: int) -> tuple[int, int]:
        if episode_id is not None and episode_id in self._episode_ranges:
            return self._episode_ranges[episode_id]
        # 无 metadata 时从 episode 内 frame_index 反推起点, 防止跨 episode。
        return max(0, absolute_index - frame_index), len(self.dataset)

    def _extract_state(self, sample: Mapping[str, Any]) -> np.ndarray | None:
        """优先读取配置字段, 并兼容当前仓库的 ALOHA/Libero/DROID 原始 schema。"""
        state_candidates = (self.state_key, "observation.state", "observation/state", "state")
        for key in state_candidates:
            if key in sample:
                return self._to_numpy(sample[key])

        # DROID LeRobot 在 repack 前把 joint 和 gripper 分开存储, 历史必须按当前 DroidInputs 顺序拼接。
        for joint_key, gripper_key in (
            ("joint_position", "gripper_position"),
            ("observation.joint_position", "observation.gripper_position"),
            ("observation/joint_position", "observation/gripper_position"),
        ):
            if joint_key in sample and gripper_key in sample:
                return np.concatenate(
                    [self._to_numpy(sample[joint_key]).reshape(-1), self._to_numpy(sample[gripper_key]).reshape(-1)]
                )
        return None

    @staticmethod
    def _read_episode_ranges(dataset, meta) -> dict[int, tuple[int, int]]:
        for owner in (dataset, getattr(dataset, "_dataset", None), meta):
            if owner is None:
                continue
            index = (
                owner.get("episode_data_index")
                if isinstance(owner, Mapping)
                else getattr(owner, "episode_data_index", None)
            )
            if index is None:
                continue
            try:
                starts = np.asarray(index["from"]).reshape(-1)
                ends = np.asarray(index["to"]).reshape(-1)
                return {
                    episode: (int(start), int(end))
                    for episode, (start, end) in enumerate(zip(starts, ends, strict=True))
                }
            except (KeyError, TypeError, ValueError):
                continue
        return {}

    @staticmethod
    def _load_labels(path: str | Path | None) -> dict[tuple[str, int], str]:
        if path is None:
            return {}
        labels = {}
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                episode = item.get("episode_id", item.get("episode_index"))
                timestep = item.get("frame_index", item.get("timestep"))
                if episode is None or timestep is None:
                    continue
                # 训练输入是 m_t, 优先读取 memory_before 防止未来信息泄漏。
                memory = item.get("memory_before", item.get("memory", item.get("memory_after", "")))
                labels[(str(episode), int(timestep))] = str(memory or "")
        return labels
