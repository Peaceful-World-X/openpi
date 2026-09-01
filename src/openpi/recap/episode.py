from __future__ import annotations

from collections import Counter
import dataclasses
import itertools
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class ReCAPFrame:
    """单帧 transition; action 字段保留以兼容旧采集文件。"""

    t: int
    observation: dict[str, Any]
    action: np.ndarray
    reward: float = 0.0
    policy_action: np.ndarray | None = None
    human_action: np.ndarray | None = None
    executed_action: np.ndarray | None = None
    success: bool | None = None
    is_human_intervention: bool = False
    # 保存终止原因, value 标注时才能区分 terminal 与 timeout。
    terminated: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.t, bool | np.bool_) or not isinstance(self.t, int | np.integer) or self.t < 0:
            raise ValueError("frame t must be a non-negative integer")
        object.__setattr__(self, "t", int(self.t))
        if not isinstance(self.observation, dict):
            raise TypeError("frame observation must be a dictionary")
        for name in ("terminated", "truncated", "is_human_intervention"):
            value = getattr(self, name)
            if not isinstance(value, bool | np.bool_):
                raise TypeError(f"frame {name} must be a bool")
            object.__setattr__(self, name, bool(value))
        if self.success is not None:
            if not isinstance(self.success, bool | np.bool_):
                raise TypeError("frame success must be a bool or None")
            object.__setattr__(self, "success", bool(self.success))
        action = np.asarray(self.action, dtype=np.float32)
        if action.ndim != 1 or not np.all(np.isfinite(action)) or not np.isfinite(self.reward):
            raise ValueError("frame action must be a finite one-dimensional array")
        object.__setattr__(self, "action", action)
        for name in ("policy_action", "human_action", "executed_action"):
            optional_action = getattr(self, name)
            if optional_action is None:
                continue
            optional_action = np.asarray(optional_action, dtype=np.float32)
            if optional_action.shape != action.shape or not np.all(np.isfinite(optional_action)):
                # 三类动作必须同维, 避免 JSON 审计字段与真实执行动作悄悄错位。
                raise ValueError(f"frame {name} must be finite and match action shape {action.shape}")
            object.__setattr__(self, name, optional_action)


@dataclasses.dataclass(frozen=True)
class ReCAPOfflineEpisode:
    episode_id: str
    task: str
    success: bool
    frames: list[ReCAPFrame]
    timeout: bool = False
    max_episode_length: int | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id or Path(self.episode_id).name != self.episode_id:
            raise ValueError("episode_id must be a non-empty file-safe name")
        if not isinstance(self.task, str):
            raise TypeError("episode task must be a string")
        for name in ("success", "timeout"):
            value = getattr(self, name)
            if not isinstance(value, bool | np.bool_):
                # 奖励构造直接依赖这两个结果字段, 禁止字符串/任意整数发生 truthy 误判。
                raise TypeError(f"episode {name} must be a bool")
            object.__setattr__(self, name, bool(value))
        if not isinstance(self.frames, list) or not all(isinstance(frame, ReCAPFrame) for frame in self.frames):
            raise TypeError("episode frames must be a list of ReCAPFrame")
        if self.frames:
            action_shapes = {frame.action.shape for frame in self.frames}
            if len(action_shapes) != 1:
                raise ValueError(f"episode actions must keep one shape, got {sorted(action_shapes)}")
            timesteps = [frame.t for frame in self.frames]
            if any(current <= previous for previous, current in itertools.pairwise(timesteps)):
                raise ValueError("episode frame t values must be strictly increasing")
            boundaries = [
                index for index, frame in enumerate(self.frames) if frame.terminated or frame.truncated
            ]
            if boundaries and boundaries != [len(self.frames) - 1]:
                # transition 边界之后不存在同一 episode 的合法动作, 必须在训练前拒绝。
                raise ValueError("episode termination/truncation boundary must appear only on the final frame")
        if self.max_episode_length is not None and (
            isinstance(self.max_episode_length, bool | np.bool_)
            or not isinstance(self.max_episode_length, int | np.integer)
            or self.max_episode_length < max(1, len(self.frames))
        ):
            raise ValueError("max_episode_length must be positive and cover all recorded frames")
        if self.max_episode_length is not None:
            object.__setattr__(self, "max_episode_length", int(self.max_episode_length))
        if not isinstance(self.metadata, dict):
            raise TypeError("episode metadata must be a dictionary")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _coerce_bool(value: Any, *, name: str) -> bool:
    """读取 JSON/旧 parquet 导出的 0/1 标记, 拒绝含糊字符串。"""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a bool or 0/1 integer")


def _array_specs(value: Any) -> Any:
    """单独记录 ndarray dtype/shape, 避免 JSON list 丢失图像 uint8 语义。"""
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, dict):
        children = {str(key): _array_specs(item) for key, item in value.items()}
        return {key: item for key, item in children.items() if item is not None} or None
    if isinstance(value, tuple | list):
        children = [_array_specs(item) for item in value]
        return children if any(item is not None for item in children) else None
    return None


def _numeric_arrays(value: Any, *, image: bool = False) -> Any:
    """兼容旧 JSON; 规则嵌套数值 list 恢复为数组, 图像优先保留 uint8。"""
    if isinstance(value, list) and value:
        try:
            array = np.asarray(value)
        except ValueError:
            array = np.asarray(value, dtype=object)
        if array.dtype != np.dtype("O") and array.dtype.kind in "biuf":
            if array.dtype.kind == "b":
                return array.astype(np.bool_)
            if image and array.dtype.kind in "iu" and np.all((array >= 0) & (array <= 255)):
                return array.astype(np.uint8)
            return array.astype(np.float32)
    if isinstance(value, dict):
        return {
            key: _numeric_arrays(item, image=image or key in {"image", "images"})
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_numeric_arrays(item, image=image) for item in value]
    return value


def _restore_arrays(value: Any, specs: Any, *, image: bool = False) -> Any:
    """按新 schema 的 dtype/shape 精确恢复数组, 缺少规格时回退旧 schema。"""
    if isinstance(specs, dict) and set(specs) == {"dtype", "shape"}:
        array = np.asarray(value, dtype=np.dtype(specs["dtype"]))
        expected_shape = tuple(int(item) for item in specs["shape"])
        if array.shape != expected_shape:
            raise ValueError(f"observation array shape {array.shape} does not match recorded shape {expected_shape}")
        return array
    if isinstance(value, dict):
        spec_dict = specs if isinstance(specs, dict) else {}
        return {
            key: _restore_arrays(
                item,
                spec_dict.get(key),
                image=image or key in {"image", "images"},
            )
            for key, item in value.items()
        }
    if isinstance(value, list) and isinstance(specs, list) and len(value) == len(specs):
        return [_restore_arrays(item, spec, image=image) for item, spec in zip(value, specs, strict=True)]
    return _numeric_arrays(value, image=image)


def episode_to_dict(episode: ReCAPOfflineEpisode) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "episode_id": episode.episode_id,
        "task": episode.task,
        "success": bool(episode.success),
        "timeout": bool(episode.timeout),
        "max_episode_length": episode.max_episode_length,
        "frames": [
            {
                "t": frame.t,
                "observation": _jsonable(frame.observation),
                # JSON list 不携带 dtype; 独立规格让 uint8 图像和浮点状态可无损恢复。
                "observation_array_specs": _array_specs(frame.observation),
                "action": _jsonable(frame.action),
                "policy_action": _jsonable(frame.policy_action),
                "human_action": _jsonable(frame.human_action),
                "executed_action": _jsonable(frame.executed_action),
                "reward": float(frame.reward),
                "success": frame.success,
                "is_human_intervention": bool(frame.is_human_intervention),
                "terminated": bool(frame.terminated),
                "truncated": bool(frame.truncated),
            }
            for frame in episode.frames
        ],
        "metadata": _jsonable(episode.metadata),
    }


def episode_from_dict(data: dict[str, Any]) -> ReCAPOfflineEpisode:
    schema_version = data.get("schema_version", 1)
    if (
        isinstance(schema_version, bool | np.bool_)
        or not isinstance(schema_version, int | np.integer)
        or not 1 <= int(schema_version) <= 3
    ):
        # 缺省版本按 legacy v1 读取; 显式未知版本不能静默套用当前字段语义。
        raise ValueError(f"unsupported RECAP episode schema_version: {schema_version!r}")
    if "episode_id" not in data or "frames" not in data:
        raise ValueError("episode JSON requires episode_id and frames")
    frames = []
    for index, raw in enumerate(data["frames"]):
        # 旧 schema 只有 action/obs; 新 schema 优先使用 executed_action。
        action = raw.get("executed_action")
        if action is None:
            action = raw.get("action")
        if action is None:
            raise ValueError(f"frame {index} has no action")
        frames.append(
            ReCAPFrame(
                # 交给 ReCAPFrame 校验整数类型, 避免 bool/浮点/字符串被 int() 静默截断。
                t=raw.get("t", index),
                observation=_restore_arrays(
                    raw.get("observation", raw.get("obs", {})),
                    raw.get("observation_array_specs"),
                ),
                action=np.asarray(action, dtype=np.float32),
                reward=float(raw.get("reward", 0.0)),
                policy_action=None if raw.get("policy_action") is None else np.asarray(raw["policy_action"], dtype=np.float32),
                human_action=None if raw.get("human_action") is None else np.asarray(raw["human_action"], dtype=np.float32),
                executed_action=None if raw.get("executed_action") is None else np.asarray(raw["executed_action"], dtype=np.float32),
                success=(None if raw.get("success") is None else _coerce_bool(raw["success"], name="frame success")),
                is_human_intervention=_coerce_bool(
                    raw.get("is_human_intervention", False), name="frame is_human_intervention"
                ),
                terminated=_coerce_bool(raw.get("terminated", False), name="frame terminated"),
                truncated=_coerce_bool(raw.get("truncated", False), name="frame truncated"),
            )
        )
    return ReCAPOfflineEpisode(
        episode_id=str(data["episode_id"]),
        task=str(data.get("task", "")),
        success=_coerce_bool(data.get("success", False), name="episode success"),
        frames=frames,
        timeout=_coerce_bool(data.get("timeout", False), name="episode timeout"),
        max_episode_length=data.get("max_episode_length"),
        metadata=dict(data.get("metadata", {})),
    )


def load_episodes(path: str | os.PathLike[str]) -> list[ReCAPOfflineEpisode]:
    """读取单个 JSON、JSON 数组或目录中的多个 episode。"""
    source = Path(path)
    if source.is_dir():
        episodes: list[ReCAPOfflineEpisode] = []
        for item in sorted(source.glob("*.json")):
            episodes.extend(load_episodes(item))
        _validate_unique_episode_ids(episodes)
        return episodes
    data = json.loads(source.read_text(encoding="utf-8"))
    episodes = [episode_from_dict(item) for item in data] if isinstance(data, list) else [episode_from_dict(data)]
    _validate_unique_episode_ids(episodes)
    return episodes


def _validate_unique_episode_ids(episodes: list[ReCAPOfflineEpisode]) -> None:
    """所有读取/写入入口统一拒绝重复轨迹, 避免历史数据被重复训练。"""
    duplicates = sorted(
        episode_id
        for episode_id, count in Counter(episode.episode_id for episode in episodes).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"RECAP episode ids must be unique, duplicates: {duplicates}")


def save_episodes(episodes: list[ReCAPOfflineEpisode], path: str | os.PathLike[str]) -> None:
    """原子写入 episode 文件或目录, 防止在线采集被中断时产生半个 JSON。"""
    # 目录模式以 episode_id 命名; 统一拒绝重复值, 避免调用方在不同输出形式下语义不一致。
    _validate_unique_episode_ids(episodes)
    target = Path(path)
    if target.suffix.lower() == ".json":
        target.parent.mkdir(parents=True, exist_ok=True)
        # 目录中的每个文件写成单个 object; 显式传入多个 episode 时才写 JSON array。
        payload = (
            []
            if not episodes
            else _jsonable(episode_to_dict(episodes[0]))
            if len(episodes) == 1
            else [_jsonable(episode_to_dict(episode)) for episode in episodes]
        )
        with tempfile.NamedTemporaryFile("w", dir=target.parent, prefix=f".{target.name}.", delete=False, encoding="utf-8") as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_name = tmp.name
        os.replace(temp_name, target)
        return
    target.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        save_episodes([episode], target / f"{episode.episode_id}.json")


# 旧脚本名称兼容; 新 API 使用更短的 load_episodes/save_episodes。
load_recap_episodes = load_episodes
save_recap_episodes = save_episodes
