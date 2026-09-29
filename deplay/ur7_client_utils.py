"""Small UR7 deployment helpers kept separate from the control loop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

from .ur7.se3 import relative_to_absolute
from .ur7.se3 import rot6d_to_matrix

logger = logging.getLogger(__name__)


def rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the geodesic distance between two canonical rot6d rotations."""
    relative = rot6d_to_matrix(first).T @ rot6d_to_matrix(second)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def as_rgb_uint8(
    image: Any,
    *,
    name: str,
    size: int = 224,
    horizontal_flip: bool = False,
) -> np.ndarray:
    """Convert HWC/CHW camera data to an OpenPI-compatible RGB image."""
    value = np.asarray(image)
    if value.ndim != 3:
        raise ValueError(f"{name} image must be rank 3, got {value.shape}")
    if value.shape[0] in (1, 3, 4) and value.shape[-1] not in (1, 3, 4):
        value = np.moveaxis(value, 0, -1)
    if value.shape[-1] != 3:
        raise ValueError(f"{name} image must have 3 channels, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} image contains NaN or Inf")
    if value.dtype != np.uint8:
        if np.issubdtype(value.dtype, np.floating) and np.nanmax(value, initial=0.0) <= 1.0:
            value = value * 255.0
        value = np.clip(value, 0, 255).astype(np.uint8)
    if horizontal_flip:
        value = value[:, ::-1]
    value = np.ascontiguousarray(value)
    if value.shape[:2] == (size, size):
        return value
    try:
        from PIL import Image

        # Keep the camera aspect ratio. The Pi05 preprocessing uses the same
        # resize-with-padding contract; stretching 640x480 to a square changes
        # object geometry before the image reaches the policy.
        height, width = value.shape[:2]
        scale = min(size / width, size / height)
        resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = Image.fromarray(value, mode="RGB").resize(resized_size, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        offset = ((size - resized_size[0]) // 2, (size - resized_size[1]) // 2)
        canvas.paste(resized, offset)
        return np.asarray(canvas, dtype=np.uint8)
    except ImportError:
        rows = np.linspace(0, value.shape[0] - 1, size).astype(np.int64)
        cols = np.linspace(0, value.shape[1] - 1, size).astype(np.int64)
        return np.ascontiguousarray(value[rows][:, cols])


# 双路 RealSense RGB 相机适配器。
class RealSensePair:
    """Read one RGB stream from each of two RealSense devices."""

    def __init__(
        self,
        *,
        base_serial: str | None = None,
        wrist_serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError("pyrealsense2 is required for RealSense deployment") from error
        self._rs = rs
        self._pipelines = []
        self._base = self._start(base_serial, width, height, fps)
        self._wrist = self._start(wrist_serial, width, height, fps)

    def _start(self, serial: str | None, width: int, height: int, fps: int):
        config = self._rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(self._rs.stream.color, width, height, self._rs.format.rgb8, fps)
        pipeline = self._rs.pipeline()
        pipeline.start(config)
        self._pipelines.append(pipeline)
        return pipeline

    @staticmethod
    def _read(pipeline):
        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        if not frame:
            raise RuntimeError("RealSense color frame is unavailable")
        return np.asanyarray(frame.get_data())

    def read(self) -> dict[str, np.ndarray]:
        return {"base": self._read(self._base), "wrist": self._read(self._wrist)}

    def close(self) -> None:
        for pipeline in self._pipelines:
            try:
                pipeline.stop()
            except Exception:
                logger.debug("failed to stop RealSense pipeline", exc_info=True)


# 无硬件 smoke 测试使用的假机器人。
class FakeUR7Robot:
    """No-hardware robot for local client smoke tests."""

    execute_enabled = False
    # Reset tests may opt into state-only simulation without enabling policy
    # action writes, preserving the dry-run contract of ``--smoke``.
    simulate_motion = True

    def __init__(self, state: np.ndarray | None = None):
        self._state = np.asarray(
            state if state is not None else np.r_[0.40, -0.10, 0.30, 1, 0, 0, 0, 1, 0, 1, np.zeros(12)],
            dtype=np.float64,
        )
        if self._state.shape != (22,):
            raise ValueError("FakeUR7Robot state must be 22D")
        self.actions: list[np.ndarray] = []

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def send_absolute_action(self, action: np.ndarray, *, command_gripper: bool = True) -> None:
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (10,) or not np.isfinite(value).all():
            raise ValueError("fake robot received invalid action")
        self.actions.append(value.copy())
        self._state[:10] = value

    def send_reset_action(self, action: np.ndarray) -> None:
        """Mirror the real reset API for local state-machine tests."""
        self.send_absolute_action(action, command_gripper=False)

    def open_gripper(self) -> None:
        self._state[9] = 1.0

    def get_eef_pose(self) -> np.ndarray:
        return self._state[:9].copy()

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


# 机械臂动作的单步平移、旋转和工作空间安全限制。
@dataclass(frozen=True)
class SafetyLimits:
    max_translation_step_m: float = 0.05
    max_rotation_step_rad: float = 0.35
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None


# 将服务端返回的动作统一转换为绝对 EEF action chunk。
class AbsoluteChunkPolicy:
    """Expose absolute chunks to the scheduler, decoding relative chunks first."""

    def __init__(self, policy: Any, action_space: str):
        self._policy = policy
        self._action_space = action_space

    def infer_chunk(self, observation: Mapping[str, Any]) -> np.ndarray:
        chunk = np.asarray(self._policy.infer_chunk(observation), dtype=np.float64)
        if self._action_space == "relative_se3":
            chunk = relative_to_absolute(chunk, np.asarray(observation["state"][:9]))
        return chunk

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    def close(self) -> None:
        close = getattr(self._policy, "close", None)
        if callable(close):
            close()
