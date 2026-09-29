"""UR v2.1 EEF adapter for OpenPI/Pi05.

The reconstructed dataset uses the LeRobot v2.1 layout::

    state  = eef_xyz(3), rot6d(6), gripper(1), force(3), torque(3), velocity(6),
             joint_position(6), joint_velocity(6)
    action = absolute eef_xyz(3), rot6d(6), gripper(1)

OpenPI receives the canonical state layout::

    eef_xyz(3), rot6d(6), gripper(1), force(3), torque(3), velocity(6)

Actions are converted online to body-frame SE(3) relative actions.  The
gripper value remains absolute.  rot6d is the first two columns of a rotation
matrix, matching ``vla-sft`` and the UR dataset contract.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from io import BytesIO
from typing import Any

import numpy as np

from openpi import transforms

RAW_STATE_DIM = 34
CANONICAL_STATE_DIM = 22
ACTION_DIM = 10
MODEL_ACTION_DIM = 32


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert column-major 6D rotations to right-handed matrices."""
    value = np.asarray(rot6d, dtype=np.float64)
    if value.shape[-1] != 6:
        raise ValueError(f"rot6d must have trailing dimension 6, got {value.shape}")
    first = value[..., :3]
    second = value[..., 3:]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Return the first two columns in the canonical 6D layout."""
    value = np.asarray(matrix)
    return np.concatenate((value[..., :, 0], value[..., :, 1]), axis=-1)


def canonicalize_state(state: Any) -> np.ndarray:
    """Select the canonical 22D state from the reconstructed UR7 v2.1 state.

    The dataset stores ``[eef_xyz, rot6d, gripper, force, torque, velocity,
    joint_position, joint_velocity]`` (34 values).  Pi05 uses the first 22
    values; joint position/velocity are intentionally excluded because the
    action target is end-effector pose.
    """
    value = np.asarray(state, dtype=np.float32)
    if value.shape[-1] == CANONICAL_STATE_DIM:
        result = value
    elif value.shape[-1] == RAW_STATE_DIM:
        result = value[..., :CANONICAL_STATE_DIM]
    else:
        raise ValueError(
            f"UR7 state must have trailing dimension {CANONICAL_STATE_DIM} or {RAW_STATE_DIM}, "
            f"got {value.shape}"
        )
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("UR7 state contains non-finite values")
    _ = rot6d_to_matrix(result[..., 3:9])
    return result


def absolute_to_relative(actions: Any, state: Any) -> np.ndarray:
    """Convert absolute 10D EEF actions to body-frame relative SE(3)."""
    action_value = np.asarray(actions, dtype=np.float32)
    state_value = np.asarray(state, dtype=np.float32)
    if action_value.shape[-1] != ACTION_DIM:
        raise ValueError(f"UR action must have trailing dimension 10, got {action_value.shape}")
    if state_value.shape[-1] < 9:
        raise ValueError(f"canonical state must have at least 9 values, got {state_value.shape}")
    if not np.isfinite(action_value).all() or not np.isfinite(state_value).all():
        raise ValueError("UR state/action contains non-finite values")

    state_rotation = rot6d_to_matrix(state_value[..., 3:9])
    action_rotation = rot6d_to_matrix(action_value[..., 3:9])
    inverse_state_rotation = np.swapaxes(state_rotation, -1, -2)
    translation = np.einsum("...ij,...j->...i", inverse_state_rotation, action_value[..., :3] - state_value[..., :3])
    rotation = np.einsum("...ij,...jk->...ik", inverse_state_rotation, action_rotation)
    return np.concatenate((translation, matrix_to_rot6d(rotation), action_value[..., 9:10]), axis=-1).astype(np.float32)


def relative_to_absolute(actions: Any, state: Any) -> np.ndarray:
    """Decode body-frame relative actions against one canonical current state."""
    action_value = np.asarray(actions, dtype=np.float32)
    state_value = np.asarray(state, dtype=np.float32)
    if action_value.shape[-1] < ACTION_DIM or state_value.shape[-1] < 9:
        raise ValueError("relative_to_absolute expects 10D actions and a canonical state")
    state_rotation = rot6d_to_matrix(state_value[..., 3:9])
    relative_rotation = rot6d_to_matrix(action_value[..., 3:9])
    translation = state_value[..., :3] + np.einsum("...ij,...j->...i", state_rotation, action_value[..., :3])
    rotation = np.einsum("...ij,...jk->...ik", state_rotation, relative_rotation)
    return np.concatenate((translation, matrix_to_rot6d(rotation), action_value[..., 9:10]), axis=-1).astype(np.float32)


def _image_to_hwc_uint8(value: Any, *, name: str) -> np.ndarray:
    """Normalize LeRobot/PIL/torch image values to HWC uint8 RGB."""
    if isinstance(value, Mapping) and isinstance(value.get("bytes"), bytes):
        try:
            from PIL import Image

            with Image.open(BytesIO(value["bytes"])) as image:
                value = np.asarray(image.convert("RGB"))
        except Exception as exc:  # pragma: no cover - depends on optional image decoder
            raise ValueError(f"failed to decode {name} embedded image") from exc
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{name} image must be rank 3, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"{name} image must have 3 channels, got {image.shape}")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and float(np.nanmax(image, initial=0.0)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _pad_last_dim(value: np.ndarray, target_dim: int) -> np.ndarray:
    if value.shape[-1] > target_dim:
        raise ValueError(f"cannot pad dimension {value.shape[-1]} to smaller dimension {target_dim}")
    if value.shape[-1] == target_dim:
        return value
    return np.pad(value, [(0, 0)] * (value.ndim - 1) + [(0, target_dim - value.shape[-1])])


@dataclasses.dataclass(frozen=True)
class UR7Inputs(transforms.DataTransformFn):
    """Map reconstructed LeRobot v2.1 samples to Pi05 inputs."""

    action_dim: int = MODEL_ACTION_DIM

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        raw_state = np.asarray(data["state"], dtype=np.float32)
        state = canonicalize_state(raw_state)
        if "actions" in data:
            absolute_actions = np.asarray(data["actions"], dtype=np.float32)
            relative_actions = absolute_to_relative(absolute_actions, state[:9])
        else:
            relative_actions = None

        base_image = _image_to_hwc_uint8(data["image"], name="image")
        wrist_image = _image_to_hwc_uint8(data["wrist_image"], name="wrist_image")
        if base_image.shape != wrist_image.shape:
            raise ValueError(f"base and wrist images have different shapes: {base_image.shape} vs {wrist_image.shape}")

        result: dict[str, Any] = {
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.ones((), dtype=bool),
                "left_wrist_0_rgb": np.ones((), dtype=bool),
                "right_wrist_0_rgb": np.zeros((), dtype=bool),
            },
            "state": _pad_last_dim(state, self.action_dim),
        }
        if relative_actions is not None:
            result["actions"] = _pad_last_dim(relative_actions, self.action_dim)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        elif "task" in data:
            result["prompt"] = data["task"]
        return result


@dataclasses.dataclass(frozen=True)
class UR7Outputs(transforms.DataTransformFn):
    """Decode Pi05 relative actions back to canonical absolute EEF actions."""

    action_horizon: int = 50
    decode_relative: bool = True

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        actions = np.asarray(data["actions"], dtype=np.float32)
        was_single = actions.ndim == 1
        if was_single:
            actions = actions[None, :]
        actions = actions[..., :ACTION_DIM]
        if self.decode_relative:
            state = np.asarray(data["state"], dtype=np.float32)[..., :9]
            actions = relative_to_absolute(actions, state)
        actions = actions[..., : self.action_horizon, :]
        return {"actions": actions[0] if was_single else actions}
