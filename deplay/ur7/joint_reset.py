"""Validated, unwrapped joint targets for episode reset and home."""

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np


@dataclass(frozen=True)
class JointResetConfig:
    joints_rad: tuple[float, ...]
    speed_rad_s: float = 0.2
    acceleration_rad_s2: float = 0.4
    max_delta_rad: tuple[float, ...] = (2.0, 2.0, 2.0, 3.0, 3.0, 6.0)
    position_tolerance_rad: float = 0.02
    stationary_velocity_tolerance_rad_s: float = 0.02

    def __post_init__(self) -> None:
        target = np.asarray(self.joints_rad, dtype=np.float64)
        limits = np.asarray(self.max_delta_rad, dtype=np.float64)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("joint reset joints_rad must contain six finite absolute angles")
        if limits.shape != (6,) or not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("joint reset max_delta_rad must contain six positive finite limits")
        for name in (
            "speed_rad_s", "acceleration_rad_s2", "position_tolerance_rad",
            "stationary_velocity_tolerance_rad_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"joint reset {name} must be positive and finite")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "JointResetConfig":
        if "joints_rad" not in config:
            raise ValueError("reset/home requires joints_rad; EEF-only reset is no longer used")
        if "eef_pose" in config or "joint_reference_rad" in config:
            raise ValueError("remove reset/home.eef_pose and joint_reference_rad; absolute reset uses only joints_rad")
        fields = {key: config[key] for key in cls.__dataclass_fields__ if key in config}
        # Preserve the existing YAML controls for reset speed and travel limits.
        aliases = {
            "joint_speed_rad_s": "speed_rad_s",
            "joint_acceleration_rad_s2": "acceleration_rad_s2",
            "max_joint_delta_rad": "max_delta_rad",
            "joint_position_tolerance_rad": "position_tolerance_rad",
        }
        for source, destination in aliases.items():
            if source in config:
                if destination in fields and fields[destination] != config[source]:
                    raise ValueError(f"conflicting reset settings: {source} and {destination}")
                fields[destination] = config[source]
        return cls(**fields)

    def target_from(self, current: np.ndarray) -> np.ndarray:
        """Validate total motion; never wrap a target toward the current turn."""
        current = np.asarray(current, dtype=np.float64)
        if current.shape != (6,) or not np.isfinite(current).all():
            raise RuntimeError("invalid current joint positions before reset")
        target = np.asarray(self.joints_rad, dtype=np.float64).copy()
        delta = np.abs(target - current)
        limits = np.asarray(self.max_delta_rad)
        if np.any(delta > limits):
            axis = int(np.argmax(delta / limits))
            raise RuntimeError(
                f"joint reset q{axis + 1} delta={delta[axis]:.4f}rad exceeds "
                f"limit={limits[axis]:.4f}rad; target stays on its exact configured branch"
            )
        return target
