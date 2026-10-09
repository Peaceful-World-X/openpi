"""Fixed joint branch for an absolute TCP reset target."""
from dataclasses import dataclass, replace
from collections.abc import Mapping
from typing import Any

import numpy as np

from .joint_reset import JointResetConfig


@dataclass(frozen=True)
class CartesianResetConfig:
    joint_reference_rad: tuple[float, ...]
    joint_speed_rad_s: float = 0.15
    joint_acceleration_rad_s2: float = 0.1
    max_joint_delta_rad: tuple[float, ...] = (2.0, 2.0, 2.0, 3.0, 3.0, 3.2)
    branch_tolerance_rad: float = 0.5
    joint_position_tolerance_rad: float = 0.02

    def __post_init__(self) -> None:
        # Reuse the exact, unwrapped joint target validation and motion limits.
        self._joint_config(self.joint_reference_rad)
        if not np.isfinite(self.branch_tolerance_rad) or not 0 < self.branch_tolerance_rad < np.pi:
            raise ValueError("reset.branch_tolerance_rad must be positive and below pi")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CartesianResetConfig":
        return cls(**{k: mapping[k] for k in cls.__dataclass_fields__ if k in mapping})

    def _joint_config(self, target: Any) -> JointResetConfig:
        return JointResetConfig(
            joints_rad=tuple(target),
            speed_rad_s=self.joint_speed_rad_s,
            acceleration_rad_s2=self.joint_acceleration_rad_s2,
            max_delta_rad=self.max_joint_delta_rad,
            position_tolerance_rad=self.joint_position_tolerance_rad,
        )

    def target_config(self, solution: Any) -> JointResetConfig:
        """Choose turns relative to the fixed reference, never the current q."""
        solution = np.asarray(solution, dtype=np.float64)
        reference = np.asarray(self.joint_reference_rad, dtype=np.float64)
        if solution.shape != (6,) or not np.isfinite(solution).all():
            raise RuntimeError("reset IK returned invalid joints")
        target = solution + (2 * np.pi) * np.round((reference - solution) / (2 * np.pi))
        if np.max(np.abs(target - reference)) > self.branch_tolerance_rad:
            raise RuntimeError("reset IK left the configured joint branch; review reset.joint_reference_rad")
        return self._joint_config(target)

    def target_candidates(self, solution: Any, current: Any) -> list[JointResetConfig]:
        """Keep the IK branch, choosing the closest equivalent wrist-3 turn."""
        base = self.target_config(solution)
        current = np.asarray(current, dtype=np.float64)
        if current.shape != (6,) or not np.isfinite(current).all():
            raise RuntimeError("invalid current joint positions before reset")
        base_target = np.asarray(base.joints_rad, dtype=np.float64)
        nearest_turn = int(np.round((current[5] - base_target[5]) / (2 * np.pi)))
        candidates = []
        errors = []
        for turn in (nearest_turn, nearest_turn - 1, nearest_turn + 1):
            target = base_target.copy()
            target[5] += turn * (2 * np.pi)
            candidate = replace(base, joints_rad=tuple(target))
            try:
                candidate.target_from(current)
            except RuntimeError as error:
                errors.append(error)
            else:
                candidates.append(candidate)
        if not candidates:
            raise errors[0]
        return sorted(candidates, key=lambda c: (
            abs(c.joints_rad[5] - current[5]),
            abs(c.joints_rad[5] - base_target[5]),
        ))
