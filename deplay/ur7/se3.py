"""Small NumPy-only SE(3) helpers used by the UR7 deployment client.

The OpenPI UR7 contract stores rotations as the first two columns of a
rotation matrix (the usual 6D representation).  Keeping these helpers local
to ``deplay`` makes the robot client independent from the training checkout.
"""

from __future__ import annotations

import numpy as np


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert a final-dimension 6D rotation representation to matrices."""
    value = np.asarray(rot6d, dtype=np.float64)
    if value.shape[-1] != 6:
        raise ValueError(f"rot6d must have trailing dimension 6, got {value.shape}")
    first = value[..., :3]
    second = value[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    first = first / np.maximum(first_norm, 1e-12)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    second = second / np.maximum(second_norm, 1e-12)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Return the first two columns of a rotation matrix."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in (3, 3), got {value.shape}")
    return np.concatenate((value[..., :, 0], value[..., :, 1]), axis=-1)


def _rotation_log(matrix: np.ndarray) -> np.ndarray:
    """Return the axis-angle vector of one rotation matrix."""
    trace = float(np.trace(matrix))
    cosine = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-7:
        return 0.5 * np.asarray(
            (matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]), dtype=np.float64
        )
    if np.pi - angle < 1e-5:
        # At pi the skew part is almost zero.  Recover a stable axis from the
        # largest diagonal entry and fix its sign deterministically.
        diagonal = np.diag(matrix)
        axis = np.zeros(3, dtype=np.float64)
        index = int(np.argmax(diagonal))
        axis[index] = np.sqrt(max((diagonal[index] + 1.0) * 0.5, 0.0))
        if axis[index] < 1e-8:
            axis[index] = 1.0
        other = (index + 1) % 3
        third = (index + 2) % 3
        axis[other] = (matrix[other, index] + matrix[index, other]) / (4.0 * axis[index])
        axis[third] = (matrix[third, index] + matrix[index, third]) / (4.0 * axis[index])
        axis /= max(np.linalg.norm(axis), 1e-12)
        return axis * angle
    sine = float(np.sin(angle))
    skew = np.asarray(
        (matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]), dtype=np.float64
    )
    return skew * (angle / (2.0 * sine))


def _rotation_exp(rotvec: np.ndarray) -> np.ndarray:
    """Return a rotation matrix from an axis-angle vector."""
    value = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(value))
    if angle < 1e-8:
        x, y, z = value
        skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
        return np.eye(3) + skew
    axis = value / angle
    x, y, z = axis
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def interpolate_absolute_action(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    """Interpolate two absolute EEF actions in translation and SE(3).

    The gripper is treated as an absolute scalar and linearly interpolated.
    ``weight=0`` returns ``first`` and ``weight=1`` returns ``second``.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != (10,) or b.shape != (10,):
        raise ValueError(f"actions must both have shape (10,), got {a.shape} and {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("cannot interpolate non-finite actions")
    t = float(np.clip(weight, 0.0, 1.0))
    first_rotation = rot6d_to_matrix(a[3:9])
    second_rotation = rot6d_to_matrix(b[3:9])
    delta = first_rotation.T @ second_rotation
    rotation = first_rotation @ _rotation_exp(t * _rotation_log(delta))
    result = np.empty(10, dtype=np.float64)
    result[:3] = (1.0 - t) * a[:3] + t * b[:3]
    result[3:9] = matrix_to_rot6d(rotation)
    result[9] = (1.0 - t) * a[9] + t * b[9]
    return result


def fuse_chunk_boundary(old_tail: np.ndarray, new_chunk: np.ndarray, overlap_steps: int) -> np.ndarray:
    """Blend the beginning of a new absolute chunk with the executed old tail.

    The old tail is already executed; it is used as a continuity anchor.  No
    actions are duplicated, so the returned array has exactly the new chunk's
    length.  A zero overlap returns a copy of ``new_chunk``.
    """
    old = np.asarray(old_tail, dtype=np.float64)
    new = np.asarray(new_chunk, dtype=np.float64)
    if old.ndim != 2 or new.ndim != 2 or old.shape[1:] != (10,) or new.shape[1:] != (10,):
        raise ValueError(f"chunks must have shape (N, 10), got {old.shape} and {new.shape}")
    overlap = int(overlap_steps)
    if overlap < 0:
        raise ValueError("overlap_steps must be non-negative")
    overlap = min(overlap, len(old), len(new))
    result = new.copy()
    if overlap == 0:
        return result
    for index in range(overlap):
        # Give the fresh prediction progressively more authority at the
        # boundary while retaining a short continuity ramp.
        result[index] = interpolate_absolute_action(old[-overlap + index], new[index], (index + 1) / (overlap + 1))
    return result


def relative_to_absolute(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Decode body-frame relative EEF actions against one current state."""
    value = np.asarray(actions, dtype=np.float64)
    current = np.asarray(state, dtype=np.float64)
    if value.ndim < 2 or value.shape[-1] != 10:
        raise ValueError(f"actions must have shape (N, 10), got {value.shape}")
    if current.shape[-1] < 9:
        raise ValueError(f"state must have at least 9 values, got {current.shape}")
    state_rotation = rot6d_to_matrix(current[..., 3:9])
    relative_rotation = rot6d_to_matrix(value[..., 3:9])
    translation = current[..., :3] + np.einsum("...ij,...j->...i", state_rotation, value[..., :3])
    rotation = np.einsum("...ij,...jk->...ik", state_rotation, relative_rotation)
    result = np.concatenate((translation, matrix_to_rot6d(rotation), value[..., 9:10]), axis=-1)
    return result.astype(np.float64, copy=False)
