"""UR7 robot-side OpenPI client.

The policy server returns the output of the configured OpenPI data transform.
For the UR7 Pi05 config this is an absolute canonical action:
``xyz + rot6d + gripper``.  The client therefore validates and sends that
10D target directly to ``UR7Robot.send_absolute_action``.  The optional
``relative_se3`` mode is provided only for servers deliberately configured to
omit ``UR7Outputs``.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import replace
import copy
import queue
from datetime import datetime
from datetime import timezone
from io import BytesIO
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
import zlib

import numpy as np

from .async_chunk_client import AsyncChunkClient
from .async_chunk_client import ChunkStepInfo
from .async_chunk_client import validate_rtc_config
from .policy_client import WebSocketPolicyClient
from .ur7.robot import EXECUTE_CONFIRMATION
from .ur7.robot import ServoConfig
from .ur7.robot import StateConfig
from .ur7.robot import UR7Robot
from .ur7.joint_reset import JointResetConfig
from .ur7.cartesian_reset import CartesianResetConfig
from .ur7.se3 import interpolate_absolute_action
from .interactive import InteractiveUR7Controller
from .ur7_client_utils import AbsoluteChunkPolicy
from .ur7_client_utils import FakeUR7Robot
from .ur7_client_utils import RealSensePair
from .ur7_client_utils import SafetyLimits
from .ur7_client_utils import as_rgb_uint8
from .ur7_client_utils import rotation_distance

logger = logging.getLogger(__name__)

# Backward-compatible aliases for callers that imported the old private helpers.
_as_rgb_uint8 = as_rgb_uint8
_rotation_distance = rotation_distance
_AbsoluteChunkPolicy = AbsoluteChunkPolicy


# 负责采集 UR7 状态和相机观测、调度 action chunk，并执行安全检查后的动作。
class UR7DeploymentClient:
    """Build observations, consume async chunks, and command one UR7 target."""

    def __init__(
        self,
        robot: Any,
        policy: Any,
        image_reader: Any,
        *,
        task: str,
        control_hz: float = 10.0,
        reinfer: int = 30,
        blend: int = 16,
        inference_delay_steps: int = 4,
        rtc_enabled: bool = True,
        action_space: str = "absolute",
        safety: SafetyLimits | None = None,
        image_size: int = 224,
        flip_wrist_image: bool = False,
        debug_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if action_space not in {"absolute", "relative_se3"}:
            raise ValueError("action_space must be absolute or relative_se3")
        self.robot = robot
        self.image_reader = image_reader
        self._image_health_check = getattr(getattr(image_reader, "__self__", None), "check_health", None)
        self.task = str(task)
        self.period_s = 1.0 / float(control_hz)
        self.action_space = action_space
        self.safety = safety or SafetyLimits()
        self.image_size = int(image_size)
        self.flip_wrist_image = bool(flip_wrist_image)
        self.chunks = AsyncChunkClient(
            AbsoluteChunkPolicy(policy, action_space),
            reinfer=reinfer,
            blend=blend,
            inference_delay_steps=inference_delay_steps,
            control_period_s=self.period_s,
            rtc_enabled=rtc_enabled,
        )
        self._debug_callback = debug_callback
        self._debug_step_id = 0
        self._debug_generation = 0
        self._initialized = False
        self._last_command_started = None
        self._last_period_warning = 0.0
        self._motion_lock = threading.RLock()
        self._reset_thread: threading.Thread | None = None

    def _trace(self, event_type: str, **fields: Any) -> None:
        if self._debug_callback is not None:
            try:
                self._debug_callback({
                    "event_type": event_type, "monotonic_s": time.monotonic(),
                    "generation": self._debug_generation, **fields,
                })
            except Exception:
                logger.exception("could not record action trace")

    def observation(self) -> dict[str, Any]:
        images = self.image_reader()
        if not isinstance(images, Mapping) or "base" not in images or "wrist" not in images:
            raise ValueError("image_reader must return {'base': image, 'wrist': image}")
        state = np.asarray(self.robot.get_state(), dtype=np.float32)
        if state.ndim != 1 or state.shape[0] not in (22, 34) or not np.isfinite(state).all():
            raise ValueError(f"UR7 state must be finite 22D or 34D, got {state.shape}")
        return {
            # UR7Inputs consumes these raw deployment keys directly.  The
            # generic LeRobot names (observation.state/observation.images.*)
            # are dataset keys and are not accepted by the OpenPI UR7 policy
            # transform.
            "state": state,
            "image": as_rgb_uint8(images["base"], name="base", size=self.image_size),
            "wrist_image": as_rgb_uint8(
                images["wrist"],
                name="wrist",
                size=self.image_size,
                horizontal_flip=self.flip_wrist_image,
            ),
            "task": self.task,
        }

    def _decode_action(self, action: np.ndarray, observation: Mapping[str, Any]) -> np.ndarray:
        value = np.array(action, dtype=np.float64, copy=True)
        if value.shape != (10,) or not np.isfinite(value).all():
            raise ValueError("decoded action must be a finite 10D vector")
        # Pi05 predicts the gripper as an absolute opening fraction. Keep a
        # small model overshoot from aborting the whole arm step at Robotiq.
        raw_gripper = float(value[9])
        value[9] = np.clip(raw_gripper, 0.0, 1.0)
        if value[9] != raw_gripper:
            logger.warning("clipped policy gripper.open from %.5f to %.5f", raw_gripper, value[9])
        return value

    def _validate(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        if np.linalg.norm(action[:3] - state[:3]) > self.safety.max_translation_step_m:
            raise RuntimeError("policy translation step exceeds safety limit")
        if rotation_distance(action[3:9], state[3:9]) > self.safety.max_rotation_step_rad:
            raise RuntimeError("policy rotation step exceeds safety limit")
        if self.safety.workspace_min is not None and np.any(action[:3] < self.safety.workspace_min):
            raise RuntimeError("policy target is below workspace minimum")
        if self.safety.workspace_max is not None and np.any(action[:3] > self.safety.workspace_max):
            raise RuntimeError("policy target is above workspace maximum")
        return action

    def step(self, *, motion_allowed: Any | None = None) -> ChunkStepInfo:
        """Infer, validate and execute exactly one action step."""
        if self._reset_thread is not None:
            self._reset_thread.join()
            self._reset_thread = None
        started = time.monotonic()
        if self._image_health_check is not None:
            self._image_health_check()
        observation = self.observation() if self.chunks.needs_observation else {}
        action, info = self.chunks.step(observation)
        self._initialized = True
        state = np.asarray(self.robot.get_state(), dtype=np.float64)
        observation = {"state": state}
        action = self._decode_action(action, observation)
        step_id = self._debug_step_id
        self._debug_step_id += 1
        self._trace(
            "action_selected", step_id=step_id, scheduler=asdict(info),
            observation_state=observation["state"], state_before=state,
            action=action, execute_enabled=bool(getattr(self.robot, "execute_enabled", False)),
        )
        safe_action = self._validate(action, state)
        with self._motion_lock:
            if motion_allowed is not None and not motion_allowed():
                raise InterruptedError("client motion was stopped before action execution")
            if info.chunk_index == 0:
                logger.info(
                    "action target: chunk=%d index=%d actual_xyz=%s target_xyz=%s "
                    "delta_xyz_m=%s gripper=%.4f fused_steps=%d",
                    info.chunk_id,
                    info.chunk_index,
                    np.round(state[:3], 5).tolist(),
                    np.round(safe_action[:3], 5).tolist(),
                    np.round(safe_action[:3] - state[:3], 5).tolist(),
                    safe_action[9],
                    info.fused_overlap_steps,
                )
            if getattr(self.robot, "execute_enabled", False):
                command_started = time.monotonic()
                if self._last_command_started is not None:
                    interval = command_started - self._last_command_started
                    if interval > self.period_s * 1.25 and command_started - self._last_period_warning > 2:
                        logger.warning("servo command interval %.1f ms exceeds target %.1f ms",
                                       interval * 1000, self.period_s * 1000)
                        self._last_period_warning = command_started
                self._last_command_started = command_started
                try:
                    self.robot.send_absolute_action(safe_action, command_gripper=True)
                except BaseException as error:
                    self._trace("action_error", step_id=step_id, error=repr(error))
                    raise
                self._trace("action_sent", step_id=step_id, command_duration_s=time.monotonic() - command_started)
            else:
                logger.info(
                    "dry-run action chunk=%d index=%d target=%s",
                    info.chunk_id,
                    info.chunk_index,
                    np.round(safe_action, 5).tolist(),
                )
        if self.chunks.needs_prefetch:
            # The background camera readers return fresh snapshots without
            # waiting for the next camera frame in the servo thread.
            current_observation = self.observation()
            if self.chunks.prefetch(current_observation) and not info.prefetch_started:
                info = replace(info, prefetch_started=True)
        if self._debug_callback is not None:
            self._trace(
                "action_observed", step_id=step_id, state_after=self.robot.get_state(),
                elapsed_s=time.monotonic() - started,
            )
        remaining = self.period_s - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
        return info

    def reset_chunks(self) -> None:
        """Discard pending inference and action chunks after a stop."""
        with self._motion_lock:
            self.chunks.reset()
            self._initialized = False
            self._last_command_started = None
            self._debug_generation += 1
            self._trace("chunks_reset")

    def stop(self, *, wait_for_scheduler: bool = True) -> None:
        """Stop robot servo output and invalidate pending actions."""
        # Issue the hardware stop before waiting for a possibly slow model
        # request to release the scheduler lock.
        self.robot.stop()
        if wait_for_scheduler:
            self.reset_chunks()
        elif self._reset_thread is None or not self._reset_thread.is_alive():
            self._reset_thread = threading.Thread(
                target=self.reset_chunks, name="ur7-reset-scheduler", daemon=True
            )
            self._reset_thread.start()

    def move_to_joints(self, config: JointResetConfig) -> None:
        """Reset the exact joint branch under the same lock as policy writes."""
        with self._motion_lock:
            current = self.robot.get_joint_positions()
            target = config.target_from(current)
            logger.info(
                "joint reset current=%s target=%s delta=%s",
                np.round(current, 5).tolist(), np.round(target, 5).tolist(),
                np.round(target - current, 5).tolist(),
            )
            if getattr(self.robot, "execute_enabled", False) or getattr(self.robot, "simulate_motion", False):
                actual = self.robot.move_to_joint_target(config)
                logger.info("joint reset reached=%s", np.round(actual, 5).tolist())
            else:
                logger.info("dry-run joint reset; no motion sent")

    def move_to_pose(
        self,
        pose: Any,
        *,
        duration_s: float,
        position_tolerance_m: float | None = None,
        rotation_tolerance_rad: float | None = None,
        workspace_min: Any | None = None,
        workspace_max: Any | None = None,
        joint_reset_config: CartesianResetConfig | None = None,
    ) -> None:
        """Reset a TCP pose on a fixed joint branch when configured."""
        target_pose = np.asarray(pose, dtype=np.float64)
        if target_pose.shape != (9,) or not np.isfinite(target_pose).all():
            raise ValueError("EEF pose must be a finite 9D xyz+rot6d vector")
        if duration_s <= 0:
            raise ValueError("pose motion duration must be positive")
        if position_tolerance_m is not None and position_tolerance_m <= 0:
            raise ValueError("position tolerance must be positive")
        if rotation_tolerance_rad is not None and rotation_tolerance_rad <= 0:
            raise ValueError("rotation tolerance must be positive")
        lower = None if workspace_min is None else np.asarray(workspace_min, dtype=np.float64)
        upper = None if workspace_max is None else np.asarray(workspace_max, dtype=np.float64)
        if lower is not None and (lower.shape != (3,) or not np.isfinite(lower).all()):
            raise ValueError("workspace_min must be a finite 3D vector")
        if upper is not None and (upper.shape != (3,) or not np.isfinite(upper).all()):
            raise ValueError("workspace_max must be a finite 3D vector")
        if lower is not None and upper is not None and np.any(lower > upper):
            raise ValueError("workspace_min must not exceed workspace_max")
        if lower is not None and np.any(target_pose[:3] < lower):
            raise ValueError("target EEF pose is below workspace_min")
        if upper is not None and np.any(target_pose[:3] > upper):
            raise ValueError("target EEF pose is above workspace_max")
        with self._motion_lock:
            if joint_reset_config is not None:
                if getattr(self.robot, "execute_enabled", False) or getattr(self.robot, "simulate_motion", False):
                    self.robot.move_to_pose_on_joint_branch(
                        target_pose, joint_reset_config,
                        position_tolerance_m=position_tolerance_m or 0.01,
                        rotation_tolerance_rad=rotation_tolerance_rad or 0.05,
                    )
                else:
                    logger.info("dry-run joint reset: TCP target=%s joint reference=%s", target_pose.tolist(), joint_reset_config.joint_reference_rad)
                return
            current_state = np.asarray(self.robot.get_state(), dtype=np.float64)
            current = current_state[:9]
            gripper = float(current_state[9])
            first = np.concatenate((current, [gripper]))
            target = np.concatenate((target_pose, [gripper]))
            translation_distance = float(np.linalg.norm(target_pose[:3] - current[:3]))
            rotation_delta = rotation_distance(target_pose[3:9], current[3:9])
            max_translation_step = min(self.safety.max_translation_step_m * 0.5, 0.025)
            max_rotation_step = min(self.safety.max_rotation_step_rad * 0.5, 0.2)
            steps = max(
                1,
                int(round(duration_s * (1.0 / self.period_s))),
                int(np.ceil(translation_distance / max_translation_step)),
                int(np.ceil(rotation_delta / max_rotation_step)),
            )
            sender = getattr(self.robot, "send_reset_action", self.robot.send_absolute_action)
            for index in range(1, steps + 1):
                action = interpolate_absolute_action(first, target, index / steps)
                if getattr(self.robot, "execute_enabled", False) or getattr(
                    self.robot, "simulate_motion", False
                ):
                    sender(action)
                else:
                    logger.info("dry-run reset target=%s", np.round(action, 5).tolist())
                if index < steps:
                    time.sleep(self.period_s)
            self.robot.stop()
            if getattr(self.robot, "execute_enabled", False):
                actual = np.asarray(self.robot.get_state(), dtype=np.float64)
                actual_position_error = float(np.linalg.norm(actual[:3] - target_pose[:3]))
                actual_rotation_error = rotation_distance(actual[3:9], target_pose[3:9])
                if (
                    position_tolerance_m is not None
                    and actual_position_error > position_tolerance_m
                ):
                    raise RuntimeError(
                        f"EEF reset position error {actual_position_error:.4f}m exceeds "
                        f"{position_tolerance_m:.4f}m"
                    )
                if (
                    rotation_tolerance_rad is not None
                    and actual_rotation_error > rotation_tolerance_rad
                ):
                    raise RuntimeError(
                        f"EEF reset rotation error {actual_rotation_error:.4f}rad exceeds "
                        f"{rotation_tolerance_rad:.4f}rad"
                    )

    def open_gripper(self, *, wait_s: float = 0.0, min_fraction: float | None = None) -> None:
        """Open the gripper only while the client is stopped."""
        if wait_s < 0:
            raise ValueError("gripper wait_s must be non-negative")
        if min_fraction is not None and not 0.0 <= min_fraction <= 1.0:
            raise ValueError("gripper min_fraction must be in [0, 1]")
        with self._motion_lock:
            opener = getattr(self.robot, "open_gripper", None)
            if opener is None:
                raise RuntimeError("configured robot does not expose open_gripper()")
            if getattr(self.robot, "execute_enabled", False) or getattr(
                self.robot, "simulate_motion", False
            ):
                opener()
                if wait_s:
                    time.sleep(wait_s)
                if min_fraction is not None and (
                    getattr(self.robot, "execute_enabled", False)
                    or getattr(self.robot, "simulate_motion", False)
                ):
                    state = np.asarray(self.robot.get_state(), dtype=np.float64)
                    if state.shape[0] < 10 or state[9] < min_fraction:
                        raise RuntimeError(
                            f"gripper opening fraction {state[9] if state.shape[0] >= 10 else 'unknown'} "
                            f"is below {min_fraction}"
                        )
            else:
                logger.info("dry-run open gripper")

    def run(self, *, max_steps: int) -> list[ChunkStepInfo]:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.chunks.reset()
        records: list[ChunkStepInfo] = []
        try:
            for _ in range(max_steps):
                info = self.step()
                records.append(info)
            return records
        finally:
            self.robot.stop()

    def close(self) -> None:
        if self._reset_thread is not None:
            self._reset_thread.join()
            self._reset_thread = None
        self.chunks.close()


def _solid_images() -> dict[str, np.ndarray]:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    return {"base": image, "wrist": image.copy()}


def _load_config(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required for --config") from error
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError("client config must contain a YAML mapping")
    return value


def _cfg(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _debug_value(value: Any) -> Any:
    """Convert one request/response value to a replayable JSON value."""
    if isinstance(value, np.ndarray):
        buffer = BytesIO()
        np.save(buffer, np.asarray(value), allow_pickle=False)
        compressed = zlib.compress(buffer.getvalue(), level=3)
        return {
            "type": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "encoding": "zlib+npy+base64",
            "data_base64": base64.b64encode(compressed).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "encoding": "base64",
            "length": len(value),
            "data_base64": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, Mapping):
        return {str(key): _debug_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_debug_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _make_debug_recorder(path: str | Path) -> tuple[Callable[[dict[str, Any]], None], Callable[[], None]]:
    """Create a JSONL recorder with request images saved as lossless PNGs."""
    log_path = Path(path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir = log_path.parent / f"{log_path.stem}_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    write_lock = threading.Lock()
    pending: queue.Queue = queue.Queue(maxsize=512)
    closed = False
    dropped = 0

    def write_event(event: dict[str, Any]) -> None:
        payload = dict(event)
        payload["schema_version"] = 1
        payload.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        observation = payload.get("observation")
        if isinstance(observation, Mapping):
            observation = dict(observation)
            request_id = int(payload.get("request_id", 0))
            attempt = int(payload.get("attempt", 0))
            for key, filename in (("image", "base"), ("wrist_image", "wrist")):
                image = observation.get(key)
                if image is None:
                    continue
                try:
                    from PIL import Image

                    array = np.asarray(image)
                    image_path = image_dir / f"request_{request_id:06d}_attempt_{attempt}_{filename}.png"
                    Image.fromarray(array, mode="RGB").save(image_path, format="PNG")
                    observation[key] = {
                        "type": "raw_image",
                        "path": str(image_path.relative_to(log_path.parent)),
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "format": "PNG",
                    }
                except Exception as error:
                    observation[key] = {"type": "raw_image_error", "error": repr(error)}
            payload["observation"] = observation
        payload = _debug_value(payload)
        with write_lock:
            stream.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
            stream.flush()

    def writer() -> None:
        while True:
            event = pending.get()
            try:
                if event is None:
                    return
                write_event(event)
            except Exception:
                logger.exception("failed to write trace event to %s", log_path)
            finally:
                pending.task_done()

    worker = threading.Thread(target=writer, name="ur7-trace-writer", daemon=True)
    worker.start()

    def record(event: dict[str, Any]) -> None:
        nonlocal dropped
        if closed:
            return
        # Camera/NumPy buffers can be reused; freeze before background encoding.
        payload = copy.deepcopy(event)
        payload.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        try:
            pending.put_nowait(payload)
        except queue.Full:
            dropped += 1
            if dropped == 1:
                logger.error("trace queue full; events are being dropped: %s", log_path)

    def close() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        pending.put(None)
        worker.join()
        with write_lock:
            if dropped:
                stream.write(json.dumps({"event_type": "trace_dropped", "count": dropped}) + "\n")
            stream.flush()
            stream.close()

    return record, close


class _SessionDebugRecorder:
    """Create a fresh log and image directory for every completed reset."""

    def __init__(self, path: str | Path, metadata: dict[str, Any] | None = None):
        self._base_path = Path(path).expanduser()
        self._record: Callable[[dict[str, Any]], None] | None = None
        self._close: Callable[[], None] = lambda: None
        self._session = 0
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._metadata = metadata or {}
        self._lock = threading.RLock()

    def start_session(self) -> None:
        with self._lock:
            self._close()
            self._session += 1
            suffix = self._base_path.suffix or ".jsonl"
            stem = self._base_path.stem
            path = self._base_path.with_name(f"{stem}_{self._run_id}_session_{self._session:03d}{suffix}")
            self._record, self._close = _make_debug_recorder(path)
            self._record({"event_type": "session_start", "metadata": self._metadata})
            logger.info("started policy debug session %03d: %s", self._session, path)

    def __call__(self, event: dict[str, Any]) -> None:
        with self._lock:
            if self._record is not None:
                self._record(event)

    def close(self) -> None:
        with self._lock:
            self._close()
            self._record = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an OpenPI Pi05 action client on UR7")
    parser.add_argument("--config", default=None, help="YAML deployment configuration")
    parser.add_argument("--server", default=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="per-request inference timeout in seconds; first JAX inference may compile for a while",
    )
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--reconnect-attempts", type=int, default=1)
    parser.add_argument("--task", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--control-hz", type=float, default=None)
    parser.add_argument("--reinfer", type=int, default=None,
                        help="RTC queue threshold; synchronous actions per request when RTC is off")
    parser.add_argument("--blend", type=int, default=None,
                        help="number of new actions blended after removing elapsed actions")
    parser.add_argument("--inference-delay-steps", type=int, default=None,
                        help="initial delay; adapts from the mean of the last 10 async requests")
    parser.add_argument("--action-space", choices=("absolute", "relative_se3"), default=None)
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--gripper-connection", default=None)
    parser.add_argument("--base-serial")
    parser.add_argument("--wrist-serial")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--rtc", dest="rtc_enabled", action="store_true", help="enable time-aligned RTC fusion")
    parser.add_argument("--no-rtc", dest="rtc_enabled", action="store_false", help="synchronous inference, execute reinfer actions per request")
    parser.add_argument("--smoke", action="store_true", help="Use a fake robot and black images")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="record every policy request/response, state, and image for replay",
    )
    parser.add_argument(
        "--debug-log",
        default=None,
        help="JSONL debug log path; defaults to .data/debug/ur7_client_<timestamp>.jsonl",
    )
    parser.add_argument("--interactive", dest="interactive", action="store_true", help="Enable keyboard state machine")
    parser.add_argument("--non-interactive", dest="interactive", action="store_false", help="Run a fixed number of steps")
    parser.set_defaults(interactive=None, rtc_enabled=None)
    args = parser.parse_args()
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")
    config = _load_config(args.config) if args.config else {}
    server = args.server or _cfg(config, "server", "uri", default="ws://127.0.0.1:8000")
    task = args.task or _cfg(config, "task", default="")
    control_hz = args.control_hz or _cfg(config, "control", "control_hz", default=10.0)
    rtc_config = config.get("rtc", {})
    if not isinstance(rtc_config, Mapping):
        parser.error("rtc must contain enabled, reinfer, blend and inference_delay_steps")
    unknown = set(rtc_config) - {"enabled", "reinfer", "blend", "inference_delay_steps"}
    if unknown:
        parser.error(f"obsolete/unknown rtc keys: {sorted(unknown)}; use enabled, reinfer, blend, inference_delay_steps")
    reinfer = args.reinfer if args.reinfer is not None else rtc_config.get("reinfer", 30)
    blend = args.blend if args.blend is not None else rtc_config.get("blend", 16)
    inference_delay_steps = (args.inference_delay_steps if args.inference_delay_steps is not None
                             else rtc_config.get("inference_delay_steps", 4))
    try:
        validate_rtc_config(reinfer, blend, inference_delay_steps)
    except ValueError as error:
        parser.error(str(error))
    if "enabled" in rtc_config and not isinstance(rtc_config["enabled"], bool):
        parser.error("rtc.enabled must be a YAML boolean")
    rtc_enabled = (
        bool(_cfg(config, "rtc", "enabled", default=False))
        if args.rtc_enabled is None
        else bool(args.rtc_enabled)
    )
    action_space = args.action_space or _cfg(config, "control", "action_space", default="absolute")
    robot_ip = args.robot_ip or _cfg(config, "robot", "ip", default="192.168.1.188")
    gripper_connection = args.gripper_connection or _cfg(config, "robot", "gripper_connection", default=None)
    if not task:
        parser.error("--task or config task is required")
    if args.interactive is None:
        args.interactive = not args.smoke
    reset_joints = home_joints = None
    if args.interactive:
        try:
            reset_mapping = config.get("reset", {})
            reset_joints = JointResetConfig.from_mapping(reset_mapping)
            home_joints = JointResetConfig.from_mapping({**reset_mapping, **config.get("home", {})})
        except (TypeError, ValueError) as error:
            parser.error(f"reset/home requires valid absolute joints_rad: {error}")
    debug_enabled = bool(args.debug or _cfg(config, "debug", "enabled", default=False))
    debug_close: Callable[[], None] = lambda: None
    debug_callback: Callable[[dict[str, Any]], None] | None = None
    debug_recorder: _SessionDebugRecorder | None = None
    if debug_enabled:
        configured_debug_path = args.debug_log or _cfg(config, "debug", "log_path")
        if configured_debug_path:
            debug_path = Path(str(configured_debug_path)).expanduser()
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = Path(".data") / "debug" / f"ur7_client_{stamp}.jsonl"
        debug_recorder = _SessionDebugRecorder(debug_path, metadata={
            "config": config, "server": server, "action_space": action_space,
            "control_hz": control_hz, "rtc_enabled": rtc_enabled,
            "reinfer": reinfer, "blend": blend, "execute": args.execute,
            "initial_inference_delay_steps": inference_delay_steps,
        })
        debug_callback = debug_recorder
        debug_close = debug_recorder.close
        logger.info("policy debug logging enabled; a new session starts after reset")
    if args.execute and args.confirm != EXECUTE_CONFIRMATION:
        parser.error(f"motion requires --confirm {EXECUTE_CONFIRMATION}")
    effective_timeout = args.timeout if args.timeout != 120.0 else float(_cfg(config, "server", "timeout_s", default=120.0))
    effective_connect_timeout = args.connect_timeout if args.connect_timeout != 30.0 else float(_cfg(config, "server", "connect_timeout_s", default=30.0))
    effective_reconnect_attempts = args.reconnect_attempts if args.reconnect_attempts != 1 else int(_cfg(config, "server", "reconnect_attempts", default=1))
    if effective_timeout <= 0 or effective_connect_timeout <= 0:
        parser.error("--timeout and --connect-timeout must be positive")
    if effective_reconnect_attempts < 0:
        parser.error("--reconnect-attempts must be non-negative")
    try:
        policy = WebSocketPolicyClient(
            uri=server,
            timeout_s=effective_timeout,
            connect_timeout_s=effective_connect_timeout,
            reconnect_attempts=effective_reconnect_attempts,
            debug_callback=debug_callback,
        )
    except BaseException:
        debug_close()
        raise
    robot: Any
    cameras: Any = None
    if args.smoke:
        robot = FakeUR7Robot()
        image_reader = _solid_images
    else:
        servo_config = ServoConfig(
            acceleration=float(_cfg(config, "robot", "servo", "acceleration", default=0.3)),
            velocity=float(_cfg(config, "robot", "servo", "velocity", default=0.15)),
            time_step=float(_cfg(config, "robot", "servo", "time_step", default=0.1)),
            lookahead_time=float(_cfg(config, "robot", "servo", "lookahead_time", default=0.2)),
            gain=int(_cfg(config, "robot", "servo", "gain", default=100)),
        )
        robot = UR7Robot(
            robot_ip,
            background_gripper=True,
            execute=args.execute,
            confirmation=args.confirm,
            gripper_connection=gripper_connection,
            gripper_speed=float(_cfg(config, "robot", "gripper_speed", default=0.3)),
            tcp_offset=_cfg(config, "robot", "tcp_offset"),
            state=StateConfig(
                include_force_torque=bool(_cfg(config, "robot", "state", "include_force_torque", default=True)),
                include_velocity=bool(_cfg(config, "robot", "state", "include_velocity", default=True)),
                include_joint_position=bool(_cfg(config, "robot", "state", "include_joint_position", default=False)),
                include_joint_velocity=bool(_cfg(config, "robot", "state", "include_joint_velocity", default=False)),
            ),
            servo=servo_config,
        )
        cameras = RealSensePair(
            base_serial=args.base_serial or _cfg(config, "cameras", "base", "serial"),
            wrist_serial=args.wrist_serial or _cfg(config, "cameras", "wrist", "serial"),
            width=int(_cfg(config, "cameras", "base", "width", default=640)),
            height=int(_cfg(config, "cameras", "base", "height", default=480)),
            fps=int(_cfg(config, "cameras", "base", "fps", default=30)),
        )
        image_reader = cameras.read
    client = UR7DeploymentClient(
        robot,
        policy,
        image_reader,
        task=task,
        control_hz=control_hz,
        reinfer=reinfer,
        blend=blend,
        inference_delay_steps=inference_delay_steps,
        rtc_enabled=rtc_enabled,
        action_space=action_space,
        image_size=int(_cfg(config, "control", "image_size", default=224)),
        flip_wrist_image=bool(_cfg(config, "cameras", "wrist", "horizontal_flip", default=False)),
        debug_callback=debug_callback,
        safety=SafetyLimits(
            max_translation_step_m=float(_cfg(config, "safety", "max_translation_step_m", default=0.05)),
            max_rotation_step_rad=float(_cfg(config, "safety", "max_rotation_step_rad", default=0.35)),
            workspace_min=tuple(_cfg(config, "safety", "workspace_min", default=[])) or None,
            workspace_max=tuple(_cfg(config, "safety", "workspace_max", default=[])) or None,
        ),
    )
    def trace_reset() -> None:
        state = robot.get_state()
        logger.info("joint reset TCP state: xyz=%s rot6d=%s", np.round(state[:3], 5).tolist(), np.round(state[3:9], 5).tolist())
        if debug_recorder is not None:
            debug_recorder.start_session()
            client._trace("reset_state", state=state, joints=robot.get_joint_positions())

    try:
        if args.interactive:
            controller = InteractiveUR7Controller(
                client,
                reset_joints=reset_joints,
                home_joints=home_joints,
                gripper_open_wait_s=float(_cfg(config, "reset", "gripper_open_wait_s", default=0.8)),
                gripper_open_min=float(_cfg(config, "reset", "gripper_open_min", default=0.95)),
                control_hz=float(control_hz),
                on_reset=trace_reset,
            )
            completed = controller.run(max_steps=args.steps if args.steps else None)
            logger.info("interactive session completed %d action steps", completed)
            return
        if debug_recorder is not None:
            debug_recorder.start_session()
        records = client.run(max_steps=args.steps or 16)
        logger.info(
            "completed %d steps, chunks=%d, prefetch_hits=%d",
            len(records),
            len({record.chunk_id for record in records}),
            client.chunks.prefetch_hit_count,
        )
    finally:
        client.close()
        if cameras is not None:
            cameras.close()
        robot.close()
        debug_close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
