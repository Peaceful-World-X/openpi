"""Directly replay a parquet action trajectory on a UR7."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from .ur7.robot import EXECUTE_CONFIRMATION
from .ur7.robot import RESET_SERVO_CONFIG
from .ur7.robot import ServoConfig
from .ur7.robot import StateConfig
from .ur7.robot import UR7Robot
from .ur7.se3 import interpolate_absolute_action

logger = logging.getLogger(__name__)


def _config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _get(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key, default)
    return value


def _read_actions(path: Path, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Read [eef xyz, eef rot6d, gripper.open] and timestamps as stored."""
    import pyarrow.parquet as parquet

    table = parquet.read_table(path, columns=[column, "timestamp"])
    actions = np.asarray(table[column].to_pylist(), dtype=np.float64)[:, :10]
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 10:
        raise ValueError(f"{column} must contain 10D actions, got {actions.shape}")
    return actions, timestamps - timestamps[0]


# 负责把 parquet 中的 EEF 和夹爪动作按原始时间戳直接发送给 UR7.
class UR7Replayer:
    def __init__(self, robot: Any, actions: np.ndarray, timestamps: np.ndarray, speed_scale: float):
        self.robot = robot
        self.actions = actions
        self.timestamps = timestamps
        self.speed_scale = speed_scale

    def move_to_first(self, duration_s: float, wait_s: float, *, dry_run: bool) -> None:
        if dry_run:
            return
        current = np.r_[self.robot.get_eef_pose(), self.actions[0, 9]]
        target = self.actions[0]
        sender = self.robot.send_reset_action
        steps = max(1, int(np.ceil(duration_s / RESET_SERVO_CONFIG.time_step)))
        for index in range(1, steps + 1):
            sender(interpolate_absolute_action(current, target, index / steps))
            time.sleep(RESET_SERVO_CONFIG.time_step)
        self.robot.stop()
        time.sleep(wait_s)

    def run(self, *, dry_run: bool, initial_wait_s: float, initial_move_duration_s: float) -> int:
        try:
            self.move_to_first(initial_move_duration_s, initial_wait_s, dry_run=dry_run)
            started = time.monotonic()
            for index, action in enumerate(self.actions):
                delay = started + self.timestamps[index] / self.speed_scale - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                if dry_run:
                    logger.info("row=%d eef=%s gripper.open=%.3f", index, action[:9].tolist(), action[9])
                else:
                    self.robot.send_absolute_action(action, command_gripper=True)
            return len(self.actions)
        finally:
            self.robot.stop()


class _DryRunRobot:
    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a UR7 parquet action trajectory")
    parser.add_argument("episode", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("client.yaml"))
    parser.add_argument("--column")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--speed-scale", type=float)
    parser.add_argument("--initial-wait-s", type=float)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.execute and args.confirm != EXECUTE_CONFIRMATION:
        parser.error(f"real motion requires --confirm {EXECUTE_CONFIRMATION}")
    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run cannot be combined")

    config = _config(args.config)
    column = args.column or _get(config, "replay", "action_column", default="action")
    speed_scale = args.speed_scale or _get(config, "replay", "speed_scale", default=1.0)
    wait_s = args.initial_wait_s if args.initial_wait_s is not None else _get(config, "replay", "initial_wait_s", default=2.0)
    actions, timestamps = _read_actions(args.episode, column)
    actions = actions[args.start:args.end]
    timestamps = timestamps[args.start:args.end]

    robot: Any = _DryRunRobot()
    if args.execute:
        robot = UR7Robot(
            _get(config, "robot", "ip", default="192.168.1.188"),
            execute=True,
            confirmation=args.confirm,
            gripper_connection=_get(config, "robot", "gripper_connection"),
            servo=ServoConfig(
                acceleration=float(_get(config, "robot", "servo", "acceleration", default=0.3)),
                velocity=float(_get(config, "robot", "servo", "velocity", default=0.15)),
                time_step=float(_get(config, "robot", "servo", "time_step", default=0.1)),
                lookahead_time=float(_get(config, "robot", "servo", "lookahead_time", default=0.2)),
                gain=int(_get(config, "robot", "servo", "gain", default=100)),
            ),
            state=StateConfig(include_force_torque=True, include_velocity=True),
        )
    try:
        count = UR7Replayer(robot, actions, timestamps, float(speed_scale)).run(
            dry_run=not args.execute,
            initial_wait_s=float(wait_s),
            initial_move_duration_s=float(_get(config, "replay", "initial_move_duration_s", default=1.5)),
        )
        logger.info("replayed %d actions", count)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
