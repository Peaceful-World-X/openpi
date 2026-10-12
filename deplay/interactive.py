"""Interactive keyboard state machine for the UR7 deployment client."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from .keyboard import TerminalKeyReader
from .ur7.joint_reset import JointResetConfig

logger = logging.getLogger(__name__)


# 表示客户端在等待复位、停止、运行及辅助动作期间所处阶段的常量。
WAIT_RESET = "wait_reset"
STOPPED = "stopped"
RUNNING = "running"
RESETTING = "resetting"
OPENING_GRIPPER = "opening_gripper"
HOMING = "homing"

_KEY_LABELS = {
    "enter": "Enter（回车）",
    "space": "Space（空格）",
    "h": "h",
    "quit": "q/Ctrl-C",
}


# 管理回车、空格和 h 按键驱动的复位、推理、停止、开夹爪与归位流程。
class InteractiveUR7Controller:
    """Run the deployment client with a responsive keyboard state machine."""

    def __init__(
        self,
        client: Any,
        *,
        reset_joints: JointResetConfig,
        home_joints: JointResetConfig,
        gripper_open_wait_s: float,
        gripper_open_min: float,
        control_hz: float,
        on_reset: Callable[[], None] | None = None,
    ):
        self.client = client
        self.reset_joints = reset_joints
        self.home_joints = home_joints
        self.gripper_open_wait_s = float(gripper_open_wait_s)
        self.gripper_open_min = float(gripper_open_min)
        self.control_period_s = 1.0 / float(control_hz)
        self.on_reset = on_reset
        self.state = WAIT_RESET
        self._error: BaseException | None = None
        self._worker: threading.Thread | None = None
        self._worker_result = 0
        self._worker_done = threading.Event()
        self._quit_requested = False
        self._run_generation = 0

    def _reset(self) -> None:
        self.state = RESETTING
        self.client.stop()
        self.client.move_to_joints(self.reset_joints)
        if self.on_reset is not None:
            self.on_reset()
        self.state = STOPPED

    def handle_key(self, key: str) -> None:
        before = self.state
        label = _KEY_LABELS.get(key, repr(key))
        logger.info("收到按键 %s，当前状态=%s", label, before)
        handled = False
        if key == "quit":
            handled = True
            self._quit_requested = True
            self._run_generation += 1
            self.state = STOPPED
            self.client.stop(wait_for_scheduler=False)
        elif key == "enter" and self.state == WAIT_RESET:
            handled = True
            self._reset()
        elif key == "enter" and self.state == STOPPED:
            handled = True
            self.state = OPENING_GRIPPER
            try:
                self.client.open_gripper(
                    wait_s=self.gripper_open_wait_s,
                    min_fraction=self.gripper_open_min,
                )
            finally:
                self.state = STOPPED
        elif key == "h" and self.state == STOPPED:
            handled = True
            self.state = HOMING
            try:
                self.client.stop()
                self.client.move_to_joints(self.home_joints)
                if self.on_reset is not None:
                    self.on_reset()
            finally:
                self.state = STOPPED
        elif key == "space" and self.state == STOPPED:
            handled = True
            self._run_generation += 1
            self.state = RUNNING
        elif key == "space" and self.state == RUNNING:
            handled = True
            self._run_generation += 1
            self.state = STOPPED
            self.client.stop(wait_for_scheduler=False)
        if handled:
            logger.info("按键 %s 已处理，状态=%s -> %s", label, before, self.state)
        else:
            logger.info("按键 %s 在状态=%s 下已忽略", label, before)

    def _start_step(self, generation: int) -> None:
        def work() -> None:
            try:
                self.client.step(
                    motion_allowed=lambda: (
                        self.state == RUNNING and self._run_generation == generation
                    )
                )
                self._worker_result = 1
            except InterruptedError:
                self._worker_result = 0
            except BaseException as error:
                self._error = error
                self._worker_result = -1
            finally:
                self._worker_done.set()

        self._worker_result = 0
        self._worker_done.clear()
        self._worker = threading.Thread(target=work, name="ur7-policy-step", daemon=True)
        self._worker.start()

    def run(self, *, max_steps: int | None = None) -> int:
        """Poll keys and execute actions until quit or an optional step limit."""
        completed = 0
        logger.info("按回车复位；停止时按空格开始、空格停止、回车开夹爪、h 归位、q 退出")
        try:
            with TerminalKeyReader() as keyboard:
                while not self._quit_requested and (
                    self.state != STOPPED or max_steps is None or completed < max_steps
                ):
                    for key in keyboard.poll():
                        self.handle_key(key)
                    if self._worker is not None and not self._worker.is_alive():
                        self._worker.join()
                        self._worker = None
                        if self._worker_result < 0:
                            self.state = STOPPED
                            self.client.stop()
                            raise self._error  # type: ignore[misc]
                        completed += self._worker_result
                        if max_steps is not None and completed >= max_steps:
                            self.state = STOPPED
                            self.client.stop()
                    if self.state == RUNNING and self._worker is None:
                        self._start_step(self._run_generation)
                    elif self._worker is not None:
                        # Wake as soon as the servo step completes, instead
                        # of adding up to 20 ms of keyboard polling jitter.
                        self._worker_done.wait(min(0.02, self.control_period_s))
                    else:
                        time.sleep(min(0.02, self.control_period_s))
                    if self.state == STOPPED and max_steps is not None and completed >= max_steps:
                        break
        finally:
            if self._worker is not None:
                self._worker.join()
                self._worker = None
        return completed
