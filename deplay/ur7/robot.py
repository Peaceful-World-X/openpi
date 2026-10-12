"""UR7 RTDE adapter extracted from the RLinf real-world environment."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .robotiq_tcp import RobotiqTCP
from .robotiq_tcp import AsyncRobotiqTCP
from .se3 import matrix_to_rot6d, rot6d_to_matrix
from .joint_reset import JointResetConfig
from .cartesian_reset import CartesianResetConfig
from .tcp_config import configure_tcp, validate_tcp_offset

EXECUTE_CONFIRMATION = "ENABLE_UR7_MOTION"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServoConfig:
    """Conservative servoL parameters."""

    acceleration: float = 0.3
    velocity: float = 0.15
    time_step: float = 0.1
    lookahead_time: float = 0.2
    gain: int = 100


RESET_SERVO_CONFIG = ServoConfig(
    acceleration=1.2,
    velocity=0.5,
    time_step=0.008,
    lookahead_time=0.1,
    gain=200,
)


@dataclass(frozen=True)
class StateConfig:
    """Optional UR state blocks appended after the mandatory 10D prefix."""

    include_force_torque: bool = True
    include_velocity: bool = True
    include_joint_position: bool = False
    include_joint_velocity: bool = False

    @property
    def dimension(self) -> int:
        """Return the configured canonical state dimension."""
        return 10 + 6 * sum(
            (
                self.include_force_torque,
                self.include_velocity,
                self.include_joint_position,
                self.include_joint_velocity,
            )
        )


class UR7Robot:
    """UR RTDE state reader and explicitly armed action writer."""

    def __init__(
        self,
        robot_ip: str,
        *,
        execute: bool = False,
        confirmation: str | None = None,
        gripper_connection: str | None = None,
        gripper_speed: float = 0.3,
        tcp_offset: tuple[float, ...] | list[float] | None = None,
        servo: ServoConfig | None = None,
        teleoperation_servo: ServoConfig | None = None,
        state: StateConfig | None = None,
        background_gripper: bool = False,
    ):
        if execute and confirmation != EXECUTE_CONFIRMATION:
            raise PermissionError(
                f"robot motion requires confirmation text: {EXECUTE_CONFIRMATION}"
            )
        if not np.isfinite(gripper_speed) or not 0.0 <= gripper_speed <= 1.0:
            raise ValueError("robot.gripper_speed must be finite and in [0, 1]")
        self._gripper_speed = float(gripper_speed)
        expected_tcp = None if tcp_offset is None else validate_tcp_offset(tcp_offset)
        try:
            import rtde_receive
            from scipy.spatial.transform import Rotation
        except ImportError as error:
            raise RuntimeError("UR dependencies are not installed; install vla-sft[ur]") from error

        self._robot_ip = robot_ip
        self._rotation = Rotation
        self._receive = rtde_receive.RTDEReceiveInterface(robot_ip)
        self._control: Any | None = None
        self._execute = execute
        self._servo = servo or ServoConfig()
        self._teleoperation_servo = teleoperation_servo or self._servo
        self._state_config = state or StateConfig()
        self._gripper = None
        try:
            if execute:
                import rtde_control

                self._control = rtde_control.RTDEControlInterface(robot_ip)
            if expected_tcp is not None:
                configure_tcp(self._control, self._receive, robot_ip, expected_tcp)
            gripper_type = AsyncRobotiqTCP if background_gripper else RobotiqTCP
            self._gripper = gripper_type(gripper_connection, writable=execute) if gripper_connection else None
        except BaseException:
            # A failed TCP check must not leave a connection/register owner behind.
            try:
                if self._control is not None:
                    self._control.disconnect()
            finally:
                self._receive.disconnect()
            raise

    @property
    def execute_enabled(self) -> bool:
        """Whether this adapter can write the robot."""
        return self._execute

    def get_state(self) -> np.ndarray:
        """Read the canonical state with useful optional dimensions appended.

        The mandatory prefix is ``eef_xyz(3), eef_rot6d(6), gripper(1)``.
        Optional six-dimensional blocks follow in the order configured by
        :class:`StateConfig`; the recipe must match checkpoint metadata exactly.
        """
        eef_pose = self.get_eef_pose()
        gripper = self._gripper.get_open_fraction() if self._gripper else 1.0
        mandatory = np.concatenate((eef_pose, [gripper]))
        blocks = [mandatory]
        if self._state_config.include_force_torque:
            blocks.append(np.asarray(self._receive.getActualTCPForce(), dtype=np.float64))
        if self._state_config.include_velocity:
            blocks.append(np.asarray(self._receive.getActualTCPSpeed(), dtype=np.float64))
        if self._state_config.include_joint_position:
            blocks.append(np.asarray(self._receive.getActualQ(), dtype=np.float64))
        if self._state_config.include_joint_velocity:
            blocks.append(np.asarray(self._receive.getActualQd(), dtype=np.float64))
        state = np.concatenate(blocks)
        if state.shape != (self._state_config.dimension,) or not np.all(np.isfinite(state)):
            raise RuntimeError(f"invalid UR state: shape={state.shape}")
        return state

    def get_eef_pose(self) -> np.ndarray:
        """Read only canonical EEF xyz + rot6d for the high-rate arm loop."""
        native_pose = np.asarray(self._receive.getActualTCPPose(), dtype=np.float64)
        if native_pose.shape != (6,) or not np.all(np.isfinite(native_pose)):
            raise RuntimeError(f"invalid UR EEF pose: shape={native_pose.shape}")
        rotation = self._rotation.from_rotvec(native_pose[3:]).as_matrix()
        return np.concatenate((native_pose[:3], matrix_to_rot6d(rotation)))

    def get_joint_positions(self) -> np.ndarray:
        """Read the six UR joint positions in radians without creating a control interface."""
        joints = np.asarray(self._receive.getActualQ(), dtype=np.float64)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(f"invalid UR joint state: shape={joints.shape}")
        return joints

    def send_absolute_action(
        self,
        action: np.ndarray,
        *,
        command_gripper: bool = True,
    ) -> None:
        """Send a safety-validated absolute 10D target.

        Args:
            action: Canonical ``eef_xyz, eef_rot6d, gripper`` target.
            command_gripper: Whether to write the gripper component. Set this
                to false for arm-only commissioning even though the canonical
                action remains 10-dimensional.
        """
        self._send_absolute_action(
            action,
            command_gripper=command_gripper,
            servo=self._servo,
        )

    def send_reset_action(self, action: np.ndarray) -> None:
        """Send one arm-only reset target with the proven 8 ms GELLO servo parameters."""
        self._send_absolute_action(
            action,
            command_gripper=False,
            servo=RESET_SERVO_CONFIG,
        )

    def send_teleoperation_action(
        self,
        action: np.ndarray,
        *,
        command_gripper: bool = False,
    ) -> None:
        """Send one high-rate teleoperation target with its explicit servo profile."""
        self._send_absolute_action(
            action,
            command_gripper=command_gripper,
            servo=self._teleoperation_servo,
        )

    def send_joint_action(
        self,
        joints: np.ndarray,
        *,
        servo: ServoConfig | None = None,
    ) -> None:
        """Send one six-joint target with RTDE ``servoJ`` parameters."""
        if self._control is None:
            raise PermissionError("robot adapter is read-only")
        if self._receive.isEmergencyStopped():
            raise RuntimeError("UR emergency stop is active; refusing joint motion")
        if self._receive.isProtectiveStopped():
            raise RuntimeError("UR protective stop is active; refusing joint motion")
        value = np.asarray(joints, dtype=np.float64)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("joint action must be a finite 6D vector in radians")
        profile = servo or self._servo
        accepted = self._control.servoJ(
            value.tolist(),
            profile.acceleration,
            profile.velocity,
            profile.time_step,
            profile.lookahead_time,
            profile.gain,
        )
        if not accepted:
            raise RuntimeError("UR servoJ command was rejected")

    def move_to_joint_target(
        self,
        config: JointResetConfig,
    ) -> np.ndarray:
        """Return to exact absolute joint angles, including the wrist turn count."""
        if self._control is None:
            raise PermissionError("robot adapter is read-only")

        def stationary_joints() -> np.ndarray:
            if self._receive.isEmergencyStopped():
                raise RuntimeError("UR emergency stop is active during joint reset")
            if self._receive.isProtectiveStopped():
                raise RuntimeError("UR protective stop is active during joint reset")
            joints = self.get_joint_positions()
            velocity = np.asarray(self._receive.getActualQd(), dtype=np.float64)
            if velocity.shape != (6,) or not np.isfinite(velocity).all():
                raise RuntimeError("invalid joint velocity during reset")
            if np.max(np.abs(velocity)) > config.stationary_velocity_tolerance_rad_s:
                raise RuntimeError("UR is not stationary during joint reset")
            return joints

        target = config.target_from(stationary_joints())
        if not self._control.isJointsWithinSafetyLimits(target.tolist()):
            raise RuntimeError("joint reset target violates the controller safety limits")
        try:
            accepted = self._control.moveJ(
                target.tolist(), float(config.speed_rad_s), float(config.acceleration_rad_s2)
            )
            after = stationary_joints()
            if not accepted:
                raise RuntimeError("UR joint reset moveJ command was rejected")
        except BaseException:
            # moveJ needs stopJ (servoStop only stops servo mode). Preserve
            # the original fault if the controller no longer accepts writes.
            try:
                self._control.stopJ(float(config.acceleration_rad_s2))
            except Exception:
                pass
            raise
        error = float(np.max(np.abs(after - target)))
        if error > config.position_tolerance_rad:
            raise RuntimeError(
                f"joint reset target error={error:.4f}rad exceeds "
                f"tolerance={config.position_tolerance_rad:.4f}rad"
            )
        return after

    def _reset_forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """Query FK without the empty/zero TCP recipe bugs in ur_rtde 1.6.5.

        That SDK sends only six joint registers for an omitted TCP, leaving
        the old IK reference in the six TCP registers. Passing all-zero TCP
        uses that same six-value recipe with twelve values and can hang.
        Query a virtual 1 mm TCP for the zero-TCP case, then transform the
        computed pose back. This never changes the controller's active TCP.
        """
        tcp = validate_tcp_offset(self._control.getTCPOffset())
        query_tcp = tcp.copy()
        zero_tcp = not np.any(tcp)
        if zero_tcp:
            query_tcp[2] = 0.001
        forward = np.asarray(
            self._control.getForwardKinematics(joints.tolist(), query_tcp.tolist()),
            dtype=np.float64,
        )
        if forward.shape != (6,) or not np.isfinite(forward).all():
            raise RuntimeError("invalid forward kinematics for reset target")
        if zero_tcp:
            forward = forward.copy()
            forward[:3] -= self._rotation.from_rotvec(forward[3:]).apply(query_tcp[:3])
        return forward

    def move_to_pose_on_joint_branch(
        self,
        pose: np.ndarray,
        config: CartesianResetConfig,
        *,
        position_tolerance_m: float,
        rotation_tolerance_rad: float,
    ) -> np.ndarray:
        """Solve the TCP target on the reference branch with the nearest safe q6 turn."""
        if self._control is None:
            raise PermissionError("robot adapter is read-only")
        value = np.asarray(pose, dtype=np.float64)
        if value.shape != (9,) or not np.isfinite(value).all():
            raise ValueError("reset TCP pose must be a finite 9D vector")
        if any(not np.isfinite(x) or x <= 0 for x in (position_tolerance_m, rotation_tolerance_rad)):
            raise ValueError("reset TCP tolerances must be positive and finite")
        rotation = rot6d_to_matrix(value[3:9])
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
            raise ValueError("reset TCP orientation must define a valid rotation")
        native = np.r_[value[:3], self._rotation.from_matrix(rotation).as_rotvec()].tolist()
        reference = list(config.joint_reference_rad)
        if self._receive.isEmergencyStopped() or self._receive.isProtectiveStopped():
            raise RuntimeError("UR safety stop is active; refusing TCP reset")
        if not self._control.getInverseKinematicsHasSolution(native, reference):
            raise RuntimeError("reset TCP pose has no inverse kinematics solution")
        solution = self._control.getInverseKinematics(native, reference)
        before = self.get_joint_positions()
        candidates = config.target_candidates(solution, before)
        target_config = next((candidate for candidate in candidates
                              if self._control.isJointsWithinSafetyLimits(list(candidate.joints_rad))), None)
        if target_config is None:
            raise RuntimeError("no equivalent reset wrist target satisfies controller safety limits and configured travel limits")
        target = target_config.target_from(before)
        forward = self._reset_forward_kinematics(target)
        if forward.shape != (6,) or not np.isfinite(forward).all():
            raise RuntimeError("invalid forward kinematics for reset target")
        pos_error = float(np.linalg.norm(forward[:3] - value[:3]))
        rot_error = float((self._rotation.from_matrix(rotation).inv() * self._rotation.from_rotvec(forward[3:])).magnitude())
        if pos_error > 1e-4 or rot_error > 1e-3:
            raise RuntimeError(f"reset IK/FK mismatch: position={pos_error:.6f}m rotation={rot_error:.6f}rad")
        logger.info(
            "TCP reset joint plan: current=%s target=%s delta=%s; q6_deg=%.2f -> %.2f",
            np.round(before, 5).tolist(), np.round(target, 5).tolist(),
            np.round(target - before, 5).tolist(), np.degrees(before[5]), np.degrees(target[5]),
        )
        distance = float(np.max(np.abs(target - before)))
        speed = float(target_config.speed_rad_s)
        acceleration = float(target_config.acceleration_rad_s2)
        estimated_s = (2 * np.sqrt(distance / acceleration) if distance <= speed * speed / acceleration
                       else distance / speed + speed / acceleration)
        logger.info("TCP reset joint profile: speed=%.3frad/s acceleration=%.3frad/s^2 estimated=%.1fs",
                    speed, acceleration, estimated_s)
        after = self.move_to_joint_target(target_config)
        actual = self.get_eef_pose()
        pos_error = float(np.linalg.norm(actual[:3] - value[:3]))
        rot_error = float((self._rotation.from_matrix(rotation).inv() * self._rotation.from_matrix(rot6d_to_matrix(actual[3:9]))).magnitude())
        if pos_error > position_tolerance_m or rot_error > rotation_tolerance_rad:
            raise RuntimeError(f"TCP reset did not reach the requested pose: position={pos_error:.5f}m rotation={rot_error:.5f}rad")
        logger.info("TCP reset joint reached: q=%s q6_deg=%.2f", np.round(after, 5).tolist(), np.degrees(after[5]))
        return after

    def move_to_reset_joints(
        self,
        joints: np.ndarray,
        *,
        speed: float,
        acceleration: float,
        joint_delta_limits_rad: np.ndarray,
    ) -> np.ndarray:
        """Compatibility entry point with the same exact-branch reset checks."""
        return self.move_to_joint_target(
            JointResetConfig(
                joints_rad=tuple(joints),
                speed_rad_s=speed,
                acceleration_rad_s2=acceleration,
                max_delta_rad=tuple(joint_delta_limits_rad),
            )
        )

    def move_q6_to_reset_branch(
        self,
        q6_target_rad: float,
        *,
        expected_start_q6_rad: float,
        speed: float,
        acceleration: float,
        max_delta_rad: float,
        position_tolerance_rad: float,
        stationary_velocity_tolerance_rad_s: float,
    ) -> np.ndarray:
        """Move only wrist-3 to one explicit, non-periodic reset branch.

        Like :meth:`move_to_reset_joints`, this method does not resolve a
        ``2π``-equivalent target. The caller must provide the reviewed
        unwrapped q6 branch. This prevents a near-limit q6 from being mapped to
        an even more extreme periodic equivalent.
        """
        if self._control is None:
            raise PermissionError("robot adapter is read-only")
        if self._receive.isEmergencyStopped():
            raise RuntimeError("UR emergency stop is active; refusing q6 normalization")
        if self._receive.isProtectiveStopped():
            raise RuntimeError("UR protective stop is active; refusing q6 normalization")
        scalars = {
            "q6 target": q6_target_rad,
            "expected q6 start": expected_start_q6_rad,
            "speed": speed,
            "acceleration": acceleration,
            "max delta": max_delta_rad,
            "position tolerance": position_tolerance_rad,
            "stationary velocity tolerance": stationary_velocity_tolerance_rad_s,
        }
        if any(not np.isfinite(value) for value in scalars.values()):
            raise ValueError("q6 normalization parameters must be finite")
        if any(
            value <= 0
            for name, value in scalars.items()
            if name not in {"q6 target", "expected q6 start"}
        ):
            raise ValueError("q6 normalization limits and motion parameters must be positive")

        current = np.asarray(self._receive.getActualQ(), dtype=np.float64)
        velocity = np.asarray(self._receive.getActualQd(), dtype=np.float64)
        if current.shape != (6,) or not np.all(np.isfinite(current)):
            raise RuntimeError(f"invalid UR joint state: shape={current.shape}")
        if velocity.shape != (6,) or not np.all(np.isfinite(velocity)):
            raise RuntimeError(f"invalid UR joint velocity: shape={velocity.shape}")
        if np.max(np.abs(velocity)) > stationary_velocity_tolerance_rad_s:
            raise RuntimeError(
                "UR is not stationary before q6 normalization: "
                f"max_abs_qd={np.max(np.abs(velocity)):.4f}rad/s"
            )
        start_error = abs(float(current[5] - expected_start_q6_rad))
        if start_error > position_tolerance_rad:
            raise RuntimeError(
                "q6 changed after normalization planning: "
                f"{start_error:.4f}rad > {position_tolerance_rad:.4f}rad"
            )

        delta = float(q6_target_rad - current[5])
        if abs(delta) > max_delta_rad:
            raise RuntimeError(
                f"q6 normalization motion too large: {abs(delta):.4f}rad > {max_delta_rad:.4f}rad"
            )
        target = current.copy()
        target[5] = q6_target_rad
        accepted = self._control.moveJ(
            target.tolist(),
            float(speed),
            float(acceleration),
        )
        if not accepted:
            raise RuntimeError("UR q6 normalization moveJ command was rejected")

        after = np.asarray(self._receive.getActualQ(), dtype=np.float64)
        after_velocity = np.asarray(self._receive.getActualQd(), dtype=np.float64)
        if after.shape != (6,) or not np.all(np.isfinite(after)):
            raise RuntimeError(
                f"invalid UR joint state after q6 normalization: shape={after.shape}"
            )
        if after_velocity.shape != (6,) or not np.all(np.isfinite(after_velocity)):
            raise RuntimeError(
                f"invalid UR joint velocity after q6 normalization: shape={after_velocity.shape}"
            )
        if self._receive.isEmergencyStopped():
            raise RuntimeError("UR emergency stop became active during q6 normalization")
        if self._receive.isProtectiveStopped():
            raise RuntimeError("UR protective stop became active during q6 normalization")
        q6_error = abs(float(after[5] - q6_target_rad))
        held_axis_error = float(np.max(np.abs(after[:5] - current[:5])))
        max_abs_velocity = float(np.max(np.abs(after_velocity)))
        if q6_error > position_tolerance_rad:
            raise RuntimeError(
                "q6 normalization target error too large: "
                f"{q6_error:.4f}rad > {position_tolerance_rad:.4f}rad"
            )
        if held_axis_error > position_tolerance_rad:
            raise RuntimeError(
                "q6 normalization moved a held joint too far: "
                f"{held_axis_error:.4f}rad > {position_tolerance_rad:.4f}rad"
            )
        if max_abs_velocity > stationary_velocity_tolerance_rad_s:
            raise RuntimeError(
                f"UR did not stop after q6 normalization: max_abs_qd={max_abs_velocity:.4f}rad/s"
            )
        return after

    def _send_absolute_action(
        self,
        action: np.ndarray,
        *,
        command_gripper: bool,
        servo: ServoConfig,
    ) -> None:
        if self._control is None:
            raise PermissionError("robot adapter is read-only")
        if self._receive.isEmergencyStopped():
            raise RuntimeError("UR emergency stop is active; refusing motion")
        if self._receive.isProtectiveStopped():
            raise RuntimeError("UR protective stop is active; refusing motion")
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (10,) or not np.all(np.isfinite(value)):
            raise ValueError("action must be a finite 10D vector")
        rotvec = self._rotation.from_matrix(rot6d_to_matrix(value[3:9])).as_rotvec()
        native_pose = np.concatenate((value[:3], rotvec)).tolist()
        accepted = self._control.servoL(
            native_pose,
            servo.acceleration,
            servo.velocity,
            servo.time_step,
            servo.lookahead_time,
            servo.gain,
        )
        if not accepted:
            raise RuntimeError("UR servoL command was rejected")
        if command_gripper and self._gripper:
            self._gripper.set_open_fraction(float(value[9]), speed=self._gripper_speed)

    def recover_protective_stop(self) -> bool:
        """Explicitly unlock one active Protective Stop and restore RTDE control.

        This method never starts a saved PolyScope program and never sends a
        motion command. Callers must separately opt in and bound retries.
        """
        if self._control is None:
            raise PermissionError("robot adapter is read-only")
        if self._receive.isEmergencyStopped():
            raise RuntimeError("UR emergency stop is active; refusing recovery")
        if not self._receive.isProtectiveStopped():
            return False
        try:
            import dashboard_client
        except ImportError as error:
            raise RuntimeError("UR dashboard dependency is unavailable") from error

        try:
            self._control.servoStop()
        except Exception:
            pass
        try:
            self._control.stopScript()
        except Exception:
            pass
        dashboard = dashboard_client.DashboardClient(self._robot_ip)
        try:
            dashboard.connect(2000)
            if not dashboard.isInRemoteControl():
                raise RuntimeError("UR is not in remote control mode; refusing reset recovery")
            dashboard.closeSafetyPopup()
            dashboard.unlockProtectiveStop()
        finally:
            dashboard.disconnect()
        time.sleep(0.5)
        if self._receive.isProtectiveStopped():
            raise RuntimeError("UR protective stop remains active after reset recovery")
        if hasattr(self._control, "reuploadScript"):
            self._control.reuploadScript()
            time.sleep(0.2)
        return True

    def open_gripper(self) -> None:
        """Open the gripper before an episode reset."""
        if self._control is None:
            raise PermissionError("robot adapter is read-only")
        if self._gripper is None:
            raise RuntimeError("episode reset requires a configured gripper")
        if self._receive.isEmergencyStopped():
            raise RuntimeError("UR emergency stop is active; refusing gripper reset")
        if self._receive.isProtectiveStopped():
            raise RuntimeError("UR protective stop is active; refusing gripper reset")
        self._gripper.set_open_fraction(1.0, speed=self._gripper_speed)

    def stop(self) -> None:
        """Stop servo mode without clearing safety faults."""
        try:
            if self._control is not None:
                self._control.servoStop()
        finally:
            if isinstance(self._gripper, AsyncRobotiqTCP):
                self._gripper.cancel_pending()

    def close(self) -> None:
        """Stop and disconnect all interfaces."""
        try:
            self.stop()
        finally:
            if self._control is not None:
                self._control.disconnect()
            self._receive.disconnect()
            if self._gripper:
                self._gripper.close()
