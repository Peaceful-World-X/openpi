"""Keep the controller TCP reference consistent with the training dataset."""
from __future__ import annotations

import logging
import socket
import struct
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def validate_tcp_offset(value: Any) -> np.ndarray:
    offset = np.asarray(value, dtype=np.float64)
    if offset.shape != (6,) or not np.isfinite(offset).all():
        raise ValueError("robot.tcp_offset must be six finite values: xyz in metres, rotation vector in radians")
    return offset


def read_tcp_offset(robot_ip: str, *, timeout_s: float = 3.0) -> np.ndarray:
    """Read CartesianInfo on port 30001 without uploading a control script."""
    deadline = time.monotonic() + timeout_s
    buffer = b""
    with socket.create_connection((robot_ip, 30001), timeout=timeout_s) as connection:
        while time.monotonic() < deadline:
            connection.settimeout(max(0.001, deadline - time.monotonic()))
            block = connection.recv(65536)
            if not block:
                raise RuntimeError("UR primary interface closed before reporting the TCP offset")
            buffer += block
            while len(buffer) >= 5:
                length = struct.unpack_from("!I", buffer)[0]
                if not 5 <= length <= 2_000_000:
                    raise RuntimeError("invalid UR primary packet length")
                if len(buffer) < length:
                    break
                packet, buffer = buffer[:length], buffer[length:]
                if packet[4] != 16:  # RobotState
                    continue
                cursor = 5
                while cursor + 5 <= length:
                    size, kind = struct.unpack_from("!IB", packet, cursor)
                    if size < 5 or cursor + size > length:
                        raise RuntimeError("invalid UR state subpackage length")
                    if kind == 4 and size >= 101:  # CartesianInfo: pose then TCP offset
                        return validate_tcp_offset(struct.unpack_from("!6d", packet, cursor + 5 + 48))
                    cursor += size
    raise TimeoutError("UR did not report the active TCP offset")


def configure_tcp(control: Any, receive: Any, robot_ip: str, expected: Any) -> None:
    """Apply an explicit reference only when armed; read-only clients just verify."""
    expected = validate_tcp_offset(expected)
    actual = validate_tcp_offset(control.getTCPOffset() if control is not None else read_tcp_offset(robot_ip))
    if not np.allclose(actual, expected, atol=1e-6, rtol=0):
        if control is None:
            raise RuntimeError(
                f"TCP reference mismatch: controller={actual.tolist()}, configured={expected.tolist()}. "
                "Read-only mode does not change TCP; set the configured TCP on the teach pendant first."
            )
        if receive.isEmergencyStopped() or receive.isProtectiveStopped():
            raise RuntimeError("cannot change TCP while UR safety stop is active")
        qd = np.asarray(receive.getActualQd(), dtype=np.float64)
        if qd.shape != (6,) or not np.isfinite(qd).all() or np.max(np.abs(qd)) > 0.001:
            raise RuntimeError("UR must be stationary before changing the configured TCP reference")
        logger.info("TCP reference: controller=%s -> configured=%s", actual.tolist(), expected.tolist())
        if not control.setTcp(expected.tolist()):
            raise RuntimeError("UR rejected the configured TCP offset")
        actual = validate_tcp_offset(control.getTCPOffset())
        if not np.allclose(actual, expected, atol=1e-6, rtol=0):
            raise RuntimeError(f"TCP offset verification failed: actual={actual.tolist()}, expected={expected.tolist()}")
    logger.info("TCP reference verified: %s (xyz metres, rotation vector radians)", actual.tolist())
