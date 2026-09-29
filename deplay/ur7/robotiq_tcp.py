"""Small Robotiq text-protocol client for the UR tool-port server."""

from __future__ import annotations

import socket
import threading

import numpy as np


class RobotiqTCP:
    """Read and optionally command a Robotiq gripper over TCP port 63352."""

    def __init__(self, connection: str, *, writable: bool, timeout_s: float = 1.0):
        host, separator, port = connection.partition(":")
        self._address = (host, int(port) if separator else 63352)
        self._writable = writable
        self._socket = socket.create_connection(self._address, timeout=timeout_s)
        self._socket.settimeout(timeout_s)
        self._lock = threading.Lock()

    def _request(self, command: str) -> str:
        with self._lock:
            self._socket.sendall(f"{command}\n".encode())
            return self._socket.recv(1024).decode().strip()

    def get_open_fraction(self) -> float:
        """Return normalized opening, where 1 is open and 0 is closed."""
        response = self._request("GET POS").split()
        if len(response) != 2 or response[0] != "POS":
            raise RuntimeError(f"unexpected Robotiq response: {' '.join(response)}")
        return 1.0 - int(response[1]) / 255.0

    def set_open_fraction(self, value: float, *, speed: float = 0.3, force: float = 0.5) -> None:
        """Command a normalized opening. Requires a writable instance."""
        if not self._writable:
            raise PermissionError("Robotiq client is read-only")
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("gripper open fraction must be finite and in [0, 1]")
        raw_position = int(round((1.0 - value) * 255))
        raw_speed = int(round(np.clip(speed, 0.0, 1.0) * 255))
        raw_force = int(round(np.clip(force, 0.0, 1.0) * 255))
        response = self._request(f"SET POS {raw_position} SPE {raw_speed} FOR {raw_force} GTO 1")
        if response != "ack":
            raise RuntimeError(f"Robotiq command was not acknowledged: {response}")

    def close(self) -> None:
        """Close the socket."""
        self._socket.close()
