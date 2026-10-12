"""Small Robotiq text-protocol client for the UR tool-port server."""

from __future__ import annotations

import socket
import threading
import time

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


class AsyncRobotiqTCP:
    """One socket owner, fresh measured state, and a latest-only command slot.

    Used only by policy deployment; replay retains synchronous gripper I/O.
    No extrapolated gripper state is returned to the policy.
    """

    def __init__(self, connection: str, *, writable: bool, poll_s: float = 0.05,
                 max_state_age_s: float = 0.5):
        self._client = RobotiqTCP(connection, writable=writable)
        self._writable = writable
        self._poll_s = poll_s
        self._max_state_age_s = max_state_age_s
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._closed = threading.Event()
        self._pending = None
        self._last_command = None
        self._error = None
        try:
            self._position = self._client.get_open_fraction()
        except BaseException:
            self._client.close()
            raise
        self._updated = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="ur7-gripper-io", daemon=True)
        self._thread.start()

    def _check(self) -> None:
        if self._error is not None:
            raise RuntimeError("background gripper I/O failed") from self._error
        if self._closed.is_set():
            raise RuntimeError("gripper is closed")
        if time.monotonic() - self._updated > self._max_state_age_s:
            raise RuntimeError("gripper measured state is stale")

    def get_open_fraction(self) -> float:
        with self._lock:
            self._check()
            return self._position

    def set_open_fraction(self, value: float, *, speed: float = 0.3, force: float = 0.5) -> None:
        if not self._writable:
            raise PermissionError("Robotiq client is read-only")
        if not np.isfinite([value, speed, force]).all() or not 0 <= value <= 1:
            raise ValueError("invalid gripper command")
        # Coalesce identical hardware targets, including sub-count model noise.
        command = (round((1.0 - value) * 255), round(np.clip(speed, 0, 1) * 255),
                   round(np.clip(force, 0, 1) * 255))
        with self._lock:
            self._check()
            self._pending = command

    def _run(self) -> None:
        try:
            while not self._closed.is_set():
                with self._io_lock:
                    with self._lock:
                        if self._closed.is_set():
                            return
                        command, self._pending = self._pending, None
                    if command is not None and command != self._last_command:
                        position, speed, force = command
                        self._client.set_open_fraction(1 - position / 255, speed=speed / 255, force=force / 255)
                        self._last_command = command
                    position = self._client.get_open_fraction()
                    with self._lock:
                        self._position = position
                        self._updated = time.monotonic()
                self._closed.wait(self._poll_s)
        except Exception as error:
            with self._lock:
                self._error = error

    def cancel_pending(self) -> None:
        # Serialize with a possible in-flight command. Once returned, no
        # command queued before stop can be sent later by the worker.
        with self._io_lock:
            with self._lock:
                self._pending = None

    def close(self) -> None:
        self._closed.set()
        self.cancel_pending()
        self._thread.join(timeout=2.5)
        self._client.close()
