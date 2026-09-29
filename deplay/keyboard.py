"""Non-blocking terminal input for the interactive UR7 client."""

from __future__ import annotations

import os
import select
import sys
import termios
import tty


# 在不阻塞控制循环的情况下读取回车、空格和归位快捷键。
class TerminalKeyReader:
    """Read single key presses while restoring the terminal on exit."""

    def __init__(self, stream=None):
        self._stream = stream or sys.stdin
        self._fd = self._stream.fileno()
        self._old_settings = None

    def __enter__(self) -> "TerminalKeyReader":
        if not os.isatty(self._fd):
            raise RuntimeError("interactive mode requires a TTY; use --non-interactive for pipes")
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def poll(self) -> list[str]:
        """Return all currently available normalized keys without waiting."""
        keys: list[str] = []
        while True:
            ready, _, _ = select.select([self._stream], [], [], 0.0)
            if not ready:
                return keys
            value = os.read(self._fd, 1)
            if not value:
                return keys
            char = value.decode("utf-8", errors="ignore")
            if char in {"\r", "\n"}:
                keys.append("enter")
            elif char == " ":
                keys.append("space")
            elif char.lower() == "h":
                keys.append("h")
            elif char in {"q", "Q", "\x03"}:
                keys.append("quit")

    def close(self) -> None:
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None
