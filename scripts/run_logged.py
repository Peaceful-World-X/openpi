"""Stream a command to the console and a timestamped log using only the stdlib."""

from __future__ import annotations

import contextlib
import datetime as dt
import os
from pathlib import Path
import signal
import subprocess
import sys


def main() -> int:
    kind, *command = sys.argv[1:]
    if kind not in {"submit", "train"} or not command:
        raise ValueError("Usage: run_logged.py {submit,train} COMMAND [ARGS...]")
    log_dir = Path(os.environ.get("OPENPI_LOG_DIR", "/ossfs/workspace/.log"))
    log_dir.mkdir(parents=True, exist_ok=True)
    # The image's Python can be 3.8, before the Python 3.11 runtime is installed.
    timestamp = dt.datetime.now(tz=dt.timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S_%f")  # noqa: UP017
    log_path = log_dir / f"{timestamp}_{kind}_{os.getpid()}.log"
    environment = dict(os.environ, **{f"_OPENPI_LOG_{kind.upper()}": "1"}, PYTHONUNBUFFERED="1")
    process = None
    pending_signal = None

    def forward_signal(signum, _frame):
        nonlocal pending_signal
        pending_signal = signum
        if process is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)
    with log_path.open("xb") as log:
        console_open = True

        def emit(data: bytes) -> None:
            nonlocal console_open
            log.write(data)
            log.flush()
            if console_open:
                try:
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                except BrokenPipeError:
                    # Keep the file log if the terminal disconnects.
                    console_open = False
                    with open(os.devnull, "wb") as sink:
                        os.dup2(sink.fileno(), sys.stdout.fileno())

        emit(f"Log: {log_path}\n".encode())
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment, start_new_session=True
        ) as process:
            if pending_signal is not None:
                forward_signal(pending_signal, None)
            try:
                while data := os.read(process.stdout.fileno(), 65536):
                    emit(data)
                status = process.wait()
            except BaseException:
                forward_signal(signal.SIGTERM, None)
                raise
        status = 128 + pending_signal if pending_signal is not None else (128 - status if status < 0 else status)
        emit(f"Exit status: {status}\n".encode())
        return status


if __name__ == "__main__":
    sys.exit(main())
