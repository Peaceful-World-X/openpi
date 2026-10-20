"""WebSocket policy client for the OpenPI deployment process.

OpenPI's server protocol is one request followed by one response on a
persistent WebSocket.  This wrapper keeps that protocol synchronous and
thread-safe; :mod:`deplay.async_chunk_client` performs the asynchronous
prefetch in a worker thread so an arm loop never shares a socket unsafely.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from contextlib import suppress
import copy
import logging
import threading
import time
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)


class WebSocketPolicyClient:
    """Persistent OpenPI WebSocket connection with bounded reconnects."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        api_key: str | None = None,
        uri: str | None = None,
        connect_timeout_s: float = 10.0,
        timeout_s: float = 10.0,
        reconnect_attempts: int = 2,
        metadata_validator: Any | None = None,
        debug_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if connect_timeout_s <= 0 or timeout_s <= 0:
            raise ValueError("connect_timeout_s and timeout_s must be positive")
        if reconnect_attempts < 0:
            raise ValueError("reconnect_attempts must be non-negative")
        self._uri = uri or (host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}")
        self._api_key = api_key
        self._connect_timeout_s = float(connect_timeout_s)
        self._timeout_s = float(timeout_s)
        self._reconnect_attempts = int(reconnect_attempts)
        self._metadata_validator = metadata_validator
        self._debug_callback = debug_callback
        self._debug_request_id = 0
        self._packer = msgpack_numpy.Packer()
        self._lock = threading.RLock()
        self._ws: Any | None = None
        self._server_metadata: dict[str, Any] = {}
        self._closed = False
        self.last_action_timestamp_s: float | None = None
        self.last_server_timing: dict[str, Any] = {}
        self._connect()

    @property
    def server_metadata(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._server_metadata)

    @property
    def uri(self) -> str:
        return self._uri

    def _connect(self) -> None:
        import websockets.sync.client

        deadline = time.monotonic() + self._connect_timeout_s
        last_error: BaseException | None = None
        while not self._closed:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                kwargs = {
                    "compression": None,
                    "max_size": None,
                    "open_timeout": min(self._timeout_s, max(0.1, deadline - time.monotonic())),
                }
                if headers is not None:
                    # websockets 11 called this argument ``extra_headers``;
                    # recent versions renamed it to ``additional_headers``.
                    kwargs["additional_headers"] = headers
                try:
                    connection = websockets.sync.client.connect(self._uri, **kwargs)
                except TypeError as error:
                    if "additional_headers" not in str(error):
                        raise
                    kwargs.pop("additional_headers", None)
                    if headers is not None:
                        kwargs["extra_headers"] = headers
                    connection = websockets.sync.client.connect(self._uri, **kwargs)
                metadata_payload = connection.recv(timeout=self._timeout_s)
                if isinstance(metadata_payload, str):
                    raise RuntimeError(f"policy server metadata error: {metadata_payload}")
                metadata = msgpack_numpy.unpackb(metadata_payload)
                if not isinstance(metadata, Mapping):
                    raise RuntimeError("policy server metadata must be a mapping")
                metadata = dict(metadata)
                if self._metadata_validator is not None:
                    self._metadata_validator(metadata)
                old = self._ws
                self._ws = connection
                self._server_metadata = metadata
                if old is not None:
                    try:
                        old.close()
                    except Exception:
                        logger.debug("failed to close stale policy socket", exc_info=True)
                logger.info("connected to OpenPI policy server %s", self._uri)
                return
            except BaseException as error:
                last_error = error
                with suppress(Exception):
                    connection.close()  # type: ignore[union-attr]
                if time.monotonic() >= deadline:
                    break
                time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))
        raise TimeoutError(f"could not connect to OpenPI policy server {self._uri}") from last_error

    def _reconnect(self) -> None:
        old = self._ws
        self._ws = None
        if old is not None:
            with suppress(Exception):
                old.close()
        self._connect()

    def infer(
        self,
        observation: Mapping[str, Any],
        *,
        rtc_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send one observation and return the server's decoded action dict.

        RTC requests use a nested envelope so the server can distinguish the
        execution-space action prefix from ordinary observation fields.  Keep
        the historical flat observation payload when no context is supplied.
        """
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        if rtc_context is not None and not isinstance(rtc_context, Mapping):
            raise TypeError("rtc_context must be a mapping")
        observation_payload = dict(observation)
        context_payload = None if rtc_context is None else dict(rtc_context)
        request = observation_payload if context_payload is None else {
            "observation": observation_payload,
            "rtc_context": context_payload,
        }
        payload = self._packer.pack(request)
        with self._lock:
            if self._closed:
                raise RuntimeError("policy client is closed")
            last_error: BaseException | None = None
            for attempt in range(self._reconnect_attempts + 1):
                request_id = self._debug_request_id
                self._debug_request_id += 1
                started_s = time.monotonic()
                debug_event: dict[str, Any] = {
                    "event_type": "inference",
                    "request_id": request_id,
                    "attempt": attempt,
                    "started_monotonic_s": started_s,
                    "uri": self._uri,
                    "observation": observation_payload,
                }
                if context_payload is not None:
                    debug_event["rtc_context"] = context_payload
                try:
                    if self._ws is None:
                        self._connect()
                    assert self._ws is not None
                    self._ws.send(payload)
                    response = self._ws.recv(timeout=self._timeout_s)
                    if isinstance(response, str):
                        raise RuntimeError(f"policy server inference error:\n{response}")
                    result = msgpack_numpy.unpackb(response)
                    if not isinstance(result, Mapping):
                        raise RuntimeError("policy server response must be a mapping")
                    result = dict(result)
                    self.last_action_timestamp_s = time.monotonic()
                    timing = result.get("server_timing", {})
                    self.last_server_timing = dict(timing) if isinstance(timing, Mapping) else {}
                    debug_event.update(
                        {
                            "ok": True,
                            "finished_monotonic_s": time.monotonic(),
                            "response": result,
                        }
                    )
                    self._emit_debug(debug_event)
                    return result
                except BaseException as error:
                    debug_event.update(
                        {
                            "ok": False,
                            "finished_monotonic_s": time.monotonic(),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    self._emit_debug(debug_event)
                    last_error = error
                    if attempt >= self._reconnect_attempts:
                        break
                    logger.warning(
                        "policy request failed; reconnecting (%d/%d): %s", attempt + 1, self._reconnect_attempts, error
                    )
                    self._reconnect()
            raise RuntimeError("OpenPI policy inference failed") from last_error

    def _emit_debug(self, event: dict[str, Any]) -> None:
        """Send one request event to the optional debug recorder safely."""
        if self._debug_callback is None:
            return
        try:
            self._debug_callback(event)
        except Exception:
            logger.warning("failed to write policy debug event", exc_info=True)

    def infer_chunk(
        self,
        observation: Mapping[str, Any],
        *,
        action_dim: int = 10,
        rtc_context: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Infer and strictly validate an absolute canonical action chunk."""
        result = self.infer(observation, rtc_context=rtc_context)
        if "actions" not in result:
            raise ValueError("policy response has no 'actions' field")
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != action_dim or actions.shape[0] == 0:
            raise ValueError(f"policy actions must have shape (N, {action_dim}), got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("policy actions contain NaN or Inf")
        return actions

    def reset(self) -> None:
        """Reset local timing state; OpenPI's server protocol has no reset RPC."""
        with self._lock:
            self.last_action_timestamp_s = None
            self.last_server_timing = {}

    def close(self) -> None:
        with self._lock:
            self._closed = True
            connection, self._ws = self._ws, None
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    logger.debug("failed to close policy socket", exc_info=True)
