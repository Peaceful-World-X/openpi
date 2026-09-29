"""Asynchronous action-chunk scheduling for an OpenPI robot client.

The official OpenPI WebSocket protocol is request/response. This module keeps
one request in flight while the robot consumes the current execution window.
It supports asynchronous prefetch without fusion and a time-aligned RTC mode.
Both modes wait for the old window to finish before installing the response;
RTC only blends actions that refer to the same time indices.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any

import numpy as np

from .ur7.se3 import fuse_chunk_boundary
from .ur7.se3 import interpolate_absolute_action

logger = logging.getLogger(__name__)

ObservationProvider = Callable[[], Mapping[str, Any]]
PrefixAttentionSchedule = str


def fuse_aligned_chunks(old_chunk: np.ndarray, new_chunk: np.ndarray, overlap_steps: int) -> np.ndarray:
    """Fuse time-aligned old/new absolute action windows at their boundary."""
    old = np.asarray(old_chunk, dtype=np.float64)
    new = np.asarray(new_chunk, dtype=np.float64)
    if old.ndim != 2 or new.ndim != 2 or old.shape[1:] != (10,) or new.shape[1:] != (10,):
        raise ValueError(f"chunks must have shape (N, 10), got {old.shape} and {new.shape}")
    result = new.copy()
    overlap = min(max(0, int(overlap_steps)), len(old), len(result))
    for index in range(overlap):
        result[index] = interpolate_absolute_action(
            old[index], result[index], (index + 1) / (overlap + 1)
        )
    return result


def get_prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: PrefixAttentionSchedule = "exp",
) -> np.ndarray:
    """Return the RTC prefix weights used by ``cybopal_policy_rtc.py``.

    Weight 1 means that the old chunk is retained; weight 0 means that the
    new chunk is used.  The returned vector has length ``total``.
    """
    if total <= 0 or start < 0 or end < 0:
        raise ValueError("RTC weight bounds must be non-negative and total must be positive")
    start = min(start, end)
    index = np.arange(total, dtype=np.float64)
    if schedule == "ones":
        weights = np.ones(total, dtype=np.float64)
    elif schedule == "zeros":
        weights = (index < start).astype(np.float64)
    elif schedule in {"linear", "exp"}:
        weights = np.clip(
            (float(start - 1) - index) / float(end - start + 1) + 1.0,
            0.0,
            1.0,
        )
        if schedule == "exp":
            weights = weights * np.expm1(weights) / (np.e - 1.0)
    else:
        raise ValueError(f"invalid RTC prefix attention schedule: {schedule}")
    return np.where(index >= end, 0.0, weights).astype(np.float64)


def rtc_merge_chunk(
    new_chunk: np.ndarray,
    left_over: np.ndarray | None,
    *,
    inference_delay_steps: int,
    execute_horizon: int,
    action_chunk_size: int,
    prefix_attention_schedule: PrefixAttentionSchedule = "exp",
    pose_aware: bool = True,
) -> np.ndarray:
    """Merge one newly inferred chunk using the Cybopal RTC partition.

    For ``H=action_chunk_size`` and ``s=execute_horizon`` the output is:

    ``old[0:d] | blend(old[d:H-s], new[d:H-s]) | new[H-s:H]``.

    ``left_over`` is the snapshot captured when inference started.  This is
    the essential difference from blending only the already executed tail.
    Absolute UR7 pose actions use SE(3) interpolation in the blend region;
    setting ``pose_aware=False`` reproduces the reference elementwise formula.
    """
    value = np.asarray(new_chunk, dtype=np.float64)
    if value.ndim == 1:
        value = value.reshape(1, -1)
    if value.ndim != 2 or value.shape[1] != 10:
        raise ValueError(f"new_chunk must have shape (N, 10), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("new_chunk contains NaN or Inf")
    d = int(inference_delay_steps)
    s = int(execute_horizon)
    horizon = int(action_chunk_size)
    old = None if left_over is None else np.asarray(left_over, dtype=np.float64)
    if old is not None and old.ndim == 1:
        old = old.reshape(1, -1)
    if old is not None and (old.ndim != 2 or old.shape[1] != 10 or not np.isfinite(old).all()):
        raise ValueError(f"left_over must have shape (N, 10), got {old.shape}")
    if old is None or old.shape[0] < d or d <= 0 or s <= 0 or horizon <= s or value.shape[0] < horizon:
        return value.copy()

    blend_end = horizon - s
    weights = get_prefix_weights(d, blend_end, horizon, prefix_attention_schedule)
    rows: list[np.ndarray] = [old[index].copy() for index in range(d)]
    for index in range(d, blend_end):
        new_row = value[index]
        if index >= old.shape[0]:
            rows.append(new_row.copy())
            continue
        old_weight = float(weights[index])
        if pose_aware:
            rows.append(interpolate_absolute_action(old[index], new_row, 1.0 - old_weight))
        else:
            rows.append(old_weight * old[index] + (1.0 - old_weight) * new_row)
    rows.extend(value[blend_end:horizon].copy())
    result = np.stack(rows) if rows else value.copy()
    if result.shape != (horizon, 10):
        raise RuntimeError(f"RTC merge returned unexpected shape: {result.shape}")
    return result


@dataclass(frozen=True)
class ChunkStepInfo:
    """Scheduling evidence for one action returned by :meth:`step`."""

    chunk_id: int
    chunk_index: int
    queue_remaining: int
    prefetch_started: bool
    prefetch_hit: bool
    wait_latency_s: float
    fused_overlap_steps: int


class AsyncChunkClient:
    """Consume multiple server chunks while prefetching the next one."""

    def __init__(
        self,
        policy: Any,
        *,
        prefetch_remaining_steps: int = 3,
        rtc_overlap_steps: int = 2,
        max_chunk_steps: int | None = None,
        rtc_enabled: bool = True,
        action_chunk_size: int | None = None,
        execute_horizon: int | None = None,
        inference_delay_steps: int = 0,
        prefix_attention_schedule: PrefixAttentionSchedule = "exp",
        pose_aware_rtc: bool = False,
    ) -> None:
        if prefetch_remaining_steps < 0:
            raise ValueError("prefetch_remaining_steps must be non-negative")
        if rtc_overlap_steps < 0:
            raise ValueError("rtc_overlap_steps must be non-negative")
        if max_chunk_steps is not None and max_chunk_steps <= 0:
            raise ValueError("max_chunk_steps must be positive")
        if action_chunk_size is not None and action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive")
        if action_chunk_size is not None and execute_horizon is None:
            raise ValueError("execute_horizon is required for reference RTC")
        if action_chunk_size is not None and not 0 < execute_horizon < action_chunk_size:
            raise ValueError("execute_horizon must be in (0, action_chunk_size)")
        if inference_delay_steps < 0:
            raise ValueError("inference_delay_steps must be non-negative")
        if prefix_attention_schedule not in {"linear", "exp", "ones", "zeros"}:
            raise ValueError("invalid prefix_attention_schedule")
        self._policy = policy
        self._prefetch_remaining_steps = int(prefetch_remaining_steps)
        self._rtc_overlap_steps = int(rtc_overlap_steps)
        self._max_chunk_steps = max_chunk_steps
        self._rtc_enabled = bool(rtc_enabled)
        self._action_chunk_size = action_chunk_size
        self._execute_horizon = execute_horizon
        self._inference_delay_steps = int(inference_delay_steps)
        self._prefix_attention_schedule = prefix_attention_schedule
        self._pose_aware_rtc = bool(pose_aware_rtc)
        self._reference_rtc = action_chunk_size is not None and self._rtc_enabled
        # A 16-step window with a 3-step inference delay starts the next
        # request after 13 actions, leaving exactly 3 old actions to execute.
        self._trigger_remaining_steps = int(prefetch_remaining_steps)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-prefetch")
        self._lock = threading.RLock()
        self._future: Future[np.ndarray] | None = None
        self._pending_left_over: np.ndarray | None = None
        self._pending_source: np.ndarray | None = None
        self._pending_source_alignment = 0
        self._pending_started_s: float | None = None
        self._generation = 0
        self._current: np.ndarray | None = None
        self._cursor = 0
        self._source: np.ndarray | None = None
        self._source_offset = 0
        self._chunk_id = -1
        self._last_executed: deque[np.ndarray] = deque(maxlen=max(1, rtc_overlap_steps))
        self._last_info: ChunkStepInfo | None = None
        self._installed_overlap = 0
        self._closed = False
        self._prefetch_started_count = 0
        self._prefetch_hit_count = 0
        self._rtc_merge_count = 0

    @property
    def last_step_info(self) -> ChunkStepInfo | None:
        return self._last_info

    @property
    def queue_remaining(self) -> int:
        with self._lock:
            return 0 if self._current is None else max(0, len(self._current) - self._cursor)

    @property
    def prefetch_started_count(self) -> int:
        return self._prefetch_started_count

    @property
    def prefetch_hit_count(self) -> int:
        return self._prefetch_hit_count

    @property
    def rtc_merge_count(self) -> int:
        return self._rtc_merge_count

    def _fetch(self, observation: Mapping[str, Any], generation: int) -> np.ndarray:
        if generation != self._generation:
            raise RuntimeError("stale action-chunk request")
        chunk = np.asarray(self._policy.infer_chunk(observation), dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != 10 or chunk.shape[0] == 0:
            raise ValueError(f"server chunk must have shape (N, 10), got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("server chunk contains NaN or Inf")
        if self._max_chunk_steps is not None:
            chunk = chunk[: self._max_chunk_steps]
        if chunk.shape[0] == 0:
            raise ValueError("server chunk became empty after max_chunk_steps")
        return chunk

    def _install_window(
        self,
        window: np.ndarray,
        *,
        source: np.ndarray,
        source_offset: int,
        from_prefetch: bool,
    ) -> None:
        """Install an execution window and retain its raw source chunk."""
        value = np.asarray(window, dtype=np.float64)
        if value.ndim != 2 or value.shape[1:] != (10,) or len(value) == 0:
            raise ValueError(f"execution window must have shape (N, 10), got {value.shape}")
        self._current = value
        self._cursor = 0
        self._source = np.asarray(source, dtype=np.float64).copy()
        self._source_offset = int(source_offset)
        self._chunk_id += 1
        self._last_executed.clear()
        if from_prefetch:
            self._prefetch_hit_count += 1
        self._installed_overlap = 0

    def initialize(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Synchronously fetch the first chunk for an episode."""
        with self._lock:
            self._ensure_open()
            if self._current is not None or self._future is not None:
                raise RuntimeError("chunk client is already initialized")
            chunk = self._fetch(observation, self._generation)
            window = chunk[: self._action_chunk_size] if self._action_chunk_size else chunk
            self._install_window(window, source=chunk, source_offset=0, from_prefetch=False)
            return chunk.copy()

    def _start_prefetch(self, observation: Mapping[str, Any]) -> bool:
        if self._future is not None:
            return False
        generation = self._generation
        if self._current is None:
            raise RuntimeError("cannot prefetch without a current chunk")
        if self._source is None:
            raise RuntimeError("cannot prefetch without a source chunk")
        self._pending_left_over = self._current[self._cursor :].copy()
        self._pending_source = self._source.copy()
        self._pending_source_alignment = self._source_offset + self._cursor + self._inference_delay_steps
        self._pending_started_s = time.monotonic()
        self._future = self._executor.submit(self._fetch, dict(observation), generation)
        self._prefetch_started_count += 1
        return True

    def prefetch(self, observation: Mapping[str, Any]) -> bool:
        """Start a request using an observation captured after action execution.

        The reference controller samples the robot/camera after the control
        step that crossed the RTC trigger.  ``UR7DeploymentClient`` uses this
        method so the planning observation and the leftover snapshot share
        that same timeline boundary.
        """
        with self._lock:
            self._ensure_open()
            if self._current is None or self._future is not None:
                return False
            remaining = len(self._current) - self._cursor
            if remaining > self._trigger_remaining_steps:
                return False
            return self._start_prefetch(observation)

    def _take_prefetched(self) -> tuple[float, bool, int]:
        """Wait for and install/merge the next chunk."""
        future = self._future
        if future is None:
            raise RuntimeError("action queue is empty and no prefetch is pending")
        was_ready = future.done()
        started = time.monotonic()
        self._future = None
        chunk = future.result()
        wait_latency = time.monotonic() - started
        delay = self._inference_delay_steps
        window_size = self._action_chunk_size or len(chunk)
        new_offset = min(delay, max(0, len(chunk) - 1))
        new_window = chunk[new_offset : new_offset + window_size]
        if len(new_window) == 0:
            raise RuntimeError("new action chunk is empty after inference delay")
        overlap = 0
        if self._reference_rtc and self._pending_source is not None:
            old_window = self._pending_source[
                self._pending_source_alignment : self._pending_source_alignment + len(new_window)
            ]
            if len(old_window):
                overlap = min(self._rtc_overlap_steps, len(old_window), len(new_window))
                new_window = fuse_aligned_chunks(old_window, new_window, overlap)
                self._rtc_merge_count += 1
        elif self._action_chunk_size is None and self._rtc_enabled and self._last_executed and self._rtc_overlap_steps:
            # Preserve the legacy unwindowed scheduler behavior for callers
            # that do not configure fixed action windows.
            old_tail = np.stack(tuple(self._last_executed)[-self._rtc_overlap_steps :])
            overlap = min(len(old_tail), len(new_window))
            new_window = fuse_chunk_boundary(old_tail, new_window, overlap)
        self._install_window(new_window, source=chunk, source_offset=new_offset, from_prefetch=True)
        self._pending_left_over = None
        self._pending_source = None
        self._pending_started_s = None
        self._installed_overlap = overlap
        return wait_latency, was_ready, overlap

    def step(
        self,
        observation: Mapping[str, Any],
        *,
        prefetch_observation: ObservationProvider | None = None,
    ) -> tuple[np.ndarray, ChunkStepInfo]:
        """Return one action and schedule the next chunk when the queue is low.

        ``prefetch_observation`` is called in the control thread only to
        capture a planning observation; the network/model call runs in the
        worker.  It should therefore be a cheap latest-frame/state reader.
        """
        with self._lock:
            self._ensure_open()
            wait_latency = 0.0
            prefetch_hit = False
            fused_overlap = getattr(self, "_installed_overlap", 0)
            if self._current is None:
                self.initialize(observation)
            if self._current is not None and self._cursor >= len(self._current):
                if self._future is None:
                    raise RuntimeError("action chunk exhausted without a pending prefetch")
                wait_latency, prefetch_hit, fused_overlap = self._take_prefetched()
            assert self._current is not None
            action = self._current[self._cursor].copy()
            chunk_id = self._chunk_id
            chunk_index = self._cursor
            self._cursor += 1
            self._last_executed.append(action.copy())
            remaining = len(self._current) - self._cursor
            prefetch_started = False
            if self._future is None and prefetch_observation is not None and remaining <= self._trigger_remaining_steps:
                next_observation = prefetch_observation()
                prefetch_started = self._start_prefetch(next_observation)
            info = ChunkStepInfo(
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                queue_remaining=remaining,
                prefetch_started=prefetch_started,
                prefetch_hit=prefetch_hit,
                wait_latency_s=wait_latency,
                fused_overlap_steps=fused_overlap,
            )
            self._last_info = info
            return action, info

    def reset(self) -> None:
        """Discard all chunks and invalidate in-flight results."""
        with self._lock:
            self._generation += 1
            future, self._future = self._future, None
            if future is not None:
                future.cancel()
            self._pending_left_over = None
            self._pending_source = None
            self._pending_started_s = None
            self._current = None
            self._cursor = 0
            self._source = None
            self._source_offset = 0
            self._chunk_id = -1
            self._last_executed.clear()
            self._last_info = None
            self._installed_overlap = 0
            reset = getattr(self._policy, "reset", None)
            if callable(reset):
                reset()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            future, self._future = self._future, None
            if future is not None:
                future.cancel()
            self._pending_left_over = None
            self._pending_started_s = None
        self._executor.shutdown(wait=True, cancel_futures=True)
        close = getattr(self._policy, "close", None)
        if callable(close):
            close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("chunk client is closed")
