"""Action queue with two scheduling parameters: reinfer and blend.

RTC requests a new chunk when `reinfer` actions remain. A completed response
replaces the queue at the next control step; actions already consumed since
its observation are skipped. The first `blend` remaining actions are then
blended with the time-aligned old plan. This is client-side trajectory blending,
not a modification of the model's denoising/inference algorithm.

Without RTC, each synchronous request supplies exactly `reinfer` actions.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Any

import numpy as np
from .ur7.se3 import interpolate_absolute_action

logger = logging.getLogger(__name__)


def validate_rtc_config(reinfer: int, blend: int, inference_delay_steps: int = 4) -> None:
    if isinstance(reinfer, bool) or not isinstance(reinfer, int) or reinfer <= 0:
        raise ValueError("rtc.reinfer must be a positive integer")
    if isinstance(blend, bool) or not isinstance(blend, int) or not 0 <= blend <= reinfer:
        raise ValueError("rtc.blend must be an integer in [0, reinfer]")
    if (isinstance(inference_delay_steps, bool) or not isinstance(inference_delay_steps, int)
            or inference_delay_steps < 0):
        raise ValueError("rtc.inference_delay_steps must be a non-negative integer")


@dataclass(frozen=True)
class ChunkStepInfo:
    chunk_id: int
    chunk_index: int
    queue_remaining: int
    prefetch_started: bool
    prefetch_hit: bool
    wait_latency_s: float
    fused_overlap_steps: int
    skipped_steps: int = 0
    inference_latency_s: float = 0.0


class AsyncChunkClient:
    def __init__(self, policy: Any, *, rtc_enabled: bool = True,
                 reinfer: int = 30, blend: int = 16, inference_delay_steps: int = 4,
                 control_period_s: float = 0.1):
        validate_rtc_config(reinfer, blend, inference_delay_steps)
        if not math.isfinite(control_period_s) or control_period_s <= 0:
            raise ValueError("control_period_s must be positive and finite")
        self._policy = policy
        self._rtc_enabled = bool(rtc_enabled)
        self._reinfer = reinfer
        self._blend = blend
        self._control_period_s = control_period_s
        self._inference_delay_steps = inference_delay_steps
        self._latencies = deque(maxlen=10)
        self._pending_switch_steps = 0
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-prefetch") if rtc_enabled else None
        self._future: Future | None = None
        self._current: np.ndarray | None = None
        self._cursor = 0
        self._pending_cursor = 0
        self._pending_started_s = None
        self._generation = 0
        self._closed = False
        self._chunk_id = -1
        self._installed_overlap = 0
        self._installed_skipped = 0
        self._last_info = None
        self._prefetch_started_count = 0
        self._prefetch_hit_count = 0
        self._rtc_merge_count = 0

    @property
    def queue_remaining(self) -> int:
        with self._lock:
            return 0 if self._current is None else len(self._current) - self._cursor

    @property
    def inference_delay_steps(self) -> int:
        return self._inference_delay_steps

    def _record_latency(self, seconds: float) -> None:
        self._latencies.append(seconds)
        if len(self._latencies) == 10:
            mean = sum(self._latencies) / 10
            previous = self._inference_delay_steps
            self._inference_delay_steps = max(0, math.ceil(mean / self._control_period_s))
            logger.info("RTC delay: samples=10 mean_ms=%.1f estimated_steps=%d previous=%d",
                        mean * 1000, self._inference_delay_steps, previous)
            if self._inference_delay_steps >= self._reinfer:
                logger.warning("RTC average inference delay reaches the reinfer budget (%d >= %d)",
                               self._inference_delay_steps, self._reinfer)

    @property
    def needs_observation(self) -> bool:
        with self._lock:
            return self._current is None or (not self._rtc_enabled and self.queue_remaining == 0)

    @property
    def needs_prefetch(self) -> bool:
        with self._lock:
            return (self._rtc_enabled and self._current is not None
                    and self._future is None and self.queue_remaining <= self._reinfer)

    @property
    def last_step_info(self):
        return self._last_info

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
        chunk = np.array(self._policy.infer_chunk(observation), dtype=np.float64, copy=True)
        if chunk.ndim != 2 or chunk.shape[1:] != (10,) or not np.isfinite(chunk).all():
            raise ValueError(f"server actions must be finite (N, 10), got {chunk.shape}")
        minimum = self._reinfer + int(self._rtc_enabled)
        if len(chunk) < minimum:
            raise ValueError(f"server returned {len(chunk)} actions; reinfer={self._reinfer} "
                             f"requires at least {minimum} with RTC={self._rtc_enabled}")
        return chunk

    def _fetch_timed(self, observation: Mapping[str, Any], generation: int, started: float):
        chunk = self._fetch(observation, generation)
        # Timestamp completion here, not when the control thread consumes it.
        elapsed = time.monotonic() - started
        self._log_inference_latency(elapsed, "RTC异步")
        return chunk, elapsed

    def _log_inference_latency(self, elapsed: float, mode: str) -> None:
        steps = max(0, math.ceil(elapsed / self._control_period_s))
        logger.info(
            "推理完成[%s]: 本次延迟=%.1f ms | 本次延迟步数=%d | 当前RTC延迟估计=%d 步",
            mode, elapsed * 1000, steps, self._inference_delay_steps,
        )

    def _install(self, chunk: np.ndarray) -> None:
        self._current = chunk
        self._cursor = 0
        self._chunk_id += 1
        self._installed_overlap = 0
        self._installed_skipped = 0

    def initialize(self, observation: Mapping[str, Any]) -> np.ndarray:
        with self._lock:
            self._ensure_open()
            if self._current is not None or self._future is not None:
                raise RuntimeError("chunk client already initialized")
            started = time.monotonic()
            chunk = self._fetch(observation, self._generation)
            self._log_inference_latency(time.monotonic() - started, "首次同步")
            self._install(chunk if self._rtc_enabled else chunk[:self._reinfer])
            logger.info("scheduler initialized: rtc=%s reinfer=%d blend=%d delay=%d received=%d queued=%d",
                        self._rtc_enabled, self._reinfer, self._blend, self._inference_delay_steps,
                        len(chunk), len(self._current))
            return chunk.copy()

    def prefetch(self, observation: Mapping[str, Any]) -> bool:
        """Called after action execution, with a fresh observation of that boundary."""
        with self._lock:
            self._ensure_open()
            if not self.needs_prefetch:
                return False
            self._pending_cursor = self._cursor
            self._pending_started_s = time.monotonic()
            self._pending_switch_steps = min(self._inference_delay_steps, self.queue_remaining)
            self._future = self._executor.submit(self._fetch_timed, dict(observation),
                                                self._generation, self._pending_started_s)
            self._prefetch_started_count += 1
            return True

    def _take_ready(self) -> float:
        # Only called for a completed future. Never wait for inference in RTC.
        chunk, elapsed = self._future.result()
        self._record_latency(elapsed)
        consumed = self._cursor - self._pending_cursor
        if consumed >= len(chunk):
            raise RuntimeError("RTC response is stale: every predicted action was already consumed")
        old = self._current[self._cursor:]
        new = chunk[consumed:].copy()
        overlap = min(self._blend, len(old), len(new))
        for index in range(overlap):
            # Blend begins AFTER removing elapsed actions from the response.
            # If little old trajectory remains, finish the transition within
            # that available overlap rather than jumping halfway through it.
            weight = (index + 1) / (overlap + 1)
            new[index] = interpolate_absolute_action(old[index], new[index], weight)
        self._install(new)
        self._installed_overlap = overlap
        self._installed_skipped = consumed
        self._future = None
        self._pending_started_s = None
        self._prefetch_hit_count += 1
        self._rtc_merge_count += int(overlap > 0)
        logger.info("RTC switch: inference_ms=%.1f planned_delay=%d consumed=%d fused=%d queued=%d",
                    elapsed * 1000, self._pending_switch_steps, consumed, overlap, len(new))
        if overlap < self._blend:
            logger.warning("RTC available overlap is %d, shorter than blend=%d", overlap, self._blend)
        return elapsed

    def step(self, observation: Mapping[str, Any]) -> tuple[np.ndarray, ChunkStepInfo]:
        with self._lock:
            self._ensure_open()
            wait = 0.0
            latency = 0.0
            hit = False
            if self._current is None:
                started = time.monotonic()
                self.initialize(observation)
                wait = time.monotonic() - started
            elif self._rtc_enabled:
                if self._future is not None and self._future.done():
                    # Surface errors immediately, even before the estimated
                    # handoff point. Never wait for an unfinished future.
                    self._future.result()
                    if self._cursor - self._pending_cursor >= self._pending_switch_steps:
                        latency = self._take_ready()
                        hit = True
                elif self.queue_remaining == 0:
                    raise TimeoutError("RTC action queue exhausted before inference completed; stopping")
            elif self.queue_remaining == 0:
                started = time.monotonic()
                chunk = self._fetch(observation, self._generation)
                self._log_inference_latency(time.monotonic() - started, "同步")
                self._install(chunk[:self._reinfer])
                wait = time.monotonic() - started
            action = self._current[self._cursor].copy()
            index = self._cursor
            self._cursor += 1
            info = ChunkStepInfo(self._chunk_id, index, self.queue_remaining, False, hit, wait,
                                 self._installed_overlap, self._installed_skipped, latency)
            self._last_info = info
            return action, info

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            if self._future is not None:
                self._future.cancel()
            self._future = None
            self._pending_started_s = None
            self._current = None
            self._cursor = 0
            self._chunk_id = -1
            self._last_info = None
            reset = getattr(self._policy, "reset", None)
            if callable(reset):
                reset()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            if self._future is not None:
                self._future.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        close = getattr(self._policy, "close", None)
        if callable(close):
            close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("chunk client is closed")
