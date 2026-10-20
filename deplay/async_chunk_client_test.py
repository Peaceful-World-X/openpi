import time

import numpy as np

from .async_chunk_client import AsyncChunkClient


class _FakePolicy:
    def __init__(self, length=40):
        self.length = length
        self.contexts = []

    def infer_chunk(self, observation, *, rtc_context=None):
        del observation
        self.contexts.append(rtc_context)
        time.sleep(0.001)
        return np.arange(self.length * 10, dtype=np.float64).reshape(self.length, 10)


def _start_prefetch(client):
    client.initialize({"step": 0})
    for _ in range(10):
        client.step({"step": 1})
    assert client.prefetch({"step": 10})
    client._future.result()  # noqa: SLF001 - deterministic scheduler test.


def test_ttrtc_prefetch_sends_complete_remaining_chunk_and_delay():
    policy = _FakePolicy()
    client = AsyncChunkClient(
        policy,
        rtc_enabled=True,
        ttrtc_enabled=True,
        reinfer=30,
        blend=0,
        inference_delay_steps=99,
        max_delay_steps=10,
    )
    try:
        _start_prefetch(client)
        assert policy.contexts[0] is None
        context = policy.contexts[1]
        assert context["inference_delay"] == 10
        assert context["prev_chunk_left_over"].shape == (30, 10)
    finally:
        client.close()


def test_legacy_rtc_prefetch_does_not_send_ttrtc_context():
    policy = _FakePolicy()
    client = AsyncChunkClient(policy, rtc_enabled=True, ttrtc_enabled=False, blend=16)
    try:
        _start_prefetch(client)
        assert policy.contexts == [None, None]
    finally:
        client.close()
