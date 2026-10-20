import numpy as np
import pytest

from openpi.serving.websocket_policy_server import _decode_request


def test_decode_request_preserves_legacy_flat_observation():
    request = {"state": np.zeros(22, dtype=np.float32), "observation": "legacy-field"}
    observation, context = _decode_request(request)
    assert observation["observation"] == "legacy-field"
    np.testing.assert_array_equal(observation["state"], request["state"])
    assert context is None


def test_decode_request_extracts_nested_ttrtc_context():
    prefix = np.zeros((4, 10), dtype=np.float32)
    observation, context = _decode_request({
        "observation": {"state": np.zeros(22, dtype=np.float32)},
        "rtc_context": {"inference_delay": 3, "prev_chunk_left_over": prefix},
    })
    assert "state" in observation
    assert context["inference_delay"] == 3
    np.testing.assert_array_equal(context["prev_chunk_left_over"], prefix)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"rtc_context": {}}, "observation"),
        ({"observation": {}, "rtc_context": {"inference_delay": 1}}, "missing required"),
        ({"observation": {}, "rtc_context": {"inference_delay": 1, "prev_chunk_left_over": [1]}}, "2D"),
    ],
)
def test_decode_request_rejects_invalid_context(payload, message):
    """非法协议字段应给出明确错误，避免与 pytest 的 request fixture 冲突。"""
    with pytest.raises(ValueError, match=message):
        _decode_request(payload)
