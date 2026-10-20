import numpy as np
import pytest

from openpi.policies import rtc_policy
from openpi.policies import ur7_policy


def _state() -> np.ndarray:
    value = np.zeros(22, dtype=np.float32)
    value[:3] = [0.2, -0.5, 0.3]
    value[3:9] = [1, 0, 0, 0, 1, 0]
    value[9] = 0.8
    return value


def _images() -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    return image, image.copy()


def test_prepare_prefix_replays_ur7_transform_and_pads_model_action_dim():
    policy = object.__new__(rtc_policy.RTCPolicy)
    policy._action_horizon = 5
    policy._action_dim = 32
    policy._rtc_max_delay = 4
    policy._input_transforms = (ur7_policy.UR7Inputs(action_dim=32),)
    base, wrist = _images()
    prefix = np.repeat(_state()[None, :10], 2, axis=0)

    kwargs = policy._prepare_rtc_sample_kwargs(
        {"state": _state(), "image": base, "wrist_image": wrist, "task": "test"},
        prefix,
        np.asarray(2, dtype=np.int32),
    )

    assert kwargs["action_prefix"].shape == (5, 32)
    assert kwargs["action_prefix"].dtype == np.float32
    assert kwargs["delay"].item() == 2
    np.testing.assert_allclose(kwargs["action_prefix"][2:], 0.0)


def test_prepare_prefix_accepts_inclusive_max_delay_and_rejects_short_prefix():
    policy = object.__new__(rtc_policy.RTCPolicy)
    policy._action_horizon = 5
    policy._action_dim = 32
    policy._rtc_max_delay = 4
    policy._input_transforms = (ur7_policy.UR7Inputs(action_dim=32),)
    base, wrist = _images()
    obs = {"state": _state(), "image": base, "wrist_image": wrist, "task": "test"}
    prefix = np.repeat(_state()[None, :10], 4, axis=0)

    result = policy._prepare_rtc_sample_kwargs(obs, prefix, 4)
    assert result["delay"].item() == 4

    with pytest.raises(ValueError, match="available prefix length"):
        policy._prepare_rtc_sample_kwargs(obs, prefix[:3], 4)
