import numpy as np

from openpi.policies import ur7_policy


def _state34() -> np.ndarray:
    state = np.zeros(34, dtype=np.float32)
    state[:3] = [0.2, -0.5, 0.3]
    state[3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    state[9] = 0.8
    return state


def test_canonicalize_reconstructed_state_drops_joint_tail():
    state = _state34()
    state[22:] = np.arange(12, dtype=np.float32)
    canonical = ur7_policy.canonicalize_state(state)
    np.testing.assert_array_equal(canonical, state[:22])


def test_relative_action_round_trip():
    state = ur7_policy.canonicalize_state(_state34())
    absolute = np.array([0.25, -0.45, 0.35, 1, 0, 0, 0, 1, 0, 0.6], dtype=np.float32)
    relative = ur7_policy.absolute_to_relative(absolute, state)
    reconstructed = ur7_policy.relative_to_absolute(relative, state)
    np.testing.assert_allclose(reconstructed, absolute, atol=1e-5)


def test_inputs_produce_pi05_shapes():
    transform = ur7_policy.UR7Inputs(action_dim=32)
    image = np.zeros((3, 480, 640), dtype=np.float32)
    result = transform(
        {
            "state": _state34(),
            "actions": np.repeat(_state34()[None, :10], 4, axis=0),
            "image": image,
            "wrist_image": image,
            "task": "test task",
        }
    )
    assert result["state"].shape == (32,)
    assert result["actions"].shape == (4, 32)
    assert result["image"]["base_0_rgb"].shape == (480, 640, 3)
    assert not bool(result["image_mask"]["right_wrist_0_rgb"])
