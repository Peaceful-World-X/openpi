import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_delta_dual_cartesian_pose_wrap_and_gripper_passthrough():
    # Mask: per wrist 7 dims -> first 6 delta, last (gripper) keep absolute.
    mask = _transforms.make_bool_mask(6, -1, 6, -1)
    transform = _transforms.DeltaCartesianPose(mask=mask)

    state = np.array(
        [
            1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 9.0,      # left
            10.0, 20.0, 30.0, 0.4, 0.5, 0.6, 8.0,   # right
        ],
        dtype=np.float32,
    )

    # Absolute actions: positions equal to state, angles equal to state + 2pi (should wrap to 0 delta),
    # grippers are arbitrary and should remain unchanged since mask[-1] is False for each wrist.
    actions = np.array(
        [
            [
                1.0, 2.0, 3.0, 0.1 + 2 * np.pi, 0.2 + 2 * np.pi, 0.3 + 2 * np.pi, 5.0,
                10.0, 20.0, 30.0, 0.4 + 2 * np.pi, 0.5 + 2 * np.pi, 0.6 + 2 * np.pi, 6.0,
            ],
            [
                2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 7.0,
                11.0, 21.0, 31.0, 0.4, 0.5, 0.6, 8.0,
            ],
        ],
        dtype=np.float32,
    )

    out = transform({"state": state, "actions": actions})
    delta = out["actions"]

    # Step 0: masked dims become 0, angles wrap to 0, grippers unchanged.
    assert np.allclose(delta[0, 0:6], 0.0, atol=1e-6)
    assert np.allclose(delta[0, 7:13], 0.0, atol=1e-6)
    assert np.allclose(delta[0, 6], 5.0)
    assert np.allclose(delta[0, 13], 6.0)

    # Step 1: delta is (actions - state) for masked dims.
    assert np.allclose(delta[1, 0:3], np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert np.allclose(delta[1, 7:10], np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert np.allclose(delta[1, 3:6], 0.0, atol=1e-6)
    assert np.allclose(delta[1, 10:13], 0.0, atol=1e-6)


def test_absolute_dual_cartesian_pose_inverse_of_delta_for_masked_dims():
    mask = _transforms.make_bool_mask(6, -1, 6, -1)
    to_abs = _transforms.AbsoluteCartesianPose(mask=mask)

    state = np.array(
        [
            1.0, 2.0, 3.0, -3.0, 3.0, 0.0, 9.0,     # left
            10.0, 20.0, 30.0, -3.0, 3.0, 0.0, 8.0,  # right
        ],
        dtype=np.float32,
    )

    # Delta actions: add state back for masked dims; angles should wrap into [-pi, pi].
    delta_actions = np.zeros((1, 14), dtype=np.float32)
    delta_actions[0, 6] = 5.0   # gripper passthrough
    delta_actions[0, 13] = 6.0  # gripper passthrough

    out = to_abs({"state": state, "actions": delta_actions.copy()})
    abs_actions = out["actions"]

    assert np.allclose(abs_actions[0, 0:3], state[0:3])
    assert np.allclose(abs_actions[0, 7:10], state[7:10])
    assert np.allclose(abs_actions[0, 3:6], ((state[3:6] + np.pi) % (2 * np.pi)) - np.pi)
    assert np.allclose(abs_actions[0, 10:13], ((state[10:13] + np.pi) % (2 * np.pi)) - np.pi)
    assert np.allclose(abs_actions[0, 6], 5.0)
    assert np.allclose(abs_actions[0, 13], 6.0)

def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})
