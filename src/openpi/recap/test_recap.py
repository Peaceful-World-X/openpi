# ruff: noqa: I001

import dataclasses
from pathlib import Path
import tempfile
from types import SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# 当前依赖组合要求先加载 Torch/LeRobot data loader, 避免随后与 JAX value 模块初始化时触发底层崩溃。
from openpi.training.data_loader import ReCAPFieldsDataset
from openpi.training.data_loader import create_torch_dataset

from openpi.models.model import Observation
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import _combine_recap_velocities
from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0_config import ReCAPConfig
from openpi.models.value_function import DistributionalValueModel
from openpi.models.value_function import bin_to_value
from openpi.models.value_function import dist_to_value
from openpi.models.value_function import distributional_cross_entropy
from openpi.models.value_function import value_to_bin
from openpi.models.value_function import two_hot
from openpi.models.value_model_config import ValueModelConfig
from openpi.policies.policy_config import create_trained_policy
from openpi.recap import InterventionDecision
from openpi.recap import OnlineReCAPRunner
from openpi.recap import ReCAPFrame
from openpi.recap import ReCAPOfflineEpisode
from openpi.recap import ReCAPPipelineHooks
from openpi.recap import ReCAPRolloutCollector
from openpi.recap import ReCAPStep
from openpi.recap import load_episodes
from openpi.recap import save_episodes
from openpi.recap.advantage import compute_n_step_advantage
from openpi.recap.advantage import label_advantages
from openpi.recap.advantage import write_labels
from openpi.recap.episode import episode_from_dict
from openpi.recap.episode import episode_to_dict
from openpi.recap.pipeline import _latest_policy_checkpoint
from openpi.recap.rewards import ReCAPRewardConfig
from openpi.recap.rewards import build_episode_rewards
from openpi.recap.rewards import compute_episode_returns
from openpi.recap.value_trainer import ValueObservationEncoder
from openpi.recap.value_trainer import ValuePredictor
from openpi.recap.value_trainer import ValueTrainer
from openpi.recap.value_trainer import save_value_checkpoint
from openpi.training import config as training_config
from openpi.transforms import ReCAPLeRobotInputs
from openpi.transforms import InjectReCAPInferenceCondition
from openpi.transforms import TokenizeReCAPAdvantage
from openpi.transforms import compose


def _frame(
    value: float,
    *,
    t: int = 0,
    reward: float = 0.0,
    intervention: bool = False,
    terminated: bool = True,
) -> ReCAPFrame:
    return ReCAPFrame(
        t,
        {"state": np.asarray([value], dtype=np.float32), "prompt": "test"},
        np.asarray([value], dtype=np.float32),
        reward=reward,
        is_human_intervention=intervention,
        terminated=terminated,
    )


def _episode(
    episode_id: str,
    task: str,
    *,
    success: bool = True,
    frames: list[ReCAPFrame] | None = None,
    source: str | None = None,
    max_episode_length: int = 10,
) -> ReCAPOfflineEpisode:
    metadata = {} if source is None else {"recap_source": source}
    return ReCAPOfflineEpisode(
        episode_id,
        task,
        success,
        frames or [_frame(0.0)],
        max_episode_length=max_episode_length,
        metadata=metadata,
    )


def _dummy_value_config() -> ValueModelConfig:
    return ValueModelConfig(
        state_dim=2,
        hidden_dim=8,
        num_cameras=1,
        max_token_len=4,
        dtype="float32",
        siglip_variant="mu/16",
        gemma_variant="dummy",
        image_resolution=(16, 16),
    )


def _value_batch(batch_size: int = 1) -> dict:
    return {
        "image": {"cam": np.zeros((batch_size, 16, 16, 3), dtype=np.float32)},
        "image_mask": {"cam": np.ones(batch_size, dtype=np.bool_)},
        "state": np.zeros((batch_size, 2), dtype=np.float32),
        "tokenized_prompt": np.ones((batch_size, 4), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((batch_size, 4), dtype=np.bool_),
    }


def test_recap_config_defaults_and_validation():
    config = Pi0Config()
    assert config.recap.enabled is False
    assert config.recap.advantage_dropout_prob == 0.3
    assert config.recap.guidance_scale == 2.0
    with pytest.raises(ValueError, match="guidance_scale"):
        ReCAPConfig(guidance_scale=-1.0)
    with pytest.raises(ValueError, match="positive_fraction"):
        ReCAPConfig(positive_fraction=0.0)
    with pytest.raises(ValueError, match="n_step_lookahead"):
        ReCAPConfig(n_step_lookahead=0)
    with pytest.raises(TypeError, match="enabled"):
        ReCAPConfig(enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dropout"):
        ReCAPConfig(advantage_dropout_prob=True)
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        ValueModelConfig(gradient_clip_norm=0.0)
    with pytest.raises(ValueError, match="freeze_mode"):
        ValueModelConfig(freeze_mode="invalid")


def test_observation_recap_roundtrip():
    raw = {
        "image": {},
        "image_mask": {},
        "state": np.zeros((1, 2), dtype=np.float32),
        "advantage_indicator": np.asarray([True]),
        "use_advantage": np.asarray([False]),
        "is_human_intervention": np.asarray([True]),
        "tokenized_advantage_positive": np.ones((1, 8), dtype=np.int32),
        "tokenized_advantage_negative": np.zeros((1, 8), dtype=np.int32),
        "tokenized_advantage_mask": np.ones((1, 8), dtype=np.bool_),
    }
    restored = Observation.from_dict(raw).to_dict()
    assert bool(restored["advantage_indicator"][0])
    assert restored["tokenized_advantage_positive"].shape == (1, 8)


def test_advantage_token_transform_shape_and_defaults():
    class FakeTokenizer:
        def __init__(self):
            self.calls = 0

        def tokenize(self, text):
            self.calls += 1
            base = 1 if text.endswith("positive") else 2
            return np.asarray([base, 3], dtype=np.int32), np.asarray([True, True])

    tokenizer = FakeTokenizer()
    transform = TokenizeReCAPAdvantage(tokenizer, max_len=4)
    data = {"advantage_indicator": np.asarray(1, dtype=np.bool_)}
    result = transform(data)
    transform(data)
    assert tokenizer.calls == 2
    assert result["tokenized_advantage_positive"].shape == (4,)
    assert result["tokenized_advantage_mask"].dtype == np.bool_
    assert bool(result["use_advantage"])

    class MismatchedTokenizer:
        def tokenize(self, text):
            length = 2 if text.endswith("positive") else 1
            return np.arange(length, dtype=np.int32), np.ones(length, dtype=np.bool_)

    with pytest.raises(ValueError, match="identical token masks"):
        TokenizeReCAPAdvantage(MismatchedTokenizer(), max_len=4)(
            {"advantage_indicator": np.asarray(1, dtype=np.bool_)}
        )


def test_pi0_recap_prefix_and_dual_loss_dropout():
    batch_size = 4
    token_length = 3

    class FakeImageEmbedder:
        def __call__(self, images, *, train):
            del train
            return jnp.ones((images.shape[0], 2, 1), dtype=jnp.float32), None

    class FakeLanguageEmbedder:
        def __call__(self, tokens, *, method):
            assert method == "embed"
            return tokens[..., None].astype(jnp.float32)

    fake_model = SimpleNamespace(
        recap_enabled=True,
        PaliGemma=SimpleNamespace(img=FakeImageEmbedder(), llm=FakeLanguageEmbedder()),
    )
    observation = Observation(
        images={"cam": jnp.zeros((batch_size, 2, 2, 3), dtype=jnp.float32)},
        image_masks={"cam": jnp.ones((batch_size,), dtype=jnp.bool_)},
        state=jnp.zeros((batch_size, 2), dtype=jnp.float32),
        advantage_indicator=jnp.asarray([True, False, True, False]),
        use_advantage=jnp.asarray([True, True, False, True]),
        tokenized_advantage_positive=jnp.full((batch_size, token_length), 7, dtype=jnp.int32),
        tokenized_advantage_negative=jnp.full((batch_size, token_length), 9, dtype=jnp.int32),
        tokenized_advantage_mask=jnp.ones((batch_size, token_length), dtype=jnp.bool_),
    )
    tokens, mask, ar_mask = Pi0.embed_prefix(fake_model, observation)
    np.testing.assert_array_equal(np.asarray(tokens[0, -token_length:, 0]), np.full(token_length, 7))
    np.testing.assert_array_equal(np.asarray(tokens[1, -token_length:, 0]), np.full(token_length, 9))
    assert not np.asarray(mask[2, -token_length:]).any()
    assert ar_mask.shape == (2 + token_length,)

    flow_calls = []

    def fake_flow(_rng, obs, _actions, *, train):
        assert train
        flow_calls.append(np.asarray(obs.use_advantage))
        return jnp.repeat(obs.use_advantage[:, None], 2, axis=1).astype(jnp.float32)

    fake_model._flow_matching_loss = fake_flow  # noqa: SLF001
    fake_model.recap_alpha = 2.0
    fake_model.recap_advantage_dropout_prob = 0.5
    rng = jax.random.key(11)
    actions = jnp.zeros((batch_size, 2, 1), dtype=jnp.float32)
    loss = Pi0.compute_recap_loss(fake_model, rng, observation, actions, train=True)
    dropout_rng = jax.random.split(rng, 3)[0]
    keep = jax.random.bernoulli(dropout_rng, p=0.5, shape=(batch_size,))
    expected_condition = np.logical_and(np.asarray(observation.use_advantage), np.asarray(keep))
    np.testing.assert_array_equal(flow_calls[0], np.zeros(batch_size, dtype=bool))
    np.testing.assert_array_equal(flow_calls[1], expected_condition)
    np.testing.assert_allclose(np.asarray(loss), 2.0 * np.repeat(expected_condition[:, None], 2, axis=1))


def test_recap_classifier_free_guidance_velocity():
    conditional = jnp.asarray([[[1.0]], [[3.0]]], dtype=jnp.float32)
    unconditional = jnp.asarray([[[0.5]], [[1.0]]], dtype=jnp.float32)
    combined = _combine_recap_velocities(
        conditional,
        unconditional,
        guidance_scale=2.0,
        use_advantage=jnp.asarray([True, False]),
    )
    np.testing.assert_allclose(np.asarray(combined), np.asarray([[[1.5]], [[1.0]]]))
    with pytest.raises(ValueError, match="one flag"):
        _combine_recap_velocities(conditional, unconditional, 2.0, jnp.asarray([True]))


def test_pi0_recap_sample_actions_uses_conditional_and_unconditional_prefixes():
    class FakeLLM:
        def __init__(self):
            self.prefix_calls = 0

        def __call__(self, parts, *, mask, positions, kv_cache=None, adarms_cond=None):
            del mask, positions, adarms_cond
            if parts[1] is None:
                self.prefix_calls += 1
                return None, parts[0][:, :1]
            suffix = jnp.broadcast_to(kv_cache, parts[1].shape)
            return (None, suffix), None

    fake_llm = FakeLLM()
    fake_model = SimpleNamespace(
        recap_enabled=True,
        recap_guidance_scale=2.0,
        action_horizon=1,
        action_dim=1,
        PaliGemma=SimpleNamespace(llm=fake_llm),
    )

    def embed_prefix(obs):
        # 条件 prefix 速度为 1, 无条件 prefix 速度为 0.5。
        tokens = 0.5 + 0.5 * obs.use_advantage[:, None, None].astype(jnp.float32)
        return tokens, jnp.ones(tokens.shape[:2], dtype=jnp.bool_), jnp.zeros((1,), dtype=jnp.bool_)

    def embed_suffix(_obs, x_t, _time):
        return x_t, jnp.ones(x_t.shape[:2], dtype=jnp.bool_), jnp.zeros((1,), dtype=jnp.bool_), None

    fake_model.embed_prefix = embed_prefix
    fake_model.embed_suffix = embed_suffix
    fake_model.action_out_proj = lambda value: value
    image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    observation = Observation(
        images={key: jnp.zeros((2, 1, 1, 3), dtype=jnp.float32) for key in image_keys},
        image_masks={key: jnp.ones((2,), dtype=jnp.bool_) for key in image_keys},
        state=jnp.zeros((2, 1), dtype=jnp.float32),
        use_advantage=jnp.asarray([True, False]),
    )
    noise = jnp.zeros((2, 1, 1), dtype=jnp.float32)
    guided = Pi0.sample_actions(fake_model, jax.random.key(0), observation, num_steps=1, noise=noise)
    np.testing.assert_allclose(np.asarray(guided[:, 0, 0]), np.asarray([-1.5, -0.5]))
    assert fake_llm.prefix_calls == 2

    fake_llm.prefix_calls = 0
    direct = Pi0.sample_actions(
        fake_model,
        jax.random.key(0),
        observation,
        num_steps=1,
        noise=noise,
        advantage_guidance_scale=1.0,
    )
    np.testing.assert_allclose(np.asarray(direct[:, 0, 0]), np.asarray([-1.0, -0.5]))
    assert fake_llm.prefix_calls == 1


def test_recap_lerobot_inputs_maps_explicit_canonical_columns():
    train_config = training_config.get_config("pi05_recap")
    assert train_config.data.base_config.action_sequence_keys == ("action",)
    raw = {
        "observation.images.base_0_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.images.left_wrist_0_rgb": np.ones((4, 4, 3), dtype=np.uint8),
        "observation.images.right_wrist_0_rgb": np.full((4, 4, 3), 2, dtype=np.uint8),
        "observation.state": np.zeros(2, dtype=np.float32),
        "action": np.zeros((3, 2), dtype=np.float32),
        "prompt": "test",
        "advantage_indicator": np.asarray(1, dtype=np.bool_),
        "use_advantage": np.asarray(1, dtype=np.bool_),
        "is_human_intervention": np.asarray(0, dtype=np.bool_),
    }
    result = ReCAPLeRobotInputs()(raw)
    assert tuple(result["image"]) == ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    assert result["actions"].shape == (3, 2)
    assert bool(result["advantage_indicator"])

    chw = dict(raw)
    chw.update(
        {
            f"observation.images.{key}": np.transpose(
                raw[f"observation.images.{key}"], (2, 0, 1)
            ).astype(np.float32)
            / 255.0
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        }
    )
    normalized = ReCAPLeRobotInputs()(chw)
    assert normalized["image"]["base_0_rgb"].shape == (4, 4, 3)
    np.testing.assert_allclose(normalized["image"]["base_0_rgb"].max(), -1.0)

    value_encoder = ValueObservationEncoder(_dummy_value_config())
    encoded = value_encoder.encode({"image": {"cam": chw["observation.images.base_0_rgb"]}, "image_mask": {"cam": True}, "state": np.zeros(2, np.float32), "prompt": "test"})
    np.testing.assert_allclose(encoded["image"]["cam"].max(), -1.0)

    with pytest.raises(ValueError, match="missing canonical camera"):
        ReCAPLeRobotInputs()({"observation.state": np.zeros(2), "action": np.zeros(2)})


def test_recap_inference_condition_defaults_and_allows_override():
    transform = InjectReCAPInferenceCondition()
    defaults = transform({"state": np.zeros(2, dtype=np.float32)})
    assert bool(defaults["advantage_indicator"])
    assert bool(defaults["use_advantage"])
    assert not bool(defaults["is_human_intervention"])

    overridden = transform(
        {
            "advantage_indicator": np.zeros((), dtype=np.bool_),
            "use_advantage": np.zeros((), dtype=np.bool_),
        }
    )
    assert not bool(overridden["advantage_indicator"])
    assert not bool(overridden["use_advantage"])


def test_recap_pytorch_policy_checkpoint_is_rejected():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        checkpoint = Path(directory)
        (checkpoint / "model.safetensors").touch()
        with pytest.raises(NotImplementedError, match="JAX checkpoints only"):
            create_trained_policy(training_config.get_config("debug_pi05_recap"), checkpoint)


def test_composite_transform_preserves_recap_sidecar_fields():
    transform = compose([lambda data: {"state": data["state"]}])
    result = transform(
        {
            "state": np.zeros(2, dtype=np.float32),
            "advantage_indicator": np.ones((), dtype=np.bool_),
            "use_advantage": np.ones((), dtype=np.bool_),
            "is_human_intervention": np.zeros((), dtype=np.bool_),
        }
    )
    assert all(field in result for field in ("advantage_indicator", "use_advantage", "is_human_intervention"))


def test_recap_fake_dataset_exercises_positive_and_negative_conditions():
    config = training_config.get_config("debug_pi05_recap")
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_torch_dataset(data_config, config.model.action_horizon, config.model)
    assert bool(dataset[0]["use_advantage"])
    assert bool(dataset[0]["advantage_indicator"])
    assert not bool(dataset[1]["advantage_indicator"])


def test_distributional_value_dummy_vlm_batch_one_and_gradient():
    model = DistributionalValueModel(_dummy_value_config(), rngs=nnx.Rngs(jax.random.key(0)))
    batch = _value_batch()
    logits = model.compute_logits(batch)
    assert logits.shape == (1, 201)
    assert model.compute_value(batch).shape == (1,)

    def loss_fn(value_model):
        return value_model.compute_loss(batch, np.asarray([-0.5], dtype=np.float32), train=True)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(grads))


def test_two_hot_and_distribution_expectation():
    target = two_hot(np.asarray([-0.5025], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(target.sum(axis=-1)), np.ones(1))
    logits = jnp.full((1, 201), -1e6).at[0, 100].set(0.0)
    np.testing.assert_allclose(np.asarray(dist_to_value(logits)), np.asarray([-0.5]), atol=1e-5)
    np.testing.assert_array_equal(np.asarray(value_to_bin([-1.0, 0.0])), [0, 200])
    np.testing.assert_allclose(np.asarray(bin_to_value([0, 200])), [-1.0, 0.0])
    with pytest.raises(ValueError, match="num_bins"):
        two_hot([0.0], num_bins=1)
    with pytest.raises(ValueError, match="identical shapes"):
        distributional_cross_entropy(jnp.zeros((1, 2)), jnp.zeros((1, 3)))


def test_paper_rewards_are_normalized_per_task():
    episodes = [
        _episode("a0", "a", success=True, frames=[_frame(0, t=0, terminated=False), _frame(0, t=1)], max_episode_length=10),
        _episode("a1", "a", success=False, frames=[_frame(0, t=0, terminated=False), _frame(0, t=1)], max_episode_length=10),
        _episode("b0", "b", success=True, frames=[_frame(0, t=0, terminated=False), _frame(0, t=1)], max_episode_length=20),
    ]
    rewards = build_episode_rewards(episodes, ReCAPRewardConfig())
    np.testing.assert_allclose(rewards[0], [-0.1, 0.0])
    np.testing.assert_allclose(rewards[1], [-0.1, -1.0])
    np.testing.assert_allclose(rewards[2], [-0.05, 0.0])
    # return 保持完整累计值, support 裁剪留给 two-hot 映射阶段。
    np.testing.assert_allclose(compute_episode_returns(np.asarray([-2.0, 2.0])), [0.0, 2.0])


def test_n_step_terminal_window_has_no_off_by_one():
    advantage = compute_n_step_advantage(
        np.asarray([-0.1, -0.1], dtype=np.float32),
        np.asarray([-0.4, -0.2], dtype=np.float32),
        n_step_lookahead=1,
        terminated=np.asarray([False, True]),
    )
    np.testing.assert_allclose(advantage, np.asarray([0.1, 0.1], dtype=np.float32))


def test_success_signal_does_not_end_n_step_window_without_termination():
    first = dataclasses.replace(_frame(0.0, t=0, terminated=False), success=True)
    second = _frame(0.0, t=1, terminated=True)
    episode = _episode("success_signal", "task", frames=[first, second])
    labeled = label_advantages(
        [episode],
        [np.zeros(2, dtype=np.float32)],
        [np.asarray([0.1, 0.2], dtype=np.float32)],
        positive_fraction=0.5,
        n_step_lookahead=2,
    )
    assert labeled[0].advantages[0] == pytest.approx(0.3)


def test_global_threshold_demo_and_intervention_rules():
    episodes = [
        _episode("demo", "a", source="demo"),
        _episode("a0", "a", source="rollout"),
        _episode("a1", "a", source="rollout", frames=[_frame(0, intervention=True)]),
        _episode("b0", "b", source="rollout"),
        _episode("b1", "b", source="rollout"),
    ]
    rewards = [
        np.asarray([-1.0], np.float32),
        np.asarray([-0.9], np.float32),
        np.asarray([-0.7], np.float32),
        np.asarray([-0.3], np.float32),
        np.asarray([-0.1], np.float32),
    ]
    values = [np.zeros(1, np.float32) for _ in episodes]
    labeled = label_advantages(episodes, values, rewards, positive_fraction=0.5, n_step_lookahead=1)
    # 论文下游 demonstration SFT 固定 I=True; rollout 仍使用全局阈值。
    assert labeled[0].advantage_indicator.tolist() == [True]
    assert labeled[2].advantage_indicator.tolist() == [True]
    assert labeled[1].threshold == pytest.approx(-0.3)
    assert labeled[3].threshold == pytest.approx(-0.3)
    assert labeled[3].advantage_indicator.tolist() == [True]


def test_demo_labels_are_preserved_positive_in_label_artifacts():
    episode = _episode(
        "demo", "task", source="demo", frames=[_frame(0, t=0, terminated=False), _frame(0, t=1)]
    )
    labeled = label_advantages(
        [episode],
        [np.zeros(2, dtype=np.float32)],
        [np.asarray([-1.0, 0.0], dtype=np.float32)],
        positive_fraction=0.4,
        n_step_lookahead=1,
    )
    np.testing.assert_array_equal(labeled[0].advantage_indicator, [True, True])
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        write_labels(labeled, directory)
        records = (Path(directory) / "recap_labels.jsonl").read_text(encoding="utf-8").splitlines()
        assert all('"label_source": "demo"' in record for record in records)


def test_label_writer_uses_standard_lerobot_identity():
    empty_episode = ReCAPOfflineEpisode(episode_id="empty", task="task", success=False, frames=[])
    episode = _episode(
        "episode_custom_id",
        "task",
        source="demo",
        frames=[_frame(0, t=0, terminated=False), _frame(0, t=1)],
    )
    labeled = label_advantages(
        [empty_episode, episode],
        [np.empty(0, dtype=np.float32), np.zeros(2, dtype=np.float32)],
        [np.empty(0, dtype=np.float32), np.asarray([-0.2, 0.0], dtype=np.float32)],
        positive_fraction=0.5,
        n_step_lookahead=1,
    )
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        write_labels(labeled, directory)
        with np.load(Path(directory) / "lerobot_fields.npz", allow_pickle=False) as fields:
            np.testing.assert_array_equal(fields["episode_index"], [0, 0])
            np.testing.assert_array_equal(fields["frame_index"], [0, 1])
            np.testing.assert_array_equal(fields["episode_id"], ["episode_custom_id"] * 2)


def test_episode_roundtrip_empty_file_and_legacy_action():
    image = np.full((2, 3, 3), 255, dtype=np.uint8)
    frame = ReCAPFrame(
        0,
        {
            "image": {"cam": image},
            "image_mask": {"cam": np.asarray(1, dtype=np.bool_)},
            "state": np.asarray([1.0], dtype=np.float32),
            "prompt": "test",
        },
        np.asarray([1.0], dtype=np.float32),
    )
    episode = _episode("e", "task", frames=[frame])
    restored = episode_from_dict(episode_to_dict(episode))
    assert isinstance(restored.frames[0].observation["image"]["cam"], np.ndarray)
    assert restored.frames[0].observation["image"]["cam"].dtype == np.uint8
    assert restored.frames[0].observation["state"].dtype == np.float32
    with pytest.raises(ValueError, match="schema_version"):
        episode_from_dict({**episode_to_dict(episode), "schema_version": 4})
    with pytest.raises(TypeError, match="success"):
        dataclasses.replace(episode, success="false")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strictly increasing"):
        dataclasses.replace(
            episode,
            frames=[_frame(0, t=1, terminated=False), _frame(0, t=1)],
        )
    with pytest.raises(ValueError, match="boundary"):
        dataclasses.replace(
            episode,
            frames=[_frame(0, t=0), _frame(0, t=1)],
        )
    with pytest.raises(ValueError, match="one shape"):
        dataclasses.replace(
            episode,
            frames=[
                _frame(0, t=0, terminated=False),
                dataclasses.replace(_frame(0, t=1), action=np.zeros(2, dtype=np.float32)),
            ],
        )
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        directory_path = Path(directory)
        save_episodes([episode], directory_path)
        loaded = load_episodes(directory_path)
        assert loaded[0].frames[0].action.shape == (1,)
        empty_path = directory_path / "empty.json"
        save_episodes([], empty_path)
        assert load_episodes(empty_path) == []
        with pytest.raises(ValueError, match="must be unique"):
            save_episodes([episode, episode], directory_path / "duplicates")
        save_episodes([episode], directory_path / "duplicate_read" / "first.json")
        save_episodes([episode], directory_path / "duplicate_read" / "second.json")
        with pytest.raises(ValueError, match="must be unique"):
            load_episodes(directory_path / "duplicate_read")


def test_fake_collector_action_chunk_intervention_and_timeout():
    class Env:
        def reset(self, *, seed=None):
            self.t = 0
            return {"state": np.zeros(1, np.float32)}

        def step(self, action):
            self.t += 1
            return ReCAPStep(
                {"state": np.zeros(1, np.float32)},
                0.0,
                terminated=self.t >= 2,
                truncated=False,
                success=self.t >= 2,
            )

        def close(self):
            pass

    class Policy:
        def infer(self, observation):
            assert bool(observation["advantage_indicator"])
            return {"actions": np.asarray([[1.0], [2.0]], dtype=np.float32)}

    calls = 0

    def intervene(_observation, _policy_action):
        nonlocal calls
        calls += 1
        return InterventionDecision(np.asarray([-1.0], dtype=np.float32), is_intervention=calls == 2)

    episode = ReCAPRolloutCollector(
        Env(), Policy(), intervention_callback=intervene, smoothing=0.9, max_episode_length=2
    ).collect(1)[0]
    assert episode.metadata["recap_source"] == "rollout"
    # 人工纠正默认绕过 smoothing, 避免紧急动作被上一帧策略动作稀释。
    assert episode.frames[1].executed_action.item() == pytest.approx(-1.0)


def test_collector_rejects_action_dimension_changes_before_environment_step():
    class Env:
        def reset(self, *, seed=None):
            self.steps = 0
            return {"state": np.zeros(1, np.float32)}

        def step(self, action):
            self.steps += 1
            return ReCAPStep(
                {"state": np.zeros(1, np.float32)},
                0.0,
                terminated=False,
                truncated=False,
            )

        def close(self):
            pass

    class Policy:
        def infer(self, observation):
            del observation
            return np.zeros(1 if env.steps == 0 else 2, dtype=np.float32)

    env = Env()
    with pytest.raises(ValueError, match="action shape changed"):
        ReCAPRolloutCollector(env, Policy(), smoothing=0.5, max_episode_length=2).collect(1)


def test_value_encoder_requires_visual_and_language_inputs():
    encoder = ValueObservationEncoder(_dummy_value_config())
    base = {"image": {"cam": np.zeros((16, 16, 3), np.uint8)}, "image_mask": {"cam": True}, "state": np.zeros(2)}
    with pytest.raises(ValueError, match="prompt"):
        encoder.encode(base)
    with pytest.raises(ValueError, match="at least one image"):
        encoder.encode({"image": {}, "image_mask": {}, "state": np.zeros(2), "prompt": "test"})
    encoded = encoder.encode({**base, "image_mask": {"cam": np.asarray([True])}, "prompt": "test"})
    assert encoded["image_mask"]["cam"].shape == ()
    with pytest.raises(ValueError, match="one flag"):
        encoder.encode({**base, "image_mask": {"cam": np.asarray([True, False])}, "prompt": "test"})
    with pytest.raises(ValueError, match="state"):
        encoder.encode({"image": base["image"], "image_mask": base["image_mask"], "prompt": "test"})


def _write_sidecar(path: Path, *, episode_order=(0, 0), use_dtype=np.bool_) -> None:
    np.savez(
        path,
        advantage_indicator=np.asarray([True, False], dtype=use_dtype),
        use_advantage=np.asarray([True, True], dtype=np.bool_),
        is_human_intervention=np.asarray([False, False], dtype=np.bool_),
        episode_index=np.asarray(episode_order, dtype=np.int64),
        frame_index=np.asarray([0, 1], dtype=np.int64),
    )


def test_recap_sidecar_dtype_length_and_identity_validation():
    class Dataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"state": np.asarray([index]), "episode_index": 0, "frame_index": index}

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        path = Path(directory) / "fields.npz"
        _write_sidecar(path)
        dataset = ReCAPFieldsDataset(Dataset(), path)
        assert dataset[0]["advantage_indicator"].dtype == np.bool_

        _write_sidecar(path, use_dtype=np.int32)
        with pytest.raises(TypeError):
            ReCAPFieldsDataset(Dataset(), path)

        _write_sidecar(path, episode_order=(1, 1))
        dataset = ReCAPFieldsDataset(Dataset(), path)
        with pytest.raises(ValueError, match="identity mismatch"):
            dataset[0]


def test_value_checkpoint_predict_and_optimizer_resume():
    config = _dummy_value_config()
    observation = {key: value[0] if not isinstance(value, dict) else {name: item[0] for name, item in value.items()} for key, value in _value_batch().items()}
    trainer = ValueTrainer(config, seed=0)
    loss = trainer.train_step([observation], np.asarray([-0.5], dtype=np.float32))
    assert np.isfinite(loss)
    assert trainer.step == 1
    assert trainer.last_grad_norm is not None
    compiled_train = trainer._train_jit  # noqa: SLF001
    assert compiled_train._cache_size() == 1  # noqa: SLF001
    assert np.isfinite(trainer.train_step([observation], np.asarray([-0.4], dtype=np.float32)))
    assert compiled_train._cache_size() == 1  # noqa: SLF001
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        checkpoint = Path(directory) / "relative_value"
        save_value_checkpoint(
            trainer.model,
            checkpoint,
            config,
            step=trainer.step,
            opt_state=trainer.opt_state,
        )
        prediction = ValuePredictor.from_checkpoint(checkpoint).predict([observation, observation], batch_size=1)
        assert prediction.shape == (2,)
        resumed = ValueTrainer.from_checkpoint(checkpoint, restore_optimizer=True)
        assert resumed.step == 2
        assert np.isfinite(resumed.train_step([observation], np.asarray([-0.5], dtype=np.float32)))
        initialized = ValueTrainer.from_checkpoint(
            checkpoint,
            restore_optimizer=False,
            learning_rate=2e-4,
            gradient_clip_norm=0.5,
            freeze_mode="backbones",
        )
        assert initialized.step == 0
        assert initialized.config.learning_rate == pytest.approx(2e-4)
        assert initialized.config.freeze_mode == "backbones"


def test_online_runner_algorithm_order_and_paths():
    calls = []
    collected = 0

    def train_value(episodes, value):
        calls.append(("value", episodes, value))

    def label(episodes, value, labels):
        calls.append(("label", episodes, value, labels))

    def policy(episodes, labels, policy_dir):
        calls.append(("policy", episodes, labels, policy_dir))
        checkpoint = policy_dir / "123"
        checkpoint.mkdir()
        return checkpoint

    def collect(policy_checkpoint, _rollout, _count):
        nonlocal collected
        assert policy_checkpoint.name == "123"
        calls.append(("collect",))
        episode = _episode(f"rollout_{collected}", "task", source="rollout")
        collected += 1
        return [episode]

    hooks = ReCAPPipelineHooks(train_value, label, policy, collect)
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        demos = root / "demos"
        save_episodes([_episode("0", "task")], demos)
        history = OnlineReCAPRunner(demos, root / "outputs", num_iterations=2, collect_episodes=1, hooks=hooks).run()
        assert [call[0] for call in calls] == ["value", "label", "policy", "collect"] * 2
        assert history[1]["episodes"] == 3
        assert history[0]["training_episodes"] == 1
        assert history[1]["training_episodes"] == 2
        assert history[1]["policy_checkpoint"].name == "123"
        saved = load_episodes(root / "outputs" / "iter_000" / "episodes")
        assert saved[0].metadata["recap_source"] == "demo"
        assert all(
            (root / "outputs" / "iter_000" / name).is_dir()
            for name in ("episodes", "value", "labels", "policy")
        )
        with pytest.raises(FileExistsError, match="already exists"):
            OnlineReCAPRunner(demos, root / "outputs", hooks=hooks).run()


def test_online_runner_validates_collection_hook_before_writing():
    hooks = ReCAPPipelineHooks(lambda *_: None, lambda *_: None, lambda *_: None)
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        demos = root / "demos"
        output = root / "outputs"
        save_episodes([_episode("0", "task")], demos)
        with pytest.raises(ValueError, match="collect_rollout"):
            OnlineReCAPRunner(demos, output, collect_episodes=1, hooks=hooks).run()
        assert not output.exists()


def test_latest_policy_checkpoint_uses_largest_numeric_step():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        run_dir = Path(directory)
        (run_dir / "2").mkdir()
        (run_dir / "10").mkdir()
        (run_dir / "metadata").mkdir()
        assert _latest_policy_checkpoint(run_dir).name == "10"

        empty = run_dir / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="no numeric checkpoints"):
            _latest_policy_checkpoint(empty)


@pytest.mark.parametrize(
    ("rollouts", "error"),
    [
        ([_episode("demo", "task", source="rollout")], "already exist"),
        ([_episode("one", "task", source="rollout")], "count mismatch"),
    ],
)
def test_online_runner_rejects_unaggregatable_rollouts(rollouts, error):
    hooks = ReCAPPipelineHooks(
        lambda *_: None,
        lambda *_: None,
        lambda *_: None,
        lambda *_: rollouts,
    )
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        demos = root / "demos"
        save_episodes([_episode("demo", "task")], demos)
        requested = 1 if "already" in error else 2
        with pytest.raises(ValueError, match=error):
            OnlineReCAPRunner(
                demos,
                root / "outputs",
                collect_episodes=requested,
                hooks=hooks,
            ).run()
