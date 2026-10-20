import dataclasses
import os
import pathlib

import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp

from openpi.models import gemma
from openpi.models import pi0_rtc_config
from openpi.models import siglip
from openpi.training import config as _config
from openpi.training import weight_loaders

from . import train


def test_real_action_metrics_ignore_padded_dimensions():
    raw_loss = jnp.zeros((2, 4, 32), dtype=jnp.float32).at[..., :10].set(2.0)
    postfix_mask = jnp.asarray([[True, True, False, True], [False, True, True, True]])
    padded_loss = raw_loss.at[..., 10:].set(1e6)

    before = train._compute_real_action_metrics(raw_loss, postfix_mask, 10)  # noqa: SLF001
    after = train._compute_real_action_metrics(padded_loss, postfix_mask, 10)  # noqa: SLF001

    np.testing.assert_allclose(before[0], after[0])
    np.testing.assert_allclose(before[1], after[1])
    np.testing.assert_allclose(before[2], after[2])


def test_ttrtc_training_logs_checkpoint_and_resume(tmp_path, monkeypatch):
    """运行真实训练入口的两步和续训，验证 TTRTC 日志及 checkpoint。"""
    original_siglip = siglip.Module

    def small_siglip(*args, **kwargs):
        """只缩小视觉网络，保留真实前向、反向和 H50/D32。"""
        kwargs["variant"] = "mu/32"
        return original_siglip(*args, **kwargs)

    monkeypatch.setattr(siglip, "Module", small_siglip)
    # FakeDataset 的随机 token 取值范围是 [0, 2048)。
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 2048)
    baseline = _config.get_config("pi05_bund_20260917_n906_v21_new_80G_8gpu_full_ttrtc")
    config = dataclasses.replace(
        baseline,
        model=pi0_rtc_config.Pi0RTCConfig(
            pi05=True,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            max_token_len=8,
        ),
        data=_config.FakeDataConfig(),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        batch_size=2,
        fsdp_devices=1,
        num_workers=0,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="ttrtc_test",
        num_train_steps=2,
        log_interval=1,
        save_interval=1,
        keep_period=1,
        wandb_enabled=False,
    )
    metrics = []
    original_init_wandb = train.init_wandb

    def record_log(data, **kwargs):
        """收集训练标量，不向外部日志服务发送数据。"""
        if "loss" in data:
            metrics.append((kwargs["step"], data))

    def init_wandb(*args, **kwargs):
        """完成 disabled 初始化后挂接记录，避免 init 替换 log 函数。"""
        original_init_wandb(*args, **kwargs)
        monkeypatch.setattr(train.wandb, "log", record_log)

    monkeypatch.setattr(train, "init_wandb", init_wandb)
    train.main(config)
    assert (config.checkpoint_dir / "1" / "params").is_dir()
    assert (config.checkpoint_dir / "1" / "train_state").is_dir()
    train.main(dataclasses.replace(config, resume=True, num_train_steps=4))
    assert (config.checkpoint_dir / "3" / "params").is_dir()
    assert [step for step, _ in metrics] == [0, 1, 2, 3]
    expected = {"loss", "grad_norm", "loss_real_action", "ttrtc_delay_mean", "ttrtc_postfix_fraction"}
    expected |= {f"loss_real_action_dim_{index:02d}" for index in range(10)}
    for _, info in metrics:
        assert expected <= info.keys()
        assert all(np.shape(value) == () and np.isfinite(value) for value in info.values())
        assert info["grad_norm"] > 0
        np.testing.assert_allclose(info["ttrtc_postfix_fraction"], 1 - info["ttrtc_delay_mean"] / 50)


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
