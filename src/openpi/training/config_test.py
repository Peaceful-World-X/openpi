import dataclasses

import pytest

from openpi.models import pi0_config
from openpi.models import pi0_rtc_config
from openpi.training import config as _config


def test_pi0_rtc_config_samples_inclusive_delay_bound():
    model = pi0_rtc_config.Pi0RTCConfig(pi05=True, action_dim=32, action_horizon=50, max_delay=10)
    assert model.max_delay == 10
    with pytest.raises(ValueError, match="max_delay"):
        pi0_rtc_config.Pi0RTCConfig(pi05=True, action_dim=32, action_horizon=10, max_delay=10)


def test_train_config_validates_real_action_dim():
    model = pi0_config.Pi0Config(action_dim=32, action_horizon=50)
    with pytest.raises(ValueError, match="real_action_dim"):
        _config.TrainConfig(name="invalid_real_dim", model=model, real_action_dim=33)


@pytest.mark.parametrize("gpu_count", [4, 8])
def test_ttrtc_config_preserves_original_hardware_and_training_budget(gpu_count):
    """切换 TTRTC 不改变同卡数普通配置的模型规模、数据或训练预算。"""
    base_name = f"pi05_bund_20260917_n906_v21_new_80G_{gpu_count}gpu_full"
    baseline = _config.get_config(base_name)
    config = _config.get_config(f"{base_name}_ttrtc")
    assert isinstance(config.model, pi0_rtc_config.Pi0RTCConfig)
    assert config.model.max_delay == 10
    assert config.real_action_dim == 10
    assert config.fsdp_devices == gpu_count
    for field in dataclasses.fields(baseline):
        if field.name not in {"name", "model", "real_action_dim"}:
            assert getattr(config, field.name) == getattr(baseline, field.name), field.name
    for field in dataclasses.fields(baseline.model):
        assert getattr(config.model, field.name) == getattr(baseline.model, field.name), field.name
