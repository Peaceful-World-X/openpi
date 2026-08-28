import dataclasses
import logging
from typing import TYPE_CHECKING

import flax.nnx as nnx
import safetensors.torch
from typing_extensions import override

from openpi.models import pi0_config
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

if TYPE_CHECKING:
    from openpi.models.pi0_rtc import Pi0RTC


@dataclasses.dataclass(frozen=True)
class Pi0RTCConfig(pi0_config.Pi0Config):
    """Training-Time RTC 配置, 其余模型字段完全继承标准 Pi0Config。"""

    # max_delay 是随机整数 delay 的排他上界, 必须落在动作时域内。
    max_delay: int = 10

    def __post_init__(self):
        super().__post_init__()
        # 允许 max_delay 等于时域, 因为实际采样最大值仍是 action_horizon - 1。
        if not 1 <= self.max_delay <= self.action_horizon:
            raise ValueError(
                "Training-Time RTC requires 1 <= max_delay <= action_horizon; "
                f"got max_delay={self.max_delay}, action_horizon={self.action_horizon}."
            )

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RTC":
        from openpi.models.pi0_rtc import Pi0RTC

        # 子类直接继承标准层定义, 因此模型参数路径不会增加 RTC 包装前缀。
        return Pi0RTC(self, rngs=nnx.Rngs(rng))

    @override
    def load_pytorch(self, train_config, weight_path: str):
        from openpi.models_pytorch.pi0_rtc_pytorch import PI0RTCPytorch

        # 使用 RTC 子类承接相同 state_dict, 避免修改通用 BaseModelConfig 加载器。
        logger.info(f"train_config: {train_config}")
        model = PI0RTCPytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model
