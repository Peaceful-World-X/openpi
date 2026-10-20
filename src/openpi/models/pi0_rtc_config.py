"""Configuration for the JAX training-time RTC Pi0 model."""

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import pi0_config
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_rtc import Pi0RTC


@dataclasses.dataclass(frozen=True)
class Pi0RTCConfig(pi0_config.Pi0Config):
    """Pi0 configuration with an inclusive training-time RTC delay bound."""

    # A delay d freezes action tokens [0, d).  The sampled range is
    # [0, max_delay], so max_delay must leave at least one postfix token.
    max_delay: int = 10

    def __post_init__(self):
        super().__post_init__()
        if (
            isinstance(self.max_delay, bool)
            or not isinstance(self.max_delay, int)
            or not 1 <= self.max_delay < self.action_horizon
        ):
            raise ValueError(
                "Training-time RTC requires 1 <= max_delay < action_horizon; "
                f"got max_delay={self.max_delay}, action_horizon={self.action_horizon}."
            )

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RTC":
        from openpi.models.pi0_rtc import Pi0RTC

        return Pi0RTC(self, rngs=nnx.Rngs(rng))
