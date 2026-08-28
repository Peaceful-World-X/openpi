from collections.abc import Sequence
from typing import Any

import jax
import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.shared import array_typing as at


class RTCPolicy(_policy.Policy):
    """为 RTC 模型准备执行空间动作前缀和控制延迟。"""

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        super().__init__(
            model,
            rng=rng,
            transforms=transforms,
            output_transforms=output_transforms,
            sample_kwargs=sample_kwargs,
            metadata=metadata,
            pytorch_device=pytorch_device,
            is_pytorch=is_pytorch,
        )
        # 两种后端暴露动作时域的位置不同, RTCPolicy 在服务边界统一读取。
        self._action_horizon = (
            model.action_horizon if hasattr(model, "action_horizon") else model.config.action_horizon
        )
        self._rtc_max_delay = getattr(model, "max_delay", None)
        if self._rtc_max_delay is None:
            raise ValueError("RTCPolicy requires a model configured with max_delay")

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
        delay: int | np.ndarray | None = None,
    ) -> dict:  # type: ignore[misc]
        # WebSocket 只传一个字典, 因此协议字段既可显式传入也可放在观测顶层。
        action_prefix = action_prefix if action_prefix is not None else obs.get("action_prefix")
        delay = delay if delay is not None else obs.get("delay")
        model_obs = {key: value for key, value in obs.items() if key not in ("action_prefix", "delay")}
        request_sample_kwargs = self._prepare_rtc_sample_kwargs(model_obs, action_prefix, delay)
        return self._infer(model_obs, noise=noise, request_sample_kwargs=request_sample_kwargs)

    # 将执行空间旧动作重放训练输入变换, 并整理为 sampler 要求的定长前缀。
    def _prepare_rtc_sample_kwargs(
        self,
        obs: dict,
        action_prefix: np.ndarray | None,
        delay: int | np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        if action_prefix is None and delay is None:
            return {}
        # 服务边界拒绝不完整协议和未训练延迟, 避免模型静默外推。
        if action_prefix is None or delay is None:
            raise ValueError("Training-Time RTC requires both action_prefix and delay")

        # MsgPack 解码可能返回只读视图; 显式复制后变换才能安全执行原地动作操作。
        prefix = np.array(action_prefix, copy=True)
        if prefix.ndim != 2:
            raise ValueError(f"Expected an unbatched action_prefix with 2 dimensions, got shape {prefix.shape}")

        delay_array = np.asarray(delay)
        if delay_array.size != 1:
            raise ValueError(f"Expected a scalar delay, got shape {delay_array.shape}")
        if not np.issubdtype(delay_array.dtype, np.integer):
            raise ValueError(f"delay must be an integer number of controller steps, got dtype {delay_array.dtype}")
        delay_value = int(delay_array.item())
        if delay_value < 0:
            raise ValueError(f"delay must be non-negative, got {delay_value}")
        if delay_value >= self._rtc_max_delay:
            raise ValueError(
                f"delay={delay_value} is outside the trained range [0, {self._rtc_max_delay}); "
                "increase model.max_delay and fine-tune again"
            )
        if delay_value > min(prefix.shape[0], self._action_horizon):
            raise ValueError(
                f"delay={delay_value} exceeds the available prefix length "
                f"{min(prefix.shape[0], self._action_horizon)}"
            )

        rtc_inputs = jax.tree.map(lambda x: x, obs)
        prefix_injected = False
        for transform in self._input_transforms:
            # 先完成 Repack 再注入 actions, 防止字典重建丢失运行时前缀。
            if not prefix_injected and not isinstance(transform, _transforms.RepackTransform):
                rtc_inputs["actions"] = np.array(prefix, copy=True)
                prefix_injected = True
            rtc_inputs = transform(rtc_inputs)
        if not prefix_injected:
            rtc_inputs["actions"] = np.array(prefix, copy=True)
        transformed_prefix = np.asarray(rtc_inputs["actions"])
        if transformed_prefix.shape[0] < delay_value:
            raise ValueError(
                f"Input transforms shortened action_prefix to {transformed_prefix.shape[0]} steps, below delay={delay_value}"
            )

        # sampler 接口固定为完整动作时域; 未提供的后缀只作占位且不会被前缀掩码读取。
        transformed_prefix = transformed_prefix[: self._action_horizon]
        if transformed_prefix.shape[0] < self._action_horizon:
            padding = np.zeros(
                (self._action_horizon - transformed_prefix.shape[0], transformed_prefix.shape[-1]),
                dtype=transformed_prefix.dtype,
            )
            transformed_prefix = np.concatenate([transformed_prefix, padding], axis=0)

        return {
            "action_prefix": transformed_prefix,
            "delay": np.asarray(delay_value, dtype=np.int32),
        }
