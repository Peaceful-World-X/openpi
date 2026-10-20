"""Policy adapter for stateless training-time RTC inference requests."""

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
    """Prepare an execution-space action prefix for a JAX RTC model.

    The wire protocol uses ``prev_chunk_left_over`` and ``inference_delay``.
    The model sampler receives the equivalent ``action_prefix`` and ``delay``
    tensors after the same input transforms used by training.
    """

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
        if is_pytorch:
            raise ValueError("TTRTC inference currently supports the JAX Pi0RTC model only")
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
        self._action_horizon = getattr(model, "action_horizon", None)
        self._action_dim = getattr(model, "action_dim", None)
        self._rtc_max_delay = getattr(model, "max_delay", None)
        if self._action_horizon is None or self._action_dim is None or self._rtc_max_delay is None:
            raise ValueError("RTCPolicy requires a Pi0RTC model with action_horizon, action_dim and max_delay")
        if self._action_horizon <= 0 or self._action_dim <= 0 or self._rtc_max_delay < 1:
            raise ValueError(
                f"Invalid RTC model bounds: action_horizon={self._action_horizon}, "
                f"action_dim={self._action_dim}, max_delay={self._rtc_max_delay}"
            )
        # Metadata is sent before the first request and lets clients validate
        # their deployment settings without hard-coding model internals.
        self._metadata = {
            **self._metadata,
            "rtc_enabled": True,
            "rtc_max_delay": int(self._rtc_max_delay),
            "rtc_max_delay_steps": int(self._rtc_max_delay),
            "action_horizon": int(self._action_horizon),
            "rtc_action_horizon": int(self._action_horizon),
        }

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        inference_delay: int | np.ndarray | None = None,
        prev_chunk_left_over: np.ndarray | None = None,
        # These aliases keep direct callers compatible with openpi_rtc while
        # the WebSocket wire protocol uses the more descriptive names above.
        action_prefix: np.ndarray | None = None,
        delay: int | np.ndarray | None = None,
    ) -> dict:  # type: ignore[misc]
        if inference_delay is None:
            inference_delay = delay
        if prev_chunk_left_over is None:
            prev_chunk_left_over = action_prefix

        # Direct policy callers may still put protocol fields in the
        # observation dictionary.  Remove them before applying input transforms.
        model_obs = {
            key: value
            for key, value in obs.items()
            if key not in ("inference_delay", "prev_chunk_left_over", "action_prefix", "delay")
        }
        request_sample_kwargs = self._prepare_rtc_sample_kwargs(
            model_obs,
            prev_chunk_left_over,
            inference_delay,
        )
        return self._infer(model_obs, noise=noise, request_sample_kwargs=request_sample_kwargs)

    def _prepare_rtc_sample_kwargs(
        self,
        obs: dict,
        action_prefix: np.ndarray | None,
        delay: int | np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        if action_prefix is None and delay is None:
            return {}
        if action_prefix is None or delay is None:
            raise ValueError("RTC requests must include both inference_delay and prev_chunk_left_over")

        prefix = np.array(action_prefix, dtype=np.float32, copy=True)
        if prefix.ndim != 2:
            raise ValueError(f"prev_chunk_left_over must be an unbatched 2D array, got shape {prefix.shape}")
        if prefix.shape[-1] != 10:
            raise ValueError(
                "prev_chunk_left_over must contain absolute UR7 10D actions, "
                f"got trailing dimension {prefix.shape[-1]}"
            )
        if not np.isfinite(prefix).all():
            raise ValueError("prev_chunk_left_over contains non-finite values")

        delay_array = np.asarray(delay)
        if delay_array.size != 1:
            raise ValueError(f"inference_delay must be a scalar, got shape {delay_array.shape}")
        if not np.issubdtype(delay_array.dtype, np.integer):
            raise ValueError(f"inference_delay must be an integer, got dtype {delay_array.dtype}")
        delay_int = int(delay_array.item())
        if delay_int < 0 or delay_int > self._rtc_max_delay:
            raise ValueError(
                f"inference_delay={delay_int} is outside the trained inclusive range "
                f"[0, {self._rtc_max_delay}]"
            )
        available = min(prefix.shape[0], self._action_horizon)
        if delay_int > available:
            raise ValueError(
                f"inference_delay={delay_int} exceeds available prefix length {available}"
            )

        # Replay exactly the runtime transform sequence so absolute execution
        # actions become the normalized relative actions consumed by Pi0.5.
        rtc_inputs = jax.tree.map(lambda x: x, obs)
        # Repack transforms rebuild the dictionary.  Inject after the last
        # one, so a later repack cannot silently discard the prefix.
        last_repack = max(
            (index for index, transform in enumerate(self._input_transforms)
             if isinstance(transform, _transforms.RepackTransform)),
            default=-1,
        )
        prefix_injected = False
        for index, transform in enumerate(self._input_transforms):
            if not prefix_injected and index > last_repack:
                rtc_inputs["actions"] = np.array(prefix, copy=True)
                prefix_injected = True
            rtc_inputs = transform(rtc_inputs)
        if not prefix_injected:
            rtc_inputs["actions"] = np.array(prefix, copy=True)

        transformed_prefix = np.asarray(rtc_inputs.get("actions"))
        if transformed_prefix.ndim != 2:
            raise ValueError(f"Input transforms must preserve a 2D action prefix, got {transformed_prefix.shape}")
        if not np.isfinite(transformed_prefix).all():
            raise ValueError("transformed action prefix contains non-finite values")
        transformed_prefix = transformed_prefix[..., : self._action_dim]
        if transformed_prefix.shape[-1] < self._action_dim:
            transformed_prefix = np.pad(
                transformed_prefix,
                ((0, 0), (0, self._action_dim - transformed_prefix.shape[-1])),
            )
        if transformed_prefix.shape[0] < delay_int:
            raise ValueError(
                f"Input transforms shortened action prefix to {transformed_prefix.shape[0]} steps, "
                f"below inference_delay={delay_int}"
            )
        transformed_prefix = transformed_prefix[: self._action_horizon]
        if transformed_prefix.shape[0] < self._action_horizon:
            transformed_prefix = np.concatenate(
                [
                    transformed_prefix,
                    np.zeros(
                        (self._action_horizon - transformed_prefix.shape[0], transformed_prefix.shape[-1]),
                        dtype=transformed_prefix.dtype,
                    ),
                ],
                axis=0,
            )

        return {
            "action_prefix": transformed_prefix,
            "delay": np.asarray(delay_int, dtype=np.int32),
        }
