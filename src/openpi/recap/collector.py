from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import dataclasses
from typing import Any, Protocol
import uuid

import numpy as np

from openpi.recap.episode import ReCAPFrame
from openpi.recap.episode import ReCAPOfflineEpisode
from openpi.recap.episode import save_episodes


@dataclasses.dataclass(frozen=True)
class ReCAPStep:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    success: bool | None = None
    info: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, dict):
            raise TypeError("environment step observation must be a dictionary")
        if not isinstance(self.info, dict):
            raise TypeError("environment step info must be a dictionary")
        if not np.isfinite(self.reward):
            raise ValueError("environment reward must be finite")
        for name in ("terminated", "truncated"):
            value = getattr(self, name)
            if not isinstance(value, bool | np.bool_):
                raise TypeError(f"environment step {name} must be a bool")
            object.__setattr__(self, name, bool(value))
        if self.success is not None:
            if not isinstance(self.success, bool | np.bool_):
                raise TypeError("environment step success must be a bool or None")
            object.__setattr__(self, "success", bool(self.success))


class ReCAPEnvironment(Protocol):
    def reset(self, *, seed: int | None = None) -> dict[str, Any]: ...

    def step(self, action: np.ndarray) -> ReCAPStep: ...

    def close(self) -> None: ...


@dataclasses.dataclass(frozen=True)
class InterventionDecision:
    action: np.ndarray | None
    is_intervention: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.is_intervention, bool | np.bool_):
            raise TypeError("intervention is_intervention must be a bool")
        object.__setattr__(self, "is_intervention", bool(self.is_intervention))


class ReCAPPolicy(Protocol):
    def infer(self, observation: dict[str, Any]) -> np.ndarray | Mapping[str, Any]: ...


class ReCAPRolloutCollector:
    """执行策略 rollout, 并集中处理动作安全约束和人工接管。"""

    def __init__(
        self,
        environment: ReCAPEnvironment,
        policy: ReCAPPolicy,
        *,
        task: str = "",
        intervention_callback: Callable[[dict[str, Any], np.ndarray], InterventionDecision | None] | None = None,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        smoothing: float = 0.0,
        smooth_interventions: bool = False,
        max_episode_length: int = 1000,
        seed: int | None = None,
        output_dir: str | None = None,
        advantage_condition: bool = True,
    ) -> None:
        if max_episode_length < 1:
            raise ValueError("max_episode_length must be positive")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.environment = environment
        self.policy = policy
        self.task = task
        self.intervention_callback = intervention_callback
        self.action_low = None if action_low is None else np.asarray(action_low, dtype=np.float32)
        self.action_high = None if action_high is None else np.asarray(action_high, dtype=np.float32)
        if (self.action_low is None) != (self.action_high is None):
            raise ValueError("action_low and action_high must be supplied together")
        if self.action_low is not None and (
            self.action_low.shape != self.action_high.shape
            or not np.all(np.isfinite(self.action_low))
            or not np.all(np.isfinite(self.action_high))
            or np.any(self.action_low > self.action_high)
        ):
            raise ValueError("action bounds must be finite, shape-matched, and low <= high")
        self.smoothing = smoothing
        self.smooth_interventions = bool(smooth_interventions)
        self.max_episode_length = max_episode_length
        self.seed = seed
        self.output_dir = output_dir
        self.advantage_condition = bool(advantage_condition)
        self._previous_action: np.ndarray | None = None
        self._pending_actions: list[np.ndarray] = []

    def _policy_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """为 RECAP 推理默认注入 positive 条件, 同时保留调用方显式字段。"""
        result = copy.deepcopy(observation)
        if self.advantage_condition:
            result.setdefault("advantage_indicator", np.asarray(1, dtype=np.bool_))
            result.setdefault("use_advantage", np.asarray(1, dtype=np.bool_))
            result.setdefault("is_human_intervention", np.asarray(0, dtype=np.bool_))
        return result

    @staticmethod
    def _extract_action_chunk(result: np.ndarray | Mapping[str, Any]) -> list[np.ndarray]:
        """兼容 OpenPI ``{"actions": [H, D]}`` 和旧版单步 ndarray policy。"""
        if isinstance(result, Mapping):
            if "actions" not in result:
                raise ValueError("policy inference mapping must contain an 'actions' field")
            result = result["actions"]
        action = np.asarray(result, dtype=np.float32)
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]
        if action.ndim == 1:
            return [action]
        if action.ndim != 2 or action.shape[0] < 1:
            raise ValueError(f"policy actions must have shape [D] or [H, D], got {action.shape}")
        return [action[index] for index in range(action.shape[0])]

    def _safe_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 1 or not np.all(np.isfinite(action)):
            raise ValueError("policy/intervention action must be a finite one-dimensional array")
        if self.action_low is not None:
            if action.shape != self.action_low.shape or action.shape != self.action_high.shape:
                raise ValueError("action shape does not match action bounds")
            action = np.clip(action, self.action_low, self.action_high)
        return action

    def _executed_action(self, action: np.ndarray, *, is_intervention: bool) -> np.ndarray:
        """只对最终执行动作平滑; 人工紧急纠正默认绕过平滑。"""
        if self._previous_action is not None and action.shape != self._previous_action.shape:
            # NumPy 会广播不同动作维度, 在机器人控制中必须显式拒绝而不是生成错误指令。
            raise ValueError(
                f"action shape changed within an episode: {self._previous_action.shape} -> {action.shape}"
            )
        if self._previous_action is not None and self.smoothing and (not is_intervention or self.smooth_interventions):
            action = self.smoothing * self._previous_action + (1.0 - self.smoothing) * action
        self._previous_action = action.copy()
        return action

    def collect(self, num_episodes: int) -> list[ReCAPOfflineEpisode]:
        if num_episodes < 1:
            raise ValueError("num_episodes must be positive")
        episodes = []
        try:
            for episode_index in range(num_episodes):
                observation = self.environment.reset(seed=None if self.seed is None else self.seed + episode_index)
                if not isinstance(observation, dict):
                    raise TypeError("environment.reset must return an observation dictionary")
                frames: list[ReCAPFrame] = []
                self._previous_action = None
                self._pending_actions = []
                reset_policy = getattr(self.policy, "reset", None)
                if callable(reset_policy):
                    reset_policy()
                episode_success: bool | None = None
                timeout = False
                for timestep in range(self.max_episode_length):
                    if not self._pending_actions:
                        self._pending_actions = self._extract_action_chunk(
                            self.policy.infer(self._policy_observation(observation))
                        )
                    policy_action = self._safe_action(self._pending_actions.pop(0))
                    decision = (
                        self.intervention_callback(observation, policy_action.copy())
                        if self.intervention_callback is not None
                        else None
                    )
                    if decision is None:
                        decision = InterventionDecision(action=None, is_intervention=False)
                    if not isinstance(decision, InterventionDecision):
                        raise TypeError("intervention callback must return InterventionDecision or None")
                    human_action = None if decision.action is None else self._safe_action(decision.action)
                    if decision.is_intervention and human_action is None:
                        raise ValueError("an intervention decision must provide a human action")
                    executed_action = human_action if decision.is_intervention else policy_action
                    executed_action = self._executed_action(executed_action, is_intervention=decision.is_intervention)
                    if decision.is_intervention:
                        # 接管后旧动作块对应的观测已失效, 下一帧重新向策略请求动作。
                        self._pending_actions = []
                    step = self.environment.step(executed_action)
                    if not isinstance(step, ReCAPStep):
                        raise TypeError("environment.step must return ReCAPStep")
                    frames.append(
                        ReCAPFrame(
                            t=timestep,
                            observation=copy.deepcopy(observation),
                            action=executed_action,
                            reward=float(step.reward),
                            policy_action=policy_action,
                            human_action=human_action,
                            executed_action=executed_action,
                            success=step.success,
                            # 帧级终止原因和人工标记供后续 advantage/positive 标签使用。
                            terminated=bool(step.terminated),
                            truncated=bool(step.truncated),
                            is_human_intervention=bool(decision.is_intervention),
                        )
                    )
                    observation = step.observation
                    episode_success = step.success if step.success is not None else episode_success
                    if step.terminated or step.truncated:
                        timeout = bool(step.truncated)
                        break
                else:
                    timeout = True
                    if frames:
                        # 采集器自身截断也必须写入帧级边界, 供 n-step 标注停止 bootstrap。
                        frames[-1] = dataclasses.replace(frames[-1], truncated=True)
                episodes.append(
                    ReCAPOfflineEpisode(
                        episode_id=f"episode_{episode_index:06d}_{uuid.uuid4().hex[:8]}",
                        task=self.task,
                        success=bool(episode_success) if episode_success is not None else False,
                        frames=frames,
                        timeout=timeout,
                        max_episode_length=self.max_episode_length,
                        metadata={"recap_source": "rollout"},
                    )
                )
        finally:
            try:
                self.environment.close()
            finally:
                if self.output_dir is not None and episodes:
                    save_episodes(episodes, self.output_dir)
        return episodes
