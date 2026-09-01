from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import numpy as np

from openpi.recap.episode import ReCAPOfflineEpisode


@dataclasses.dataclass(frozen=True)
class ReCAPRewardConfig:
    """论文稀疏奖励配置; environment 模式仅用于已有规范化奖励的数据。"""

    mode: str = "paper"
    failure_penalty: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"paper", "environment"}:
            raise ValueError("reward mode must be 'paper' or 'environment'")
        if self.failure_penalty is not None and (
            not np.isfinite(self.failure_penalty) or self.failure_penalty <= 0
        ):
            raise ValueError("failure_penalty must be finite and positive")


def task_max_episode_lengths(episodes: Sequence[ReCAPOfflineEpisode]) -> dict[str, int]:
    """按任务确定归一化 horizon, 优先使用环境声明的最大 episode 长度。"""
    horizons: dict[str, int] = {}
    for episode in episodes:
        horizon = episode.max_episode_length or len(episode.frames)
        if horizon < 1 and episode.frames:
            raise ValueError(f"episode {episode.episode_id!r} has an invalid maximum length")
        if horizon >= 1:
            horizons[episode.task] = max(horizons.get(episode.task, 0), int(horizon))
    return horizons


def episode_rewards(
    episode: ReCAPOfflineEpisode,
    *,
    task_horizon: int,
    config: ReCAPRewardConfig,
) -> np.ndarray:
    """生成 value 和 advantage 共用的逐帧奖励, 避免两个阶段目标不一致。"""
    if not episode.frames:
        return np.empty(0, dtype=np.float32)
    if config.mode == "environment":
        rewards = np.asarray([frame.reward for frame in episode.frames], dtype=np.float32)
        if not np.all(np.isfinite(rewards)):
            raise ValueError(f"episode {episode.episode_id!r} contains non-finite environment rewards")
        return rewards
    if task_horizon < 1:
        raise ValueError("task_horizon must be positive")

    rewards = np.full(len(episode.frames), -1.0 / task_horizon, dtype=np.float32)
    if episode.success:
        rewards[-1] = 0.0
    else:
        # 默认令失败 terminal reward 归一化为 -1, 确保失败 return 位于 support 下界。
        failure_penalty = config.failure_penalty or float(task_horizon)
        rewards[-1] = -float(failure_penalty) / task_horizon
    return rewards


def build_episode_rewards(
    episodes: Sequence[ReCAPOfflineEpisode],
    config: ReCAPRewardConfig | None = None,
) -> list[np.ndarray]:
    """为多任务 episode 生成按任务最大长度归一化的奖励。"""
    config = config or ReCAPRewardConfig()
    horizons = task_max_episode_lengths(episodes)
    return [
        episode_rewards(episode, task_horizon=horizons.get(episode.task, 1), config=config)
        for episode in episodes
    ]


def compute_episode_returns(
    rewards: np.ndarray,
    *,
    value_min: float = -1.0,
    value_max: float = 0.0,
) -> np.ndarray:
    """计算未折扣 Monte Carlo return。

    返回保持完整累计值; ``two_hot``/``value_to_bin`` 在映射到 value support 时
    才裁剪。若在这里逐步裁剪, 会改变后续时间步看到的真实累计回报。
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    if rewards.ndim != 1 or not np.all(np.isfinite(rewards)):
        raise ValueError("rewards must be a finite one-dimensional array")
    if not np.isfinite(value_min) or not np.isfinite(value_max) or value_min >= value_max:
        raise ValueError("value_min must be finite and smaller than value_max")
    result = np.empty_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running += float(rewards[index])
        result[index] = running
    return result
