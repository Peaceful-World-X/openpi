from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path

import numpy as np

from openpi.recap.episode import ReCAPOfflineEpisode
from openpi.recap.rewards import build_episode_rewards


def compute_n_step_reward_sums(
    rewards: np.ndarray,
    *,
    n_step_lookahead: int,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
) -> np.ndarray:
    """计算最多 N 步 reward; 遇到 terminal 后停止, 避免跨 episode bootstrap。"""
    if (
        isinstance(n_step_lookahead, bool | np.bool_)
        or not isinstance(n_step_lookahead, int | np.integer)
        or n_step_lookahead < 1
    ):
        raise ValueError("n_step_lookahead must be >= 1")
    rewards = np.asarray(rewards, dtype=np.float32)
    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("rewards must be finite")
    done = np.zeros(len(rewards), dtype=bool) if terminated is None else np.asarray(terminated, dtype=bool)
    timeout = np.zeros(len(rewards), dtype=bool) if truncated is None else np.asarray(truncated, dtype=bool)
    if done.shape != rewards.shape or timeout.shape != rewards.shape:
        raise ValueError("terminated and truncated must have the same shape as rewards")
    boundary = np.logical_or(done, timeout)
    output = np.zeros_like(rewards)
    for index in range(len(rewards)):
        end = min(index + n_step_lookahead, len(rewards))
        output[index] = np.sum(rewards[index:end], dtype=np.float32)
        terminal_indices = np.flatnonzero(boundary[index:end])
        if len(terminal_indices):
            output[index] = np.sum(rewards[index : index + int(terminal_indices[0]) + 1], dtype=np.float32)
    return output


def compute_n_step_advantage(
    rewards: np.ndarray,
    values: np.ndarray,
    *,
    n_step_lookahead: int = 50,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
) -> np.ndarray:
    """计算 A_t = sum(r) + V(s_{t+N}) - V(s_t), terminal 状态不 bootstrap。"""
    if (
        isinstance(n_step_lookahead, bool | np.bool_)
        or not isinstance(n_step_lookahead, int | np.integer)
        or n_step_lookahead < 1
    ):
        raise ValueError("n_step_lookahead must be a positive integer")
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if rewards.ndim != 1 or values.ndim != 1:
        raise ValueError("rewards and values must be one-dimensional")
    if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(values)):
        raise ValueError("rewards and values must be finite")
    if values.shape not in (rewards.shape, (len(rewards) + 1,)):
        raise ValueError(f"values must have shape {rewards.shape} or {(len(rewards) + 1,)}, got {values.shape}")
    current_values = values[: len(rewards)]
    done = np.zeros(len(rewards), dtype=bool) if terminated is None else np.asarray(terminated, dtype=bool)
    if done.shape != rewards.shape:
        raise ValueError("terminated must have the same shape as rewards")
    timeout = np.zeros(len(rewards), dtype=bool) if truncated is None else np.asarray(truncated, dtype=bool)
    if timeout.shape != rewards.shape:
        raise ValueError("truncated must have the same shape as rewards")
    boundary = np.logical_or(done, timeout)
    sums = compute_n_step_reward_sums(
        rewards, n_step_lookahead=n_step_lookahead, terminated=done, truncated=timeout
    )
    output = np.empty_like(rewards)
    for index in range(len(rewards)):
        target = min(index + n_step_lookahead, len(values) - 1)
        # 只检查实际计入 reward sum 的 N 个 transition; t+N 的 terminal 属于下一次动作。
        reward_end = min(index + n_step_lookahead, len(rewards))
        boundary_indices = np.flatnonzero(boundary[index:reward_end])
        if boundary_indices.size:
            boundary_index = index + int(boundary_indices[0])
            target = min(boundary_index + 1, len(values) - 1)
            # terminal 没有后继状态; timeout 只有显式提供 len(rewards)+1 value 时才 bootstrap。
            can_bootstrap_timeout = bool(
                timeout[boundary_index]
                and not done[boundary_index]
                and len(values) == len(rewards) + 1
            )
            bootstrap = float(values[target]) if can_bootstrap_timeout else 0.0
        elif index + n_step_lookahead < len(values):
            bootstrap = float(values[index + n_step_lookahead])
        else:
            bootstrap = 0.0
        output[index] = sums[index] + bootstrap - current_values[index]
    return output


def progress_value_proxy(episode: ReCAPOfflineEpisode) -> np.ndarray:
    """无 value checkpoint 时的显式调试 proxy, 正式流程应传入模型 value。"""
    length = len(episode.frames)
    if length == 0:
        return np.empty(0, dtype=np.float32)
    horizon = float(episode.max_episode_length or length)
    if not episode.success:
        return np.full(length, -1.0, dtype=np.float32)
    return -np.maximum(length - 1 - np.arange(length), 0).astype(np.float32) / horizon


@dataclasses.dataclass(frozen=True)
class LabeledEpisode:
    episode: ReCAPOfflineEpisode
    rewards: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    advantage_indicator: np.ndarray
    threshold: float


def label_advantages(
    episodes: Sequence[ReCAPOfflineEpisode],
    values: Sequence[np.ndarray] | None = None,
    rewards: Sequence[np.ndarray] | None = None,
    *,
    positive_fraction: float = 0.4,
    n_step_lookahead: int = 50,
    use_proxy: bool = False,
) -> list[LabeledEpisode]:
    """按全局 rollout quantile 生成标签, demo/人工纠正动作固定为 positive。"""
    if not 0.0 < positive_fraction <= 1.0:
        raise ValueError("positive_fraction must be in (0, 1]")
    if values is not None and len(values) != len(episodes):
        raise ValueError("values must contain one array per episode")
    if rewards is not None and len(rewards) != len(episodes):
        raise ValueError("rewards must contain one array per episode")
    if rewards is None:
        rewards = build_episode_rewards(episodes)
    advantage_arrays: list[np.ndarray] = []
    value_arrays: list[np.ndarray] = []
    reward_arrays: list[np.ndarray] = []
    for index, episode in enumerate(episodes):
        episode_reward = np.asarray(rewards[index], dtype=np.float32)
        if values is None:
            if not use_proxy:
                raise ValueError("value predictions are required unless use_proxy=True")
            predicted = progress_value_proxy(episode)
        else:
            predicted = np.asarray(values[index], dtype=np.float32)
        if predicted.shape != episode_reward.shape:
            raise ValueError(
                f"value shape {predicted.shape} does not match episode frame count {episode_reward.shape}"
            )
        # success 是任务结果信号, 只有 terminated/truncated 才定义 transition 边界。
        done = np.asarray([frame.terminated for frame in episode.frames], dtype=bool)
        timeout = np.asarray([frame.truncated for frame in episode.frames], dtype=bool)
        if len(episode.frames) and not np.any(np.logical_or(done, timeout)):
            # 旧 JSON 只在 episode 顶层保存 success/timeout, 需要把边界回填到最后一帧。
            if episode.timeout:
                timeout[-1] = True
            else:
                done[-1] = True
        advantages = compute_n_step_advantage(
            episode_reward, predicted, n_step_lookahead=n_step_lookahead, terminated=done, truncated=timeout
        )
        reward_arrays.append(episode_reward)
        value_arrays.append(predicted)
        advantage_arrays.append(advantages)
    rollout_candidates: list[np.ndarray] = []
    all_candidates: list[np.ndarray] = []
    for episode, advantages in zip(episodes, advantage_arrays, strict=True):
        intervention = np.asarray([frame.is_human_intervention for frame in episode.frames], dtype=bool)
        candidates = advantages[~intervention]
        if candidates.size:
            all_candidates.append(candidates)
            if episode.metadata.get("recap_source") == "rollout":
                rollout_candidates.append(candidates)
    # 参考实现对当前批次所有 autonomous rollout frame 使用一个全局 quantile。
    # 初始 demonstration-only 阶段没有 rollout, 才回退到全部非干预帧。
    threshold_candidates = rollout_candidates or all_candidates
    flat = np.concatenate(threshold_candidates) if threshold_candidates else np.empty(0, dtype=np.float32)
    threshold = float(np.quantile(flat, 1.0 - positive_fraction)) if flat.size else 0.0
    labeled = []
    for episode, rewards, predicted, advantages in zip(episodes, reward_arrays, value_arrays, advantage_arrays, strict=True):
        intervention = np.asarray([frame.is_human_intervention for frame in episode.frames], dtype=bool)
        if episode.metadata.get("recap_source") == "demo":
            # 论文下游 SFT 阶段固定 I=True; 参考数据转换同样保留全部 demonstration 为 positive。
            indicator = np.ones(len(episode.frames), dtype=np.bool_)
        else:
            # 参考脚本使用 >= percentile; 在线人工纠正动作无条件视为 positive。
            indicator = np.logical_or(advantages >= threshold, intervention)
        labeled.append(LabeledEpisode(episode, rewards, predicted, advantages, indicator, threshold))
    return labeled


def write_labels(labeled: Sequence[LabeledEpisode], output: str | Path) -> dict:
    """写入可审计 JSONL 和与标准 LeRobot 帧身份对齐的 sidecar。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    indicators = []
    use_advantage = []
    interventions = []
    episode_indices = []
    frame_indices = []
    episode_ids = []
    timesteps = []
    thresholds = {}
    lerobot_episode_index = 0
    for item in labeled:
        episode = item.episode
        # 阈值是本批次全局值; JSON metadata 仍保留字段, 便于审计。
        thresholds["global"] = float(item.threshold)
        if not episode.frames:
            # LeRobot 不存在零帧 episode; 空 JSON 轨迹不能占用 flattened episode_index。
            continue
        for frame_index, frame in enumerate(episode.frames):
            records.append(
                {
                    "episode_id": episode.episode_id,
                    "task": episode.task,
                    "success": bool(episode.success),
                    "t": int(frame.t),
                    "reward": float(item.rewards[frame_index]),
                    "value": float(item.values[frame_index]),
                    "advantage": float(item.advantages[frame_index]),
                    "advantage_indicator": bool(item.advantage_indicator[frame_index]),
                    "use_advantage": True,
                    "is_human_intervention": bool(frame.is_human_intervention),
                    "label_source": (
                        "demo"
                        if episode.metadata.get("recap_source") == "demo"
                        else "human"
                        if frame.is_human_intervention
                        else "value"
                    ),
                }
            )
            indicators.append(bool(item.advantage_indicator[frame_index]))
            use_advantage.append(True)
            interventions.append(bool(frame.is_human_intervention))
            episode_indices.append(lerobot_episode_index)
            frame_indices.append(frame_index)
            episode_ids.append(episode.episode_id)
            timesteps.append(int(frame.t))
        lerobot_episode_index += 1
    (output / "recap_labels.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    np.savez(
        output / "lerobot_fields.npz",
        advantage_indicator=np.asarray(indicators, dtype=np.bool_),
        use_advantage=np.asarray(use_advantage, dtype=np.bool_),
        is_human_intervention=np.asarray(interventions, dtype=np.bool_),
        # 使用 LeRobot 标准身份字段逐帧核对; 原始 JSON 身份同时保留用于审计。
        episode_index=np.asarray(episode_indices, dtype=np.int64),
        frame_index=np.asarray(frame_indices, dtype=np.int64),
        episode_id=np.asarray(episode_ids, dtype=np.str_),
        t=np.asarray(timesteps, dtype=np.int64),
    )
    metadata = {"thresholds": thresholds, "frames": len(records)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = {"frames": len(records), "thresholds": thresholds}
    if len(thresholds) == 1:
        result["threshold"] = next(iter(thresholds.values()))
    return result
