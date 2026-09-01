"""使用 value checkpoint 或显式 progress proxy 生成 RECAP advantage 标签。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openpi.recap.advantage import label_advantages
from openpi.recap.advantage import write_labels
from openpi.recap.episode import load_episodes
from openpi.recap.rewards import ReCAPRewardConfig
from openpi.recap.rewards import build_episode_rewards
from openpi.recap.value_trainer import ValuePredictor


def label_recap(
    episodes_path: str | Path,
    output: str | Path,
    *,
    value_checkpoint: str | Path | None = None,
    use_proxy: bool = False,
    positive_fraction: float = 0.4,
    n_step_lookahead: int = 50,
    inference_batch_size: int = 32,
    reward_mode: str = "paper",
    failure_penalty: float | None = None,
) -> dict:
    episodes = load_episodes(episodes_path)
    if not episodes:
        raise ValueError("no episodes found")
    if value_checkpoint is None and not use_proxy:
        raise ValueError("value_checkpoint is required unless --use-proxy is explicitly set")
    values = None
    if value_checkpoint is not None:
        predictor = ValuePredictor.from_checkpoint(value_checkpoint)
        episode_lengths = [len(episode.frames) for episode in episodes]
        flat_observations = [frame.observation for episode in episodes for frame in episode.frames]
        flat_values = predictor.predict(flat_observations, batch_size=inference_batch_size)
        # 全局分批后再按 episode 切分, 避免每种 episode 尾 batch shape 都触发一次 JIT 编译。
        offsets = np.cumsum(episode_lengths[:-1], dtype=np.int64)
        values = list(np.split(flat_values, offsets))
    rewards = build_episode_rewards(
        episodes,
        ReCAPRewardConfig(mode=reward_mode, failure_penalty=failure_penalty),
    )
    labeled = label_advantages(
        episodes,
        values,
        rewards,
        positive_fraction=positive_fraction,
        n_step_lookahead=n_step_lookahead,
        use_proxy=use_proxy,
    )
    return write_labels(labeled, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label RECAP advantages and write JSONL plus LeRobot sidecar")
    parser.add_argument("episodes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--value-checkpoint")
    parser.add_argument("--use-proxy", action="store_true", help="explicit debug fallback without value checkpoint")
    parser.add_argument("--positive-fraction", type=float, default=0.4)
    parser.add_argument("--n-step-lookahead", type=int, default=50)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--reward-mode", choices=("paper", "environment"), default="paper")
    parser.add_argument("--failure-penalty", type=float)
    args = parser.parse_args()
    print(
        label_recap(
            args.episodes,
            args.output,
            value_checkpoint=args.value_checkpoint,
            use_proxy=args.use_proxy,
            positive_fraction=args.positive_fraction,
            n_step_lookahead=args.n_step_lookahead,
            inference_batch_size=args.inference_batch_size,
            reward_mode=args.reward_mode,
            failure_penalty=args.failure_penalty,
        )
    )


if __name__ == "__main__":
    main()
