"""从 RECAP episode JSON 训练 distributional value function。

示例: ``uv run scripts/train_value.py episodes --output value --steps 100``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.models.value_model_config import ValueModelConfig
from openpi.recap.episode import load_episodes
from openpi.recap.rewards import ReCAPRewardConfig
from openpi.recap.rewards import build_episode_rewards
from openpi.recap.rewards import compute_episode_returns
from openpi.recap.value_trainer import ValueTrainer
from openpi.recap.value_trainer import save_value_checkpoint


def _returns(rewards: np.ndarray, *, value_min: float = -1.0, value_max: float = 0.0) -> np.ndarray:
    # 保留脚本内旧名称, 实际公式集中在 rewards 模块供训练与测试共用。
    return compute_episode_returns(rewards, value_min=value_min, value_max=value_max)


def train_value(
    episodes_path: str | Path,
    output: str | Path,
    *,
    steps: int = 1000,
    batch_size: int = 32,
    state_dim: int = 32,
    hidden_dim: int = 512,
    value_bins: int = 201,
    value_min: float = -1.0,
    value_max: float = 0.0,
    learning_rate: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    freeze_mode: str = "none",
    num_cameras: int = 3,
    max_token_len: int = 48,
    dtype: str = "bfloat16",
    siglip_variant: str = "So400m/14",
    gemma_variant: str = "gemma_300m",
    eval_fraction: float = 0.2,
    eval_batch_size: int = 32,
    seed: int = 0,
    reward_mode: str = "paper",
    failure_penalty: float | None = None,
    init_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, float | int]:
    episodes = load_episodes(episodes_path)
    if not episodes:
        raise ValueError("no episodes found")
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    order = np.arange(len(episodes))
    rng.shuffle(order)
    eval_count = 0 if eval_fraction == 0.0 else min(max(1, round(len(order) * eval_fraction)), max(0, len(order) - 1))
    eval_ids = set(order[:eval_count].tolist())
    train_ids = [index for index in range(len(episodes)) if index not in eval_ids]
    eval_ids_ordered = [index for index in range(len(episodes)) if index in eval_ids]
    # 空 episode 可以进入 eval, 但至少保留一个有帧 episode 给训练, 避免随机划分后 fit 失败。
    nonempty_ids = [index for index, episode in enumerate(episodes) if episode.frames]
    if nonempty_ids and not any(episodes[index].frames for index in train_ids):
        moved = next(index for index in eval_ids_ordered if episodes[index].frames)
        eval_ids.remove(moved)
        eval_ids_ordered.remove(moved)
        train_ids.append(moved)
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    if not any(episodes[index].frames for index in train_ids):
        raise ValueError("training episodes contain no frames")
    reward_arrays = build_episode_rewards(
        episodes,
        ReCAPRewardConfig(mode=reward_mode, failure_penalty=failure_penalty),
    )

    config = ValueModelConfig(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        value_bins=value_bins,
        value_min=value_min,
        value_max=value_max,
        learning_rate=learning_rate,
        gradient_clip_norm=gradient_clip_norm,
        freeze_mode=freeze_mode,
        num_cameras=num_cameras,
        max_token_len=max_token_len,
        dtype=dtype,
        siglip_variant=siglip_variant,
        gemma_variant=gemma_variant,
    )
    if init_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError("init_checkpoint and resume_checkpoint are mutually exclusive")
    if resume_checkpoint is not None:
        trainer = ValueTrainer.from_checkpoint(resume_checkpoint, seed=seed, restore_optimizer=True)
        config = trainer.config
    elif init_checkpoint is not None:
        # 论文每轮从 V_pre 权重重新 fine-tune, 而不是延续上一轮 optimizer。
        trainer = ValueTrainer.from_checkpoint(
            init_checkpoint,
            seed=seed,
            restore_optimizer=False,
            learning_rate=learning_rate,
            gradient_clip_norm=gradient_clip_norm,
            freeze_mode=freeze_mode,
        )
        config = trainer.config
    else:
        trainer = ValueTrainer(config, seed=seed)

    def flatten(indices):
        observations = [frame.observation for index in indices for frame in episodes[index].frames]
        targets = (
            np.concatenate(
                [
                    _returns(
                        reward_arrays[index],
                        value_min=config.value_min,
                        value_max=config.value_max,
                    )
                    for index in indices
                    if episodes[index].frames
                ]
            )
            if observations
            else np.empty(0, dtype=np.float32)
        )
        return observations, targets

    observations, targets = flatten(train_ids)
    losses = trainer.fit(observations, targets, steps=steps, batch_size=batch_size)
    eval_observations, eval_targets = flatten(eval_ids_ordered)
    eval_loss = (
        trainer.evaluate(eval_observations, eval_targets, batch_size=eval_batch_size)
        if eval_observations
        else None
    )
    output = Path(output)
    save_value_checkpoint(trainer.model, output, config, step=trainer.step, opt_state=trainer.opt_state)
    (output / "split_manifest.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "eval_episode_ids": [episodes[index].episode_id for index in sorted(eval_ids)],
                "train_episode_ids": [episodes[index].episode_id for index in train_ids],
                "reward_mode": reward_mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result: dict[str, float | int] = {
        "train_episodes": len(train_ids),
        "eval_episodes": len(eval_ids),
        "steps": trainer.step,
        "loss": losses[-1],
    }
    if eval_loss is not None:
        result["eval_loss"] = eval_loss
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a RECAP distributional value function from episode JSON")
    parser.add_argument("episodes", help="episode JSON file or directory")
    parser.add_argument("--output", required=True, help="Orbax value checkpoint directory")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--value-bins", type=int, default=201)
    parser.add_argument("--value-min", type=float, default=-1.0)
    parser.add_argument("--value-max", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--freeze-mode", choices=("none", "backbones"), default="none")
    parser.add_argument("--num-cameras", type=int, default=3)
    parser.add_argument("--max-token-len", type=int, default=48)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--siglip-variant", default="So400m/14")
    parser.add_argument("--gemma-variant", default="gemma_300m")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-mode", choices=("paper", "environment"), default="paper")
    parser.add_argument("--failure-penalty", type=float)
    parser.add_argument("--init-checkpoint", help="每轮重新 fine-tune 使用的 V_pre checkpoint")
    parser.add_argument("--resume-checkpoint", help="恢复同一次 value 训练及 optimizer")
    args = parser.parse_args()
    print(
        train_value(
            args.episodes,
            args.output,
            steps=args.steps,
            batch_size=args.batch_size,
            state_dim=args.state_dim,
            hidden_dim=args.hidden_dim,
            value_bins=args.value_bins,
            value_min=args.value_min,
            value_max=args.value_max,
            learning_rate=args.learning_rate,
            gradient_clip_norm=args.gradient_clip_norm,
            freeze_mode=args.freeze_mode,
            num_cameras=args.num_cameras,
            max_token_len=args.max_token_len,
            dtype=args.dtype,
            siglip_variant=args.siglip_variant,
            gemma_variant=args.gemma_variant,
            eval_fraction=args.eval_fraction,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
            reward_mode=args.reward_mode,
            failure_penalty=args.failure_penalty,
            init_checkpoint=args.init_checkpoint,
            resume_checkpoint=args.resume_checkpoint,
        )
    )


if __name__ == "__main__":
    main()
