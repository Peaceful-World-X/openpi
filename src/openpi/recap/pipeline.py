from __future__ import annotations

from collections.abc import Callable
import dataclasses
from pathlib import Path
from typing import Any

from openpi.recap.episode import ReCAPOfflineEpisode
from openpi.recap.episode import load_episodes
from openpi.recap.episode import save_episodes


def _latest_policy_checkpoint(run_dir: Path) -> Path:
    """解析 OpenPI CheckpointManager 生成的最后一个数值 step 目录。"""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"policy checkpoint run directory was not created: {run_dir}")
    checkpoints = [path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    if not checkpoints:
        raise FileNotFoundError(f"policy training produced no numeric checkpoints under: {run_dir}")
    return max(checkpoints, key=lambda path: int(path.name)).resolve()


@dataclasses.dataclass(frozen=True)
class ReCAPPipelineHooks:
    """可注入的阶段函数; 测试可使用 fake hook, 不启动真实训练进程。"""

    train_value: Callable[[Path, Path], Any]
    label_advantage: Callable[[Path, Path, Path], Any]
    train_policy: Callable[[Path, Path, Path], Any]
    collect_rollout: Callable[[Path, Path, int], Any] | None = None


class OnlineReCAPRunner:
    """按论文 Algorithm 1 编排 value、label、policy 和 online rollout 阶段。"""

    def __init__(
        self,
        demo_episodes: str | Path,
        output_dir: str | Path,
        *,
        num_iterations: int = 1,
        collect_episodes: int = 0,
        hooks: ReCAPPipelineHooks | None = None,
    ) -> None:
        if num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if collect_episodes < 0:
            raise ValueError("collect_episodes must be non-negative")
        self.demo_episodes = Path(demo_episodes)
        self.output_dir = Path(output_dir)
        self.num_iterations = num_iterations
        self.collect_episodes = collect_episodes
        self.hooks = hooks

    def _write_episodes(self, directory: Path, episodes: list[ReCAPOfflineEpisode]) -> None:
        save_episodes(episodes, directory)

    def run(self) -> list[dict[str, Any]]:
        if self.hooks is None:
            raise ValueError("OnlineReCAPRunner requires hooks; scripts/recap_train supplies production hooks")
        if self.collect_episodes and self.hooks.collect_rollout is None:
            # 在创建目录或启动训练前失败, 避免昂贵阶段结束后才发现在线闭环不完整。
            raise ValueError("collect_episodes > 0 requires a collect_rollout hook")
        episodes = load_episodes(self.demo_episodes)
        if not episodes:
            raise ValueError("demo dataset contains no episodes")
        history: list[dict[str, Any]] = []
        # 标记数据来源, 让阈值阶段区分初始 demonstration 与后续 rollout。
        current_episodes = [
            dataclasses.replace(
                episode,
                metadata={**episode.metadata, "recap_source": episode.metadata.get("recap_source", "demo")},
            )
            for episode in episodes
        ]
        for iteration in range(self.num_iterations):
            iteration_dir = self.output_dir / f"iter_{iteration:03d}"
            if iteration_dir.exists():
                # 在线数据不可静默覆盖或与旧 JSON 混合; 重跑必须选择新输出目录。
                raise FileExistsError(f"RECAP iteration output already exists: {iteration_dir}")
            episodes_dir = iteration_dir / "episodes"
            value_dir = iteration_dir / "value"
            labels_dir = iteration_dir / "labels"
            policy_dir = iteration_dir / "policy"
            # 固定 artifact 目录由 runner 负责创建, 自定义 hook 不应改变跨阶段路径契约。
            for directory in (episodes_dir, value_dir, labels_dir, policy_dir):
                directory.mkdir(parents=True, exist_ok=False)
            training_episode_count = len(current_episodes)
            self._write_episodes(episodes_dir, current_episodes)
            # 固定顺序对应 Algorithm 1, artifact 路径作为阶段间唯一数据契约。
            self.hooks.train_value(episodes_dir, value_dir)
            self.hooks.label_advantage(episodes_dir, value_dir, labels_dir)
            policy_result = self.hooks.train_policy(episodes_dir, labels_dir, policy_dir)
            if policy_result is None:
                policy_checkpoint = policy_dir
            elif isinstance(policy_result, str | Path):
                policy_checkpoint = Path(policy_result)
                if not policy_checkpoint.exists():
                    raise FileNotFoundError(f"train_policy returned a missing checkpoint: {policy_checkpoint}")
            else:
                raise TypeError("train_policy must return a checkpoint path or None")
            collected_count = 0
            if self.collect_episodes:
                rollout_dir = iteration_dir / "rollout"
                # 生产训练返回具体 Orbax step, 采集工厂无需猜测多层 checkpoint 目录。
                result = self.hooks.collect_rollout(policy_checkpoint, rollout_dir, self.collect_episodes)
                rollout_episodes = load_episodes(rollout_dir) if result is None else list(result)
                if not rollout_episodes:
                    raise ValueError("collect_rollout produced no episodes")
                if len(rollout_episodes) != self.collect_episodes:
                    raise ValueError(
                        "collect_rollout episode count mismatch: "
                        f"requested {self.collect_episodes}, got {len(rollout_episodes)}"
                    )
                rollout_episodes = [
                    dataclasses.replace(
                        episode,
                        metadata={**episode.metadata, "recap_source": "rollout"},
                    )
                    for episode in rollout_episodes
                ]
                historical_ids = {episode.episode_id for episode in current_episodes}
                duplicate_ids = sorted(
                    episode.episode_id for episode in rollout_episodes if episode.episode_id in historical_ids
                )
                if duplicate_ids:
                    # 最后一轮也必须立即拒绝重复身份, 不能留下下一轮无法聚合的历史集。
                    raise ValueError(f"collected episode ids already exist in historical data: {duplicate_ids}")
                # 无论 hook 返回内存对象还是自行落盘, 都统一写回带 source 标记的 schema。
                self._write_episodes(rollout_dir, rollout_episodes)
                current_episodes = [*current_episodes, *rollout_episodes]
                collected_count = len(rollout_episodes)
            history.append(
                {
                    "iteration": iteration,
                    "episodes": len(current_episodes),
                    "training_episodes": training_episode_count,
                    "aggregated_episodes": len(current_episodes),
                    "collected_episodes": collected_count,
                    "value": value_dir,
                    "labels": labels_dir,
                    "policy": policy_dir,
                    "policy_checkpoint": policy_checkpoint,
                }
            )
        return history
