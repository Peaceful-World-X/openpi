"""RECAP Algorithm 1 编排脚本。

生产环境默认调用 ``uv run scripts/train.py``; 单元测试可用 ``--hook-factory`` 注入纯 Python 阶段函数。
"""

from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path
import shlex
import subprocess
import sys

# 脚本直接执行时 Python 只把 scripts/ 放入 sys.path, 补入仓库根目录以复用同目录 CLI 模块。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpi.recap.pipeline import OnlineReCAPRunner
from openpi.recap.pipeline import ReCAPPipelineHooks
from openpi.recap.pipeline import _latest_policy_checkpoint


def _load(spec: str):
    module, function = spec.split(":", 1)
    return getattr(importlib.import_module(module), function)


def _call_factory(factory, args: argparse.Namespace):
    # 只在工厂签名接受参数时传 Namespace, 保留工厂内部异常的真实堆栈。
    try:
        parameters = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return factory(args)
    accepts_argument = any(parameter.kind is parameter.VAR_POSITIONAL for parameter in parameters) or any(
        parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        for parameter in parameters
    )
    return factory(args) if accepts_argument else factory()


def _production_hooks(args: argparse.Namespace) -> ReCAPPipelineHooks:
    # 延迟加载 JAX/Orbax 训练模块, 让 ``--help`` 和 hook-only 测试无需初始化 GPU。
    from scripts.label_recap_advantage import label_recap
    from scripts.train_value import train_value

    collect_factory = _load(args.collect_factory) if args.collect_factory else None
    policy_data_factory = _load(args.policy_data_factory) if args.policy_data_factory else None
    def value_stage(episodes_dir: Path, value_dir: Path):
        return train_value(
            episodes_dir,
            value_dir,
            steps=args.value_steps,
            batch_size=args.value_batch_size,
            learning_rate=args.value_learning_rate,
            gradient_clip_norm=args.value_gradient_clip_norm,
            freeze_mode=args.value_freeze_mode,
            eval_batch_size=args.value_eval_batch_size,
            reward_mode=args.reward_mode,
            failure_penalty=args.failure_penalty,
            init_checkpoint=args.value_base_checkpoint,
            siglip_variant=args.value_siglip_variant,
            gemma_variant=args.value_gemma_variant,
            dtype=args.value_dtype,
        )

    def label_stage(episodes_dir: Path, value_dir: Path, labels_dir: Path):
        return label_recap(
            episodes_dir,
            labels_dir,
            value_checkpoint=value_dir,
            positive_fraction=args.positive_fraction,
            n_step_lookahead=args.n_step_lookahead,
            inference_batch_size=args.value_inference_batch_size,
            reward_mode=args.reward_mode,
            failure_penalty=args.failure_penalty,
        )

    def policy_stage(episodes_dir: Path, labels_dir: Path, policy_dir: Path):
        repo_id = args.policy_repo_id
        if policy_data_factory is not None:
            # 工厂负责把聚合 JSON 物化为与 sidecar 身份一致的 LeRobot 数据集。
            repo_id = policy_data_factory(episodes_dir, labels_dir, policy_dir, repo_id)
        if not repo_id:
            raise ValueError(
                "production policy training requires --policy-repo-id or --policy-data-factory; "
                "use --hook-factory for tests or choose debug_pi05_recap explicitly"
            )
        policy_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "uv",
            "run",
            "scripts/train.py",
            args.policy_config,
            "--checkpoint-base-dir",
            str(policy_dir),
            "--exp-name",
            f"recap_{policy_dir.parent.name}",
            "--data.repo-id",
            str(repo_id),
            "--data.recap-fields-path",
            str(labels_dir / "lerobot_fields.npz"),
        ]
        # 标签 sidecar 与 policy 数据集由用户配置关联; 命令保持可审计并按需执行。
        (policy_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        subprocess.run(command, check=True)
        run_dir = policy_dir / args.policy_config / f"recap_{policy_dir.parent.name}"
        checkpoint = _latest_policy_checkpoint(run_dir)
        # 固定指针便于脱离 runner 审计和手工继续下一轮。
        (policy_dir / "checkpoint_path.txt").write_text(f"{checkpoint}\n", encoding="utf-8")
        return checkpoint

    def collect_stage(policy_checkpoint: Path, rollout_dir: Path, count: int):
        if collect_factory is None:
            raise ValueError("--collect-episodes requires --collect-factory module:function")
        return collect_factory(policy_checkpoint, rollout_dir, count)

    return ReCAPPipelineHooks(
        train_value=value_stage,
        label_advantage=label_stage,
        train_policy=policy_stage,
        collect_rollout=collect_stage if collect_factory is not None else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the iterative RECAP online RL loop")
    parser.add_argument("--demo-episodes", required=True)
    parser.add_argument("--output-dir", default="outputs/recap")
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--collect-episodes", type=int, default=0)
    parser.add_argument("--value-steps", type=int, default=1000)
    parser.add_argument("--value-batch-size", type=int, default=32)
    parser.add_argument("--value-eval-batch-size", type=int, default=32)
    parser.add_argument("--value-inference-batch-size", type=int, default=32)
    parser.add_argument("--value-learning-rate", type=float, default=1e-4)
    parser.add_argument("--value-gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--value-freeze-mode", choices=("none", "backbones"), default="none")
    parser.add_argument("--value-base-checkpoint", help="每轮重新 fine-tune 使用的 V_pre checkpoint")
    parser.add_argument("--value-siglip-variant", default="So400m/14")
    parser.add_argument("--value-gemma-variant", default="gemma_300m")
    parser.add_argument("--value-dtype", default="bfloat16")
    parser.add_argument("--reward-mode", choices=("paper", "environment"), default="paper")
    parser.add_argument("--failure-penalty", type=float)
    parser.add_argument("--positive-fraction", type=float, default=0.4)
    parser.add_argument("--n-step-lookahead", type=int, default=50)
    parser.add_argument("--policy-config", default="pi05_recap")
    parser.add_argument("--policy-repo-id", help="LeRobot repo id used by the policy stage")
    parser.add_argument(
        "--policy-data-factory",
        help="module:function(episodes_dir, labels_dir, policy_dir, base_repo_id) returning aligned LeRobot repo id",
    )
    parser.add_argument("--collect-factory", help="module:function(policy_dir, rollout_dir, count) for online rollout")
    parser.add_argument("--hook-factory", help="module:function returning ReCAPPipelineHooks")
    args = parser.parse_args()
    if not args.hook_factory and args.collect_episodes and args.num_iterations > 1 and not args.policy_data_factory:
        raise ValueError("multi-iteration online training requires --policy-data-factory to ingest collected JSON")
    if args.hook_factory:
        factory = _load(args.hook_factory)
        hooks = _call_factory(factory, args)
    else:
        hooks = _production_hooks(args)
    result = OnlineReCAPRunner(
        args.demo_episodes,
        args.output_dir,
        num_iterations=args.num_iterations,
        collect_episodes=args.collect_episodes,
        hooks=hooks,
    ).run()
    print(result)


if __name__ == "__main__":
    main()
