"""通用 RECAP rollout 采集入口, 不绑定具体机器人驱动。"""

from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path
import sys

import numpy as np

# 直接执行脚本时显式加入仓库根目录, 使用户的本地 module:function 工厂可被动态加载。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpi.recap.collector import ReCAPRolloutCollector


def load_callable(spec: str):
    if ":" not in spec:
        raise ValueError(f"callable must use module:function syntax, got {spec!r}")
    module, function = spec.split(":", 1)
    return getattr(importlib.import_module(module), function)


def _construct(factory, args: argparse.Namespace):
    # 按签名决定是否传 Namespace, 避免吞掉工厂内部 TypeError 后重复执行副作用。
    try:
        parameters = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return factory(args)
    accepts_argument = any(parameter.kind is parameter.VAR_POSITIONAL for parameter in parameters) or any(
        parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        for parameter in parameters
    )
    return factory(args) if accepts_argument else factory()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect hardware-agnostic RECAP episodes")
    parser.add_argument("--environment", required=True, help="module:function returning ReCAPEnvironment")
    parser.add_argument("--policy", required=True, help="module:function returning policy with infer(observation)")
    parser.add_argument("--intervention", help="optional module:function returning intervention callback")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="")
    parser.add_argument("--max-episode-length", type=int, default=1000)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--smooth-interventions", action="store_true")
    parser.add_argument("--action-low", type=float, nargs="+", help="per-dimension lower action bounds")
    parser.add_argument("--action-high", type=float, nargs="+", help="per-dimension upper action bounds")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    environment = _construct(load_callable(args.environment), args)
    policy = _construct(load_callable(args.policy), args)
    intervention = None
    if args.intervention:
        intervention = _construct(load_callable(args.intervention), args)
        if not callable(intervention):
            raise TypeError("intervention factory must return a callable")
    collector = ReCAPRolloutCollector(
        environment,
        policy,
        task=args.task,
        intervention_callback=intervention,
        action_low=None if args.action_low is None else np.asarray(args.action_low, dtype=np.float32),
        action_high=None if args.action_high is None else np.asarray(args.action_high, dtype=np.float32),
        smoothing=args.smoothing,
        smooth_interventions=args.smooth_interventions,
        max_episode_length=args.max_episode_length,
        seed=args.seed,
        output_dir=args.output,
    )
    episodes = collector.collect(args.episodes)
    print({"episodes": len(episodes), "frames": sum(len(item.frames) for item in episodes)})


if __name__ == "__main__":
    main()
