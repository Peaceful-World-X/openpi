#!/usr/bin/env python3
"""Fast norm stats computation for EBench LeRobot parquet datasets.

This version is modified from the original compute_norm_stats_fast.py and supports:

1. Multi-dataset norm computation.
2. Auto-discovery of LeRobot datasets under an EBench root directory.
3. EBench official base information:
   state  = state.joints(12) + state.gripper(4) + state.base(3) = 19
   action = action.joints(12) + action.gripper(4) + action.base_delta(3) = 19
4. OpenPI PI05 padding to action_dim=32.
5. Delta action transform aligned with LeRobotEBenchTeleopDataConfig:
   - first 12 joint dims are converted to delta relative to current state
   - last 7 dims, gripper(4) + base_delta(3), are kept absolute / as-is
"""

from __future__ import annotations

import argparse
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq
import tqdm

import openpi.shared.normalize as normalize


def pad_last_dim(x: np.ndarray, dim: int) -> np.ndarray:
    if x.shape[-1] > dim:
        raise ValueError(f"Cannot pad shape {x.shape} to dim={dim}")
    if x.shape[-1] == dim:
        return x.astype(np.float32, copy=False)
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, dim - x.shape[-1])
    return np.pad(x, pad_width, mode="constant").astype(np.float32, copy=False)


def to_np(table: pq.Table, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist(), dtype=np.float32)


def make_action_windows_pad_tail(actions: np.ndarray, horizon: int) -> np.ndarray:
    """Create [T, horizon, D] action windows and pad tail with the last action.

    This matches the usual training behavior better than dropping the final horizon-1 states.
    """
    if len(actions) == 0:
        return np.empty((0, horizon, actions.shape[-1]), dtype=np.float32)

    num_steps, action_dim = actions.shape
    windows = np.empty((num_steps, horizon, action_dim), dtype=np.float32)
    last = actions[-1]

    for t in range(num_steps):
        end = min(t + horizon, num_steps)
        valid = actions[t:end]
        windows[t, : len(valid)] = valid
        if len(valid) < horizon:
            windows[t, len(valid) :] = last

    return windows


def discover_lerobot_datasets(
    data_root: str,
    subset: str | None = None,
) -> list[str]:
    root = pathlib.Path(data_root).resolve()
    search_root = root / subset if subset else root

    if not search_root.exists():
        raise FileNotFoundError(f"EBench search root does not exist: {search_root}")

    # A LeRobot dataset directory contains meta/info.json.
    repo_ids = sorted({str(p.parent.parent.resolve()) for p in search_root.rglob("meta/info.json")})

    if not repo_ids:
        raise FileNotFoundError(f"No LeRobot datasets found under: {search_root}")

    print(f"[INFO] Discovered {len(repo_ids)} LeRobot datasets under {search_root}")
    for repo_id in repo_ids:
        print(f"[INFO]   {repo_id}")

    return repo_ids


def collect_parquet_files(data_paths: list[str]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []

    for data_path in data_paths:
        root = pathlib.Path(data_path).resolve()
        info_path = root / "meta" / "info.json"

        if info_path.is_file():
            info = json.loads(info_path.read_text())
            total_episodes = int(info["total_episodes"])
            chunks_size = int(info.get("chunks_size", 1000))

            found = []
            for episode_idx in range(total_episodes):
                candidate = root / f"data/chunk-{episode_idx // chunks_size:03d}/episode_{episode_idx:06d}.parquet"
                if candidate.is_file():
                    found.append(candidate)

            # Fallback in case the local file naming differs from info.json.
            if len(found) == 0:
                found = sorted(root.glob("data/**/*.parquet"))
        else:
            found = sorted(root.glob("data/**/*.parquet"))

        print(f"[INFO] {root.name}: {len(found)} parquet files")
        files.extend(found)

    files = sorted(set(files))
    return files


def process_one_parquet(
    parquet_path: pathlib.Path,
    action_horizon: int,
    action_dim: int,
    use_base: bool,
    use_delta_joint_actions: bool,
):
    try:
        table = pq.read_table(parquet_path)
    except Exception as exc:
        print(f"[WARN] Skip unreadable parquet {parquet_path}: {exc}")
        return None

    # Norm statistics only depend on state/actions.
    # Do not require video columns here: in LeRobot video datasets, images/videos
    # can be stored outside parquet and materialized by LeRobotDataset at training time.
    required = [
        "state.joints",
        "state.gripper",
        "action.joints",
        "action.gripper",
    ]

    if use_base:
        required.extend(["state.base", "action.base_delta"])

    missing = [key for key in required if key not in table.column_names]
    if missing:
        print(f"[WARN] Skip {parquet_path}: missing columns {missing}")
        return None

    try:
        state_joints = to_np(table, "state.joints")          # [T, 12]
        state_gripper = to_np(table, "state.gripper")        # [T, 4]
        action_joints = to_np(table, "action.joints")        # [T, 12]
        action_gripper = to_np(table, "action.gripper")      # [T, 4]

        if state_joints.shape[-1] != 12:
            raise ValueError(f"state.joints dim should be 12, got {state_joints.shape}")
        if state_gripper.shape[-1] != 4:
            raise ValueError(f"state.gripper dim should be 4, got {state_gripper.shape}")
        if action_joints.shape[-1] != 12:
            raise ValueError(f"action.joints dim should be 12, got {action_joints.shape}")
        if action_gripper.shape[-1] != 4:
            raise ValueError(f"action.gripper dim should be 4, got {action_gripper.shape}")

        state_parts = [state_joints, state_gripper]
        action_parts = [action_joints, action_gripper]

        if use_base:
            state_base = to_np(table, "state.base")                  # [T, 3]
            action_base_delta = to_np(table, "action.base_delta")    # [T, 3]

            if state_base.shape[-1] != 3:
                raise ValueError(f"state.base dim should be 3, got {state_base.shape}")
            if action_base_delta.shape[-1] != 3:
                raise ValueError(f"action.base_delta dim should be 3, got {action_base_delta.shape}")

            state_parts.append(state_base)
            action_parts.append(action_base_delta)

        state_raw = np.concatenate(state_parts, axis=-1).astype(np.float32)       # [T, 19] or [T, 16]
        action_raw = np.concatenate(action_parts, axis=-1).astype(np.float32)     # [T, 19] or [T, 16]

        action_windows = make_action_windows_pad_tail(action_raw, action_horizon) # [T, H, 19]

        if use_delta_joint_actions:
            # DeltaActions(mask=make_bool_mask(12, -7)) for 19-dim:
            # action[:, :, :12] -= state[:, None, :12]
            action_windows = action_windows.copy()
            action_windows[:, :, :12] -= state_raw[:, None, :12]

        state = pad_last_dim(state_raw, action_dim)
        actions = pad_last_dim(action_windows, action_dim)

        return state, actions

    except Exception as exc:
        print(f"[WARN] Skip malformed parquet {parquet_path}: {exc}")
        return None


def parse_args():
    parser = argparse.ArgumentParser()

    # Option A: manually pass dataset paths.
    parser.add_argument(
        "--data-path",
        action="append",
        default=[],
        help="One LeRobot dataset path. Can be repeated. If omitted, --data-root auto-discovery is used.",
    )

    # Option B: auto-discover all LeRobot datasets under a root.
    parser.add_argument(
        "--data-root",
        default="/shared_disk/users/wenyao.xue/EBench-Dataset",
        help="Root directory used to auto-discover LeRobot datasets.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional subdirectory under data-root, e.g. teleop_tasks. If omitted, scan the whole root.",
    )

    parser.add_argument("--output-path", required=True)
    parser.add_argument("--action-horizon", type=int, default=20)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--use-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-delta-joint-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--max-files", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.data_path:
        data_paths = args.data_path
    else:
        data_paths = discover_lerobot_datasets(args.data_root, args.subset)

    parquet_files = collect_parquet_files(data_paths)

    if args.max_files is not None:
        parquet_files = parquet_files[: args.max_files]

    if not parquet_files:
        raise FileNotFoundError("No parquet files found.")

    print(f"[INFO] Total parquet files: {len(parquet_files)}")
    print(f"[INFO] action_horizon={args.action_horizon}")
    print(f"[INFO] action_dim={args.action_dim}")
    print(f"[INFO] use_base={args.use_base}")
    print(f"[INFO] use_delta_joint_actions={args.use_delta_joint_actions}")
    print(f"[INFO] num_workers={args.num_workers}")
    print(f"[INFO] output_path={args.output_path}")

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    total_steps = 0
    processed = 0
    skipped = 0

    if args.num_workers <= 1:
        for parquet_file in tqdm.tqdm(parquet_files, desc="Processing parquet files"):
            result = process_one_parquet(
                parquet_file,
                action_horizon=args.action_horizon,
                action_dim=args.action_dim,
                use_base=args.use_base,
                use_delta_joint_actions=args.use_delta_joint_actions,
            )

            if result is None:
                skipped += 1
                continue

            state, actions = result
            stats["state"].update(state.reshape(-1, state.shape[-1]))
            stats["actions"].update(actions.reshape(-1, actions.shape[-1]))
            total_steps += state.shape[0]
            processed += 1
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    process_one_parquet,
                    parquet_file,
                    args.action_horizon,
                    args.action_dim,
                    args.use_base,
                    args.use_delta_joint_actions,
                )
                for parquet_file in parquet_files
            ]

            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing parquet files"):
                result = future.result()

                if result is None:
                    skipped += 1
                    continue

                state, actions = result
                stats["state"].update(state.reshape(-1, state.shape[-1]))
                stats["actions"].update(actions.reshape(-1, actions.shape[-1]))
                total_steps += state.shape[0]
                processed += 1

    print(f"[INFO] processed={processed}")
    print(f"[INFO] skipped={skipped}")
    print(f"[INFO] total_steps={total_steps}")

    if total_steps < 2:
        raise ValueError("Insufficient valid samples (<2) to compute normalization stats")

    norm_stats = {
        "state": stats["state"].get_statistics(),
        "actions": stats["actions"].get_statistics(),
    }

    output_path = pathlib.Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] state mean shape: {norm_stats['state'].mean.shape}")
    print(f"[INFO] actions mean shape: {norm_stats['actions'].mean.shape}")
    print(f"[INFO] Writing stats to: {output_path}")

    normalize.save(output_path, norm_stats)
    print(f"[DONE] wrote {output_path / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
