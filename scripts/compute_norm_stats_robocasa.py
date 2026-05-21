#!/usr/bin/env python3
"""Fast normalization statistics for RoboCasa LeRobot datasets only.

Expected layout:
    <pretrain>/atomic/<task>/<date>/lerobot/{meta,data,videos}
    <pretrain>/composite/<task>/<date>/lerobot/{meta,data,videos}

This script intentionally avoids slow shared-disk metadata scans:
- no recursive rglob over data/videos
- no per-episode Path.is_file() checks
- no per-parquet reload of meta/modality.json
- minimal per-repo logging
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import tqdm

import openpi.shared.normalize as normalize


Component = tuple[str, str, int, int]


@dataclasses.dataclass(frozen=True)
class RepoSpec:
    root: pathlib.Path
    state_components: list[Component]
    action_components: list[Component]
    data_path_template: str
    chunks_size: int
    episode_indices: list[int]


@dataclasses.dataclass(frozen=True)
class ParquetItem:
    path: pathlib.Path
    state_components: list[Component]
    action_components: list[Component]


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _is_lerobot_repo(path: pathlib.Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (path / "meta" / "modality.json").is_file()


def discover_lerobot_repos(data_root: str, subset: str | None = None) -> list[pathlib.Path]:
    root = pathlib.Path(data_root)
    search_root = root / subset if subset else root

    if not search_root.exists():
        raise FileNotFoundError(f"RoboCasa root does not exist: {search_root}")

    if _is_lerobot_repo(search_root):
        repos = [search_root]
    else:
        # Fast fixed-layout discovery. Do not use rglob here.
        patterns = (
            "atomic/*/*/lerobot",
            "composite/*/*/lerobot",
            "*/*/lerobot",
        )
        seen: set[str] = set()
        repos: list[pathlib.Path] = []
        for pattern in patterns:
            for repo in search_root.glob(pattern):
                if not _is_lerobot_repo(repo):
                    continue
                key = str(repo)
                if key not in seen:
                    repos.append(repo)
                    seen.add(key)
        repos.sort(key=lambda p: str(p))

    if not repos:
        raise FileNotFoundError(
            f"No RoboCasa LeRobot repos found under {search_root}. "
            "Expected atomic/<task>/<date>/lerobot or composite/<task>/<date>/lerobot."
        )

    print(f"[INFO] Discovered {len(repos)} RoboCasa LeRobot repo(s) under {search_root}", flush=True)
    if len(repos) <= 10:
        for repo in repos:
            print(f"[INFO]   {repo}", flush=True)
    else:
        for repo in repos[:5]:
            print(f"[INFO]   {repo}", flush=True)
        print(f"[INFO]   ... {len(repos) - 5} more", flush=True)

    return repos


def _ordered_components(section: dict[str, Any]) -> list[Component]:
    items: list[Component] = []
    for name, entry in section.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid modality component {name!r}: {entry!r}")
        original_key = entry.get("original_key")
        if original_key is None:
            raise ValueError(f"Missing original_key for modality component {name!r}")
        start = int(entry.get("start", 0))
        end = int(entry.get("end", 0))
        if end <= start:
            raise ValueError(f"Invalid slice for {name!r}: start={start}, end={end}")
        items.append((name, str(original_key), start, end))
    items.sort(key=lambda item: item[2])
    return items


def _episode_indices_from_metadata(repo_root: pathlib.Path, info: dict[str, Any]) -> list[int]:
    if "total_episodes" in info:
        return list(range(int(info["total_episodes"])))

    episodes_path = repo_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        return []

    indices: list[int] = []
    with episodes_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "episode_index" in item:
                indices.append(int(item["episode_index"]))
    return sorted(set(indices))


def _format_data_path(template: str, episode_idx: int, chunks_size: int) -> str:
    episode_chunk = episode_idx // chunks_size
    return template.format(
        episode_index=episode_idx,
        episode_idx=episode_idx,
        episode_chunk=episode_chunk,
        chunk_index=episode_chunk,
        chunk_idx=episode_chunk,
    )


def load_repo_spec(repo_root: pathlib.Path) -> RepoSpec:
    info_path = repo_root / "meta" / "info.json"
    modality_path = repo_root / "meta" / "modality.json"

    info = _read_json(info_path)
    modality = _read_json(modality_path)

    state_components = _ordered_components(modality.get("state", {}))
    action_components = _ordered_components(modality.get("action", {}))
    if not state_components or not action_components:
        raise ValueError(f"Invalid modality file, missing state/action components: {modality_path}")

    chunks_size = int(info.get("chunks_size", 1000))
    data_path_template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    episode_indices = _episode_indices_from_metadata(repo_root, info)
    if not episode_indices:
        raise ValueError(f"No episode indices found from metadata: {info_path}")

    return RepoSpec(
        root=repo_root,
        state_components=state_components,
        action_components=action_components,
        data_path_template=str(data_path_template),
        chunks_size=chunks_size,
        episode_indices=episode_indices,
    )


def collect_parquet_items(repo_roots: list[pathlib.Path], max_repos: int | None = None) -> list[ParquetItem]:
    if max_repos is not None:
        repo_roots = repo_roots[:max_repos]

    items: list[ParquetItem] = []
    total_repos = len(repo_roots)

    for i, repo_root in enumerate(repo_roots, 1):
        spec = load_repo_spec(repo_root)

        # Fast path: trust meta/info.json and construct paths directly.
        # This avoids tens of thousands of Path.is_file() calls on shared disks.
        for episode_idx in spec.episode_indices:
            rel_path = _format_data_path(spec.data_path_template, episode_idx, spec.chunks_size)
            items.append(
                ParquetItem(
                    path=spec.root / rel_path,
                    state_components=spec.state_components,
                    action_components=spec.action_components,
                )
            )

        if i % 25 == 0 or i == total_repos:
            print(
                f"[INFO] Collected metadata from {i}/{total_repos} repo(s), "
                f"parquet_items={len(items)}",
                flush=True,
            )

    # Deduplicate paths while preserving component references.
    dedup: dict[pathlib.Path, ParquetItem] = {item.path: item for item in items}
    return [dedup[p] for p in sorted(dedup)]


def to_np(table: pq.Table, key: str) -> np.ndarray:
    data = table[key].to_pylist()
    try:
        arr = np.asarray(data, dtype=np.float32)
    except Exception:
        arr = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in data], axis=0)

    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    return arr.astype(np.float32, copy=False)


def _extract_vector(
    table: pq.Table,
    components: list[Component],
    group: str,
) -> np.ndarray:
    parts: list[np.ndarray] = []

    for name, original_key, start, end in components:
        key = original_key if original_key in table.column_names else f"{group}.{name}"
        if key not in table.column_names:
            raise KeyError(f"Missing parquet column {original_key!r} or {group}.{name!r}")

        vec = to_np(table, key)
        if vec.shape[-1] < end:
            raise ValueError(f"{key} last dim should be >= {end}, got {vec.shape}")
        parts.append(vec[..., start:end].astype(np.float32, copy=False))

    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


def pad_last_dim(x: np.ndarray, dim: int) -> np.ndarray:
    if dim <= 0:
        return x.astype(np.float32, copy=False)
    if x.shape[-1] > dim:
        raise ValueError(f"Cannot pad shape {x.shape} to dim={dim}")
    if x.shape[-1] == dim:
        return x.astype(np.float32, copy=False)

    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, dim - x.shape[-1])
    return np.pad(x, pad_width, mode="constant").astype(np.float32, copy=False)


def make_action_windows_pad_tail(actions: np.ndarray, horizon: int) -> np.ndarray:
    if len(actions) == 0:
        return np.empty((0, horizon, actions.shape[-1]), dtype=np.float32)

    num_steps, raw_action_dim = actions.shape
    windows = np.empty((num_steps, horizon, raw_action_dim), dtype=np.float32)
    last = actions[-1]

    for t in range(num_steps):
        end = min(t + horizon, num_steps)
        valid = actions[t:end]
        windows[t, : len(valid)] = valid
        if len(valid) < horizon:
            windows[t, len(valid):] = last

    return windows


def process_one_parquet(item: ParquetItem, action_horizon: int, pad_dim: int) -> tuple[np.ndarray, np.ndarray]:
    required_columns = sorted(
        {original_key for _, original_key, _, _ in item.state_components + item.action_components}
    )
    table = pq.read_table(item.path, columns=required_columns)

    state_raw = _extract_vector(table, item.state_components, "state")
    action_raw = _extract_vector(table, item.action_components, "action")

    n = min(state_raw.shape[0], action_raw.shape[0])
    if n < 2:
        raise ValueError(f"Too few aligned samples: state={state_raw.shape}, action={action_raw.shape}")

    state_raw = state_raw[:n]
    action_raw = action_raw[:n]

    action_windows = make_action_windows_pad_tail(action_raw, action_horizon)
    state = pad_last_dim(state_raw, pad_dim)
    actions = pad_last_dim(action_windows, pad_dim)
    return state, actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", action="append", default=[], help="One RoboCasa LeRobot repo path. Can be repeated.")
    parser.add_argument("--data-root", default="/shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain")
    parser.add_argument("--subset", default=None, help="Optional subdirectory under data-root: atomic or composite.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=32, help="Padding dim. Use 32 for PI05/OpenPI.")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-repos", type=int, default=None)

    # Accepted but ignored so older commands keep working.
    parser.add_argument("--schema", default=None)
    parser.add_argument("--modality-json", default=None)
    parser.add_argument("--delta-action-mask-list", type=int, nargs="+", default=None)
    parser.add_argument("--use-base", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-delta-joint-actions", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.modality_json is not None:
        print("[WARN] --modality-json ignored; using each repo's meta/modality.json.", flush=True)
    if args.schema is not None:
        print("[WARN] --schema ignored; this script only supports RoboCasa LeRobot.", flush=True)

    if args.data_path:
        repo_roots = [pathlib.Path(p) for p in args.data_path]
        missing = [str(p) for p in repo_roots if not _is_lerobot_repo(p)]
        if missing:
            raise FileNotFoundError(f"Invalid RoboCasa LeRobot repo path(s): {missing}")
    else:
        repo_roots = discover_lerobot_repos(args.data_root, args.subset)

    parquet_items = collect_parquet_items(repo_roots, max_repos=args.max_repos)
    if args.max_files is not None:
        parquet_items = parquet_items[:args.max_files]
    if not parquet_items:
        raise FileNotFoundError("No parquet files found from RoboCasa metadata.")

    first_state_dim = max(end for _, _, _, end in parquet_items[0].state_components)
    first_action_dim = max(end for _, _, _, end in parquet_items[0].action_components)

    print(f"[INFO] Total parquet files: {len(parquet_items)}", flush=True)
    print(f"[INFO] raw_state_dim={first_state_dim}, raw_action_dim={first_action_dim}", flush=True)
    print(f"[INFO] action_horizon={args.action_horizon}, action_dim/pad_dim={args.action_dim}", flush=True)
    print(f"[INFO] num_workers={args.num_workers}", flush=True)
    print(f"[INFO] output_path={args.output_path}", flush=True)

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    total_steps = 0
    processed = 0
    skipped = 0

    def handle_result(result: tuple[np.ndarray, np.ndarray] | Exception) -> None:
        nonlocal total_steps, processed, skipped
        if isinstance(result, Exception):
            skipped += 1
            if skipped <= 20 or skipped % 100 == 0:
                print(f"[WARN] skipped={skipped}: {result}", flush=True)
            return

        state, actions = result
        stats["state"].update(state.reshape(-1, state.shape[-1]))
        stats["actions"].update(actions.reshape(-1, actions.shape[-1]))
        total_steps += int(state.shape[0])
        processed += 1

    if args.num_workers <= 1:
        for item in tqdm.tqdm(parquet_items, desc="Processing parquet files"):
            try:
                result = process_one_parquet(item, args.action_horizon, args.action_dim)
            except Exception as exc:
                result = exc
            handle_result(result)
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(process_one_parquet, item, args.action_horizon, args.action_dim)
                for item in parquet_items
            ]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing parquet files"):
                try:
                    result = future.result()
                except Exception as exc:
                    result = exc
                handle_result(result)

    print(f"[INFO] processed={processed}", flush=True)
    print(f"[INFO] skipped={skipped}", flush=True)
    print(f"[INFO] total_steps={total_steps}", flush=True)

    if total_steps < 2:
        raise ValueError("Insufficient valid samples (<2) to compute normalization stats.")

    norm_stats = {
        "state": stats["state"].get_statistics(),
        "actions": stats["actions"].get_statistics(),
    }

    output_path = pathlib.Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] state mean shape: {norm_stats['state'].mean.shape}", flush=True)
    print(f"[INFO] actions mean shape: {norm_stats['actions'].mean.shape}", flush=True)
    print(f"[INFO] Writing stats to: {output_path}", flush=True)

    normalize.save(output_path, norm_stats)
    print(f"[DONE] wrote {output_path / 'norm_stats.json'}", flush=True)


if __name__ == "__main__":
    main()
