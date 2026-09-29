"""Compute Pi05 normalization statistics for the reconstructed UR7 dataset.

This is the parquet-only counterpart of ``compute_norm_stats.py``.  It applies
the exact state/action conversion used by ``LeRobotUR7DataConfig`` while
avoiding video decoding, which makes a full dataset pass deterministic and
fast enough to run before a cloud training job.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import pathlib

import compute_norm_stats_fast as fast
import numpy as np
import polars as pl
import tqdm
import tyro

from openpi.policies import ur7_policy
from openpi.shared import normalize


@dataclasses.dataclass(frozen=True)
class CliArgs:
    data_path: pathlib.Path
    output_path: pathlib.Path
    action_horizon: int = 50
    action_dim: int = ur7_policy.MODEL_ACTION_DIM
    num_workers: int = 8


@dataclasses.dataclass(frozen=True)
class FileResult:
    state: np.ndarray
    actions: np.ndarray


def _process_file(path: pathlib.Path, args: CliArgs) -> FileResult:
    schema = pl.read_parquet_schema(path)
    if "observation.state" not in schema or "action" not in schema:
        raise ValueError(f"{path} must contain observation.state and action columns")
    table = pl.read_parquet(path, columns=["observation.state", "action"])
    state_raw = fast._series_to_matrix(  # noqa: SLF001
        table["observation.state"], column="observation.state", path=path
    ).astype(
        np.float32
    )
    action_raw = fast._series_to_matrix(table["action"], column="action", path=path).astype(np.float32)  # noqa: SLF001
    if state_raw.shape[0] != action_raw.shape[0]:
        raise ValueError(f"{path} state/action row count mismatch: {state_raw.shape} vs {action_raw.shape}")
    if state_raw.shape[-1] != ur7_policy.RAW_STATE_DIM or action_raw.shape[-1] != ur7_policy.ACTION_DIM:
        raise ValueError(
            f"{path} has unexpected dimensions: state={state_raw.shape[-1]}, action={action_raw.shape[-1]}"
        )

    state = ur7_policy.canonicalize_state(state_raw)
    actions_absolute = fast._window_actions(action_raw, args.action_horizon)  # noqa: SLF001
    actions_relative = ur7_policy.absolute_to_relative(
        actions_absolute, state[:, None, :]
    )
    state = ur7_policy._pad_last_dim(state, args.action_dim)  # noqa: SLF001
    actions_relative = ur7_policy._pad_last_dim(actions_relative, args.action_dim)  # noqa: SLF001
    return FileResult(state=state, actions=actions_relative)


def _consume(result: FileResult, stats: dict[str, normalize.RunningStats]) -> int:
    stats["state"].update(result.state)
    stats["actions"].update(result.actions)
    return result.state.shape[0]


def main(args: CliArgs) -> None:
    data_path = args.data_path.expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(data_path)
    if args.action_horizon <= 0 or args.action_dim < ur7_policy.ACTION_DIM:
        raise ValueError(f"invalid action_horizon/action_dim: {args.action_horizon}/{args.action_dim}")
    if args.num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {args.num_workers}")

    files = fast._collect_dataset_files(data_path)  # noqa: SLF001
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    progress = tqdm.tqdm(total=len(files), desc="Computing UR7 parquet stats")
    total_steps = 0
    try:
        if args.num_workers == 1:
            for path in files:
                total_steps += _consume(_process_file(path, args), stats)
                progress.update(1)
        else:
            pending: deque[Future[FileResult]] = deque()
            with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
                iterator = iter(files)
                for _ in range(min(args.num_workers * 2, len(files))):
                    pending.append(pool.submit(_process_file, next(iterator), args))
                for path in iterator:
                    total_steps += _consume(pending.popleft().result(), stats)
                    progress.update(1)
                    pending.append(pool.submit(_process_file, path, args))
                while pending:
                    total_steps += _consume(pending.popleft().result(), stats)
                    progress.update(1)
    finally:
        progress.close()

    if total_steps < 2:
        raise ValueError(f"insufficient samples: {total_steps}")
    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    normalize.save(args.output_path, norm_stats)
    print(f"Writing stats to: {args.output_path}")
    print(f"Processed {len(files)} parquet files and {total_steps} frames")


if __name__ == "__main__":
    main(tyro.cli(CliArgs))
