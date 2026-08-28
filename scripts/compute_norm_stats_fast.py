"""Compute normalization statistics directly from LeRobot parquet files.

This command avoids loading videos and training configuration. It applies the
pre-normalization ALOHA transforms used by OpenPI, then writes a standard
``norm_stats.json`` file.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import polars as pl
import tqdm
import tyro

from openpi.shared import normalize

_STATE_COLUMNS = (
    "observation.state",
    "observation.state_tcp_endpose_quat",
    "observation.state_vr_endpose_quat",
)
_ACTION_COLUMNS = ("action", "action_tcp_endpose_quat", "action_vr_endpose_quat")
_JOINT_FLIP_MASK = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


@dataclasses.dataclass(frozen=True)
class CliArgs:
    """Command-line arguments."""

    output_path: pathlib.Path
    data_path: pathlib.Path | None = None
    data_paths: list[pathlib.Path] | None = None
    action_horizon: int = 50
    action_dim: int = 32
    mask_list: list[int] | None = None
    zero_mask_list: list[int] | None = None
    adapt_to_pi: bool = False
    num_workers: int = 16


@dataclasses.dataclass(frozen=True)
class TransformOptions:
    """Validated transform settings shared by all parquet workers."""

    action_horizon: int
    action_dim: int
    delta_mask: tuple[bool, ...] | None
    zero_mask: tuple[bool, ...] | None
    adapt_to_pi: bool


@dataclasses.dataclass(frozen=True)
class FileResult:
    """Transformed values from one parquet file."""

    state: np.ndarray
    actions: np.ndarray

    @property
    def num_steps(self) -> int:
        return self.state.shape[0]


def _make_bool_mask(dims: list[int] | None, *, name: str) -> tuple[bool, ...] | None:
    if dims is None:
        return None
    if not dims:
        raise ValueError(f"{name} must not be empty")
    if any(dim == 0 for dim in dims):
        raise ValueError(f"{name} entries must be non-zero, got {dims}")

    mask: list[bool] = []
    for dim in dims:
        mask.extend([dim > 0] * abs(dim))
    return tuple(mask)


def _validate_args(args: CliArgs) -> tuple[list[pathlib.Path], TransformOptions]:
    data_paths = ([args.data_path] if args.data_path is not None else []) + list(args.data_paths or [])
    if not data_paths:
        raise ValueError("Provide --data-path or --data-paths")
    if args.action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {args.action_horizon}")
    if args.action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {args.action_dim}")
    if args.num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {args.num_workers}")

    resolved_paths = [path.expanduser().resolve() for path in data_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError(f"Duplicate data paths are not allowed: {resolved_paths}")
    for data_path in resolved_paths:
        if not data_path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {data_path}")

    delta_mask = _make_bool_mask(args.mask_list, name="mask_list")
    zero_mask = _make_bool_mask(args.zero_mask_list, name="zero_mask_list")
    if delta_mask is not None and len(delta_mask) > args.action_dim:
        raise ValueError(
            f"mask_list expands to {len(delta_mask)} dimensions, greater than action_dim={args.action_dim}"
        )
    if zero_mask is not None and len(zero_mask) != args.action_dim:
        raise ValueError(f"zero_mask_list must expand to action_dim={args.action_dim}, got {len(zero_mask)} dimensions")

    return resolved_paths, TransformOptions(
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        delta_mask=delta_mask,
        zero_mask=zero_mask,
        adapt_to_pi=args.adapt_to_pi,
    )


def _read_dataset_info(data_path: pathlib.Path) -> dict[str, Any] | None:
    info_path = data_path / "meta" / "info.json"
    if not info_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid dataset metadata from {info_path}: {exc}") from exc
    if not isinstance(info, dict):
        raise ValueError(f"Dataset metadata must be a JSON object: {info_path}")
    return info


def _collect_dataset_files(data_path: pathlib.Path) -> list[pathlib.Path]:
    info = _read_dataset_info(data_path)
    if info is None:
        search_root = data_path / "data" if (data_path / "data").is_dir() else data_path
        files = sorted(search_root.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {search_root}")
        return files

    try:
        total_episodes = int(info["total_episodes"])
        chunks_size = int(info.get("chunks_size", 1000))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid total_episodes/chunks_size in {data_path / 'meta' / 'info.json'}") from exc
    if total_episodes <= 0 or chunks_size <= 0:
        raise ValueError(f"total_episodes and chunks_size must be positive, got {total_episodes=} and {chunks_size=}")

    data_template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    if not isinstance(data_template, str):
        raise ValueError(f"data_path in dataset metadata must be a string, got {type(data_template).__name__}")

    files: list[pathlib.Path] = []
    missing: list[pathlib.Path] = []
    for episode_index in range(total_episodes):
        try:
            relative_path = data_template.format(
                episode_chunk=episode_index // chunks_size,
                episode_index=episode_index,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Unsupported data_path template in {data_path / 'meta' / 'info.json'}: {data_template}"
            ) from exc
        parquet_path = data_path / relative_path
        if parquet_path.is_file():
            files.append(parquet_path)
        else:
            missing.append(parquet_path)

    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"Dataset metadata declares {total_episodes} episodes, but {len(missing)} parquet files are missing: "
            f"{preview}{suffix}"
        )
    return files


def _collect_parquet_files(data_paths: list[pathlib.Path]) -> list[pathlib.Path]:
    files = [file for data_path in data_paths for file in _collect_dataset_files(data_path)]
    files.sort()
    if len(set(files)) != len(files):
        raise ValueError("The selected datasets contain duplicate parquet files")
    return files


def _pad_or_truncate(x: np.ndarray, target_dim: int) -> np.ndarray:
    if x.shape[-1] >= target_dim:
        return x[..., :target_dim]
    padding = np.zeros((*x.shape[:-1], target_dim - x.shape[-1]), dtype=x.dtype)
    return np.concatenate([x, padding], axis=-1)


def _window_actions(actions: np.ndarray, horizon: int) -> np.ndarray:
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError(f"Expected non-empty rank-2 actions, got shape {actions.shape}")
    indices = np.minimum(
        np.arange(actions.shape[0])[:, None] + np.arange(horizon)[None, :],
        actions.shape[0] - 1,
    )
    return actions[indices]


def _normalize_range(x: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    return (x - min_value) / (max_value - min_value)


def _unnormalize_range(x: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    return x * (max_value - min_value) + min_value


def _gripper_to_angular(value: np.ndarray) -> np.ndarray:
    value = _unnormalize_range(value, min_value=0.01844, max_value=0.05800)
    arm_length, horn_radius = 0.036, 0.022
    cosine = (horn_radius**2 + value**2 - arm_length**2) / (2 * horn_radius * value)
    value = np.arcsin(np.clip(cosine, -1.0, 1.0))
    return _normalize_range(value, min_value=0.5476, max_value=1.6296)


def _gripper_from_angular_inverse(value: np.ndarray) -> np.ndarray:
    value = _unnormalize_range(value, min_value=-0.6213, max_value=1.4910)
    return value - 0.5476


def _adapt_state_to_pi(state: np.ndarray) -> np.ndarray:
    if state.shape[-1] < len(_JOINT_FLIP_MASK):
        raise ValueError(f"adapt_to_pi requires at least 14 state dimensions, got {state.shape[-1]}")
    front = state[..., :14] * _JOINT_FLIP_MASK
    front[..., [6, 13]] = _gripper_to_angular(front[..., [6, 13]])
    return np.concatenate([front, state[..., 14:]], axis=-1) if state.shape[-1] > 14 else front


def _adapt_actions_to_pi(actions: np.ndarray) -> np.ndarray:
    if actions.shape[-1] < len(_JOINT_FLIP_MASK):
        raise ValueError(f"adapt_to_pi requires at least 14 action dimensions, got {actions.shape[-1]}")
    front = actions[..., :14] * _JOINT_FLIP_MASK
    front[..., [6, 13]] = _gripper_from_angular_inverse(front[..., [6, 13]])
    return np.concatenate([front, actions[..., 14:]], axis=-1) if actions.shape[-1] > 14 else front


def _apply_transforms(
    states_raw: np.ndarray,
    actions_windowed: np.ndarray,
    options: TransformOptions,
) -> tuple[np.ndarray, np.ndarray]:
    if options.adapt_to_pi:
        states_raw = _adapt_state_to_pi(states_raw)
        actions_windowed = _adapt_actions_to_pi(actions_windowed)

    state = _pad_or_truncate(states_raw, options.action_dim)
    actions = _pad_or_truncate(actions_windowed, options.action_dim)

    if options.zero_mask is not None:
        zero_mask = np.asarray(options.zero_mask, dtype=bool)
        state[:, zero_mask] = 0
        actions[:, :, zero_mask] = 0

    if options.delta_mask is not None:
        delta_mask = np.asarray(options.delta_mask, dtype=bool)
        dims = len(delta_mask)
        state_base = np.where(delta_mask, state[:, :dims], 0)
        actions[:, :, :dims] -= state_base[:, None, :]

    return state, actions


def _select_column(columns: set[str], candidates: tuple[str, ...], *, kind: str, path: pathlib.Path) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No supported {kind} column found in {path}; expected one of {candidates}")


def _series_to_matrix(series: pl.Series, *, column: str, path: pathlib.Path) -> np.ndarray:
    values = series.to_numpy()
    if values.ndim == 1 and values.dtype == object:
        parsed: list[np.ndarray] = []
        for value in values:
            if isinstance(value, bytes):
                parsed.append(np.frombuffer(value, dtype=np.float64).copy())
            else:
                parsed.append(np.asarray(value, dtype=np.float64))
        try:
            values = np.stack(parsed)
        except ValueError as exc:
            raise ValueError(f"Column {column!r} contains inconsistent vector dimensions in {path}") from exc

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Column {column!r} must contain non-empty vectors in {path}, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        invalid_count = int(np.size(values) - np.count_nonzero(np.isfinite(values)))
        raise ValueError(f"Column {column!r} contains {invalid_count} non-finite values in {path}")
    return values


def _process_parquet(path: pathlib.Path, options: TransformOptions) -> FileResult:
    try:
        schema = pl.read_parquet_schema(path)
        columns = set(schema)
        state_column = _select_column(columns, _STATE_COLUMNS, kind="state", path=path)
        action_column = _select_column(columns, _ACTION_COLUMNS, kind="action", path=path)
        selected_columns = [state_column, action_column]
        if "episode_index" in columns:
            selected_columns.append("episode_index")
        table = pl.read_parquet(path, columns=selected_columns)
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f"Cannot read parquet file {path}: {exc}") from exc

    states_raw = _series_to_matrix(table[state_column], column=state_column, path=path)
    actions_raw = _series_to_matrix(table[action_column], column=action_column, path=path)
    if states_raw.shape[0] != actions_raw.shape[0]:
        raise ValueError(f"State/action row count mismatch in {path}: {states_raw.shape[0]} != {actions_raw.shape[0]}")

    episode_indices = table["episode_index"].to_numpy() if "episode_index" in table.columns else None
    unique_episodes = np.unique(episode_indices) if episode_indices is not None else np.array([0])
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    for episode_index in unique_episodes:
        if episode_indices is None:
            episode_states = states_raw
            episode_actions = actions_raw
        else:
            episode_mask = episode_indices == episode_index
            episode_states = states_raw[episode_mask]
            episode_actions = actions_raw[episode_mask]
        actions_windowed = _window_actions(episode_actions, options.action_horizon)
        state, actions = _apply_transforms(episode_states, actions_windowed, options)
        all_states.append(state)
        all_actions.append(actions)

    return FileResult(state=np.concatenate(all_states), actions=np.concatenate(all_actions))


class NormStatsComputer:
    """Read parquet files concurrently and accumulate deterministic statistics."""

    def __init__(self, args: CliArgs):
        self._args = args
        self._data_paths, self._options = _validate_args(args)

    def run(self) -> dict[str, normalize.NormStats]:
        parquet_files = _collect_parquet_files(self._data_paths)
        print(f"Found {len(parquet_files)} parquet files from {len(self._data_paths)} data path(s)")
        print(
            f"action_horizon={self._options.action_horizon}, action_dim={self._options.action_dim}, "
            f"num_workers={self._args.num_workers}"
        )

        stats = {key: normalize.RunningStats() for key in ("state", "actions")}
        total_steps = 0
        progress = tqdm.tqdm(total=len(parquet_files), desc="Processing parquet files")
        try:
            if self._args.num_workers == 1:
                for path in parquet_files:
                    total_steps += self._consume(_process_parquet(path, self._options), stats)
                    progress.update(1)
            else:
                total_steps = self._run_parallel(parquet_files, stats, progress)
        finally:
            progress.close()

        if total_steps < 2:
            raise ValueError(f"Insufficient valid samples to compute normalization stats: {total_steps}")

        norm_stats = {key: running.get_statistics() for key, running in stats.items()}
        normalize.save(self._args.output_path, norm_stats)
        print(f"Writing stats to: {self._args.output_path}")
        print(f"Processed {len(parquet_files)}/{len(parquet_files)} files, {total_steps} total steps")
        return norm_stats

    def _run_parallel(
        self,
        parquet_files: list[pathlib.Path],
        stats: dict[str, normalize.RunningStats],
        progress: tqdm.tqdm,
    ) -> int:
        total_steps = 0
        file_iterator = iter(parquet_files)
        pending: deque[Future[FileResult]] = deque()
        max_in_flight = max(self._args.num_workers * 2, self._args.num_workers + 1)
        with ThreadPoolExecutor(max_workers=self._args.num_workers) as executor:
            for _ in range(min(max_in_flight, len(parquet_files))):
                pending.append(executor.submit(_process_parquet, next(file_iterator), self._options))

            for path in file_iterator:
                total_steps += self._consume(pending.popleft().result(), stats)
                progress.update(1)
                pending.append(executor.submit(_process_parquet, path, self._options))

            while pending:
                total_steps += self._consume(pending.popleft().result(), stats)
                progress.update(1)
        return total_steps

    @staticmethod
    def _consume(result: FileResult, stats: dict[str, normalize.RunningStats]) -> int:
        stats["state"].update(result.state)
        stats["actions"].update(result.actions)
        return result.num_steps


def main(args: CliArgs) -> None:
    NormStatsComputer(args).run()


if __name__ == "__main__":
    main(tyro.cli(CliArgs))
