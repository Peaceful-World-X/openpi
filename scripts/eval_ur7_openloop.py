#!/usr/bin/env python3
"""Open-loop evaluation and visual report for the trained UR7 Pi05 policy.

The evaluator uses the same ``LeRobotUR7DataConfig`` transforms as training:
the policy receives raw 34D state plus base/wrist images, while
``UR7Inputs`` converts actions to relative EEF coordinates and ``UR7Outputs``
returns absolute 10D EEF actions for comparison with the dataset.

Example::

    PYTHONPATH=src .venv/bin/python test/eval_ur7_openloop.py \
      --checkpoint-dir checkpoints/pi05_bund_demo_data_0reset_0827_v002_v21_reconstructed_lora_local4/ur7_pi05_lora_local4/80000 \
      --episode-index 0,50,100,200 --compare-horizon 50 \
      --output-dir /16T-2/sudu/code/.result/openloop/pi05_bund_demo_data_0reset_0827_v002_v21_reconstructed_lora_local4

``--output-dir`` is the report root. The script appends the exact checkpoint
folder name and writes ``episode_ID_STEP_HORIZON.png`` there. For example, a
checkpoint named ``10000`` produces ``10000/episode_000_1w_25.png``. The
horizon defaults to the full 50-step action chunk; use
``--compare-horizon 25`` to re-query every 25 frames and compare the first 25
predicted steps at each query.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import numpy as np


DEFAULT_DATA_PATH = pathlib.Path(
    "//16T-2/sudu/code/.data/bund_20260827_n216_v21_new"
)
DEFAULT_CONFIG = "pi05_bund_demo_data_0reset_0827_v002_v21_reconstructed_lora_local4"
REPORT_ROOT = pathlib.Path("/16T-2/sudu/code/.result/openloop")
DEFAULT_ACTION_NAMES = (
    "eef.x",
    "eef.y",
    "eef.z",
    "eef.rot6d.r00",
    "eef.rot6d.r10",
    "eef.rot6d.r20",
    "eef.rot6d.r01",
    "eef.rot6d.r11",
    "eef.rot6d.r21",
    "gripper.open",
)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_task(data_path: pathlib.Path, task_index: int) -> str:
    task_path = data_path / "meta" / "tasks.jsonl"
    if task_path.exists():
        for line in task_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("task_index", row.get("index", -1))) == task_index:
                return str(row.get("task", row.get("prompt", "do the task")))
    return "do the task"


def _episode_path(data_path: pathlib.Path, info: dict[str, Any], episode_index: int) -> pathlib.Path:
    chunks_size = int(info.get("chunks_size", 1000))
    pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    path = data_path / pattern.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {path}")
    return path


def _video_path(data_path: pathlib.Path, info: dict[str, Any], episode_index: int, key: str) -> pathlib.Path:
    chunks_size = int(info.get("chunks_size", 1000))
    pattern = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    path = data_path / pattern.format(
        episode_chunk=episode_index // chunks_size,
        video_key=key,
        episode_index=episode_index,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Video not found for {key}: {path}")
    return path


def _require_checkpoint_dir(path: pathlib.Path) -> pathlib.Path:
    """Require an exact committed checkpoint step directory.

    The caller must provide the directory that directly contains ``params/``
    and ``assets/`` (for example ``.../ur7_pi05_lora_local4/1000``).  In
    particular, do not silently choose a step from an experiment directory:
    evaluating an unintended checkpoint can make the report misleading.
    """
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    if not (path / "params").is_dir():
        raise FileNotFoundError(
            f"Not a committed checkpoint step directory: {path}\n"
            "Pass the exact step directory containing params/ and assets/, "
            "for example .../ur7_pi05_lora_local4/1000."
        )
    if not (path / "assets").is_dir():
        raise FileNotFoundError(
            f"Checkpoint is missing assets/: {path}\n"
            "Pass a complete committed OpenPI checkpoint step directory."
        )
    return path


def _checkpoint_label(path: pathlib.Path) -> str:
    """Return a compact checkpoint label for report names."""
    name = path.name
    if name.isdigit():
        steps = int(name)
        if steps and steps % 10_000 == 0:
            return f"{steps // 10_000}w"
    return name


class _VideoReader:
    """Small OpenCV reader that keeps a video handle across policy queries."""

    def __init__(self, path: pathlib.Path):
        import cv2

        self._cv2 = cv2
        self._path = path
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"Could not open video: {path}")

    def read(self, frame_index: int) -> np.ndarray:
        frame_index = int(frame_index)
        if frame_index < 0:
            raise ValueError(f"frame index must be non-negative, got {frame_index}")
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_index} from {self._path}")
        return np.ascontiguousarray(self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "_VideoReader":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def _read_video_frame(path: pathlib.Path, frame_index: int) -> np.ndarray:
    """Read one frame for lightweight validation/tests without a persistent reader."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(frame)


def _load_episode(data_path: pathlib.Path, info: dict[str, Any], episode_index: int):
    import pandas as pd

    parquet_path = _episode_path(data_path, info, episode_index)
    df = pd.read_parquet(parquet_path)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    if state.ndim != 2 or state.shape[1] != 34:
        raise ValueError(f"Expected observation.state shape (N,34), got {state.shape}")
    if actions.ndim != 2 or actions.shape[1] != 10:
        raise ValueError(f"Expected action shape (N,10), got {actions.shape}")
    if "task_index" in df:
        task_index = int(np.asarray(df["task_index"].iloc[0]).reshape(-1)[0])
    else:
        task_index = 0
    return (
        state,
        actions,
        task_index,
        _video_path(data_path, info, episode_index, "observation.images.base"),
        _video_path(data_path, info, episode_index, "observation.images.wrist"),
    )


def _validate_dataset(data_path: pathlib.Path, episode_index: int) -> dict[str, Any]:
    info = _read_json(data_path / "meta" / "info.json")
    total_episodes = int(info["total_episodes"])
    if episode_index < 0 or episode_index >= total_episodes:
        raise ValueError(f"episode-index={episode_index} outside [0, {total_episodes})")
    state, actions, task_index, base_video, wrist_video = _load_episode(data_path, info, episode_index)
    norm_path = data_path / "assets" / "norm_stats.json"
    if not norm_path.is_file():
        raise FileNotFoundError(f"Missing normalization stats: {norm_path}")
    return {
        "info": info,
        "frames": int(state.shape[0]),
        "state_shape": list(state.shape),
        "action_shape": list(actions.shape),
        "task_index": task_index,
        "task": _load_task(data_path, task_index),
        "base_video": str(base_video),
        "wrist_video": str(wrist_video),
    }


def _parse_episode_indices(value: str) -> list[int]:
    """Parse a comma-separated list such as ``0,50,100,200``."""
    indices: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            index = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid episode index: {item!r}") from exc
        if index < 0:
            raise argparse.ArgumentTypeError(f"episode index must be non-negative: {index}")
        if index not in indices:
            indices.append(index)
    if not indices:
        raise argparse.ArgumentTypeError("episode-index must contain at least one integer")
    return indices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Exact completed checkpoint step directory containing params/ and assets/.",
    )
    parser.add_argument(
        "--episode-index",
        type=_parse_episode_indices,
        default=[0],
        help="Dataset episode id or comma-separated ids, e.g. 0,50,100,200.",
    )
    parser.add_argument("--compare-horizon", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Report root; checkpoint folder and episode_step_horizon.png are appended automatically.",
    )
    return parser.parse_args()


def _save_report(
    output_dir: pathlib.Path,
    *,
    report_name: str,
    pred: np.ndarray,
    gt: np.ndarray,
    frame_index: np.ndarray,
    query_frame: np.ndarray,
    relative_step: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Older runs wrote a second PNG. Remove only that generated legacy plot so
    # a reused report directory contains the single combined image requested.
    legacy_mae_plot = output_dir / "mae_by_chunk_step.png"
    if legacy_mae_plot.exists():
        legacy_mae_plot.unlink()
    error = pred - gt
    abs_error = np.abs(error)
    per_dim_mae = abs_error.mean(axis=0)
    overall_mae = float(abs_error.mean())
    overall_rmse = float(np.sqrt(np.square(error).mean()))

    def plot_contiguous(ax, x: np.ndarray, y: np.ndarray, *, color: str) -> None:
        """Plot consecutive dataset frames as one line across chunk boundaries."""
        order = np.argsort(x, kind="stable")
        x = np.asarray(x)[order]
        y = np.asarray(y)[order]
        if x.size == 0:
            return
        # Keep any genuinely untested interval blank instead of drawing a
        # misleading line. The evaluator uses compare_horizon as its query
        # stride, so normal runs cover every dataset frame continuously.
        cuts = np.flatnonzero(np.diff(x) > 1) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [x.size]))
        for start, end in zip(starts, ends):
            ax.plot(x[start:end], y[start:end], color=color, linewidth=1.0, alpha=0.85)

    import matplotlib.pyplot as plt

    # Keep the ten action plots in the upper block and put the chunk-relative
    # MAE curve below them, matching the combined report layout used by the
    # reference evaluator.  There is deliberately only one output image.
    fig = plt.figure(figsize=(18, 22))
    grid = fig.add_gridspec(
        nrows=6,
        ncols=2,
        height_ratios=[1.0] * 5 + [1.15],
        hspace=0.55,
        wspace=0.24,
    )
    for dim in range(pred.shape[1]):
        ax = fig.add_subplot(grid[dim // 2, dim % 2])
        plot_contiguous(ax, frame_index, gt[:, dim], color="tab:green")
        plot_contiguous(ax, frame_index, pred[:, dim], color="tab:red")
        ax.set_title(f"{dim}: {DEFAULT_ACTION_NAMES[dim]} | MAE={per_dim_mae[dim]:.5f}")
        ax.grid(alpha=0.25)
        ax.set_xlabel("dataset frame")

    horizon = int(metadata["compare_horizon"])
    mae_by_step = []
    for step in range(horizon):
        values = abs_error[relative_step == step]
        if values.size:
            mae_by_step.append(float(values.mean()))
    mae_ax = fig.add_subplot(grid[5, :])
    if mae_by_step:
        mae_ax.plot(np.arange(len(mae_by_step)), mae_by_step, color="tab:blue", linewidth=1.6)
    mae_ax.set_title("Open-loop MAE vs prediction horizon")
    mae_ax.set_xlabel("relative step inside predicted chunk")
    mae_ax.set_ylabel("mean absolute error")
    mae_ax.grid(alpha=0.3)

    report_label = metadata["report_label"]
    fig.suptitle(
        f"UR7 Pi05 {report_label} | episode={metadata['episode_index']} | "
        f"MAE={overall_mae:.5f} RMSE={overall_rmse:.5f}\n"
        "green=ground truth, red=prediction",
        y=0.998,
    )
    fig.subplots_adjust(top=0.965, bottom=0.04, left=0.055, right=0.985)
    report_path = output_dir / report_name
    fig.savefig(report_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved report directory: {output_dir}")
    print(f"Saved report image: {report_path}")
    print(f"Overall MAE: {overall_mae:.6f}")
    print(f"Overall RMSE: {overall_rmse:.6f}")


def _evaluate_episode(
    *,
    policy: Any,
    data_path: pathlib.Path,
    validation: dict[str, Any],
    episode_index: int,
    checkpoint: pathlib.Path,
    compare_horizon: int,
    output_root: pathlib.Path,
) -> None:
    state, gt_actions, task_index, base_video, wrist_video = _load_episode(
        data_path, validation["info"], episode_index
    )
    prompt = _load_task(data_path, task_index)
    # Re-query exactly when the previous predicted chunk ends. For horizon=25
    # this gives starts 0, 25, 50, ... and consumes all dataset frames once.
    query_stride = compare_horizon
    starts = list(range(0, len(state), query_stride))
    if not starts:
        raise RuntimeError("No query frames selected")

    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    frame_chunks: list[np.ndarray] = []
    query_chunks: list[np.ndarray] = []
    step_chunks: list[np.ndarray] = []
    with _VideoReader(base_video) as base_reader, _VideoReader(wrist_video) as wrist_reader:
        for query_i, start in enumerate(starts, start=1):
            t0 = time.perf_counter()
            base_image = base_reader.read(start)
            wrist_image = wrist_reader.read(start)
            obs = {
                "state": state[start],
                "image": base_image,
                "wrist_image": wrist_image,
                "task": prompt,
            }
            out = policy.infer(obs)
            pred_raw = np.asarray(out["actions"], dtype=np.float32)
            if pred_raw.ndim == 3:
                pred_raw = pred_raw[0]
            if pred_raw.ndim != 2 or pred_raw.shape[1] != 10:
                raise ValueError(f"UR7 policy must return (H,10), got {pred_raw.shape}")
            horizon = min(compare_horizon, pred_raw.shape[0], len(gt_actions) - start)
            if horizon <= 0:
                break
            pred_chunks.append(pred_raw[:horizon])
            gt_chunks.append(gt_actions[start : start + horizon])
            frame_chunks.append(np.arange(start, start + horizon, dtype=np.int64))
            query_chunks.append(np.full(horizon, start, dtype=np.int64))
            step_chunks.append(np.arange(horizon, dtype=np.int64))
            mae = float(np.abs(pred_raw[:horizon] - gt_actions[start : start + horizon]).mean())
            print(
                f"query {query_i}/{len(starts)} frame={start} horizon={horizon} "
                f"mae={mae:.6f} infer={1000*(time.perf_counter()-t0):.1f} ms",
                flush=True,
            )

    if not pred_chunks:
        raise RuntimeError("No predictions collected")
    pred = np.concatenate(pred_chunks)
    gt = np.concatenate(gt_chunks)
    metadata = {
        "config": DEFAULT_CONFIG,
        "checkpoint": str(checkpoint),
        "data_path": str(data_path),
        "episode_index": episode_index,
        "task_index": task_index,
        "prompt": prompt,
        "query_stride": query_stride,
        "compare_horizon": compare_horizon,
        "report_label": (
            f"openloop_{episode_index:03d}_{_checkpoint_label(checkpoint)}_{compare_horizon}"
        ),
    }
    output_dir = output_root / checkpoint.name
    report_name = f"episode_{episode_index:03d}_{_checkpoint_label(checkpoint)}_{compare_horizon}.png"
    _save_report(
        output_dir,
        report_name=report_name,
        pred=pred,
        gt=gt,
        frame_index=np.concatenate(frame_chunks),
        query_frame=np.concatenate(query_chunks),
        relative_step=np.concatenate(step_chunks),
        metadata=metadata,
    )


def main() -> None:
    args = _parse_args()
    data_path = DEFAULT_DATA_PATH.expanduser().resolve()
    if args.compare_horizon <= 0 or args.compare_horizon > 50:
        raise ValueError("compare-horizon must be between 1 and 50")
    checkpoint = _require_checkpoint_dir(pathlib.Path(args.checkpoint_dir))
    validations = {
        episode_index: _validate_dataset(data_path, episode_index)
        for episode_index in args.episode_index
    }
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "episode_indices": args.episode_index,
                "episodes": [validations[index] for index in args.episode_index],
            },
            indent=2,
        )
    )

    import openpi.policies.policy_config as policy_config
    from openpi.training import config as training_config

    train_config = training_config.get_config(DEFAULT_CONFIG)
    # Load the model once, then evaluate each requested dataset episode.
    policy = policy_config.create_trained_policy(train_config, checkpoint)
    output_root = args.output_dir or REPORT_ROOT / DEFAULT_CONFIG
    for episode_index in args.episode_index:
        _evaluate_episode(
            policy=policy,
            data_path=data_path,
            validation=validations[episode_index],
            episode_index=episode_index,
            checkpoint=checkpoint,
            compare_horizon=args.compare_horizon,
            output_root=output_root,
        )


if __name__ == "__main__":
    main()
