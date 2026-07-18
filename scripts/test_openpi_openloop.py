#!/usr/bin/env python3
"""
Fast open-loop evaluation for an OpenPI policy server on a local LeRobot v2.x dataset.

Robot/camera profiles:
    --robot auto    # default; detects AgileX vs H01 from meta/info.json
    --robot agilex  # 3 cameras: cam_high, cam_left_wrist, cam_right_wrist
    --robot h01     # 5 cameras: cam_fisheye_front, wrist up/down
    --robot h01_3v  # 3 cameras: fisheye_front, left/right wrist up
    --robot ebench  # 3 cameras: video.overlook/left/right and 19-dim action
    --robot robocasa # 3 cameras: robot0_agentview_left/eye_in_hand/agentview_right and 12-dim action
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import os
import pathlib
import shutil
import signal
import subprocess
import time
import textwrap
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


GT_COLOR = "tab:green"
PRED_COLOR = "tab:red"
MAE_COLOR = "tab:blue"
MAE_FAIL_COLOR = "red"
OPENLOOP_MAE_FAIL_THRESHOLD = 0.1
ABS_ERROR_COLORS = tuple(f"C{i}" for i in range(10))
CHUNK_START_COLOR = "#808080"
CHUNK_START_ALPHA = 0.6
CHUNK_START_LINE_WIDTH = 1.0
CHUNK_START_LINE_POINTS = 8

ROBOCASA_STATE_RAW_TO_MODEL = [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15]
ROBOCASA_ACTION_RAW_TO_MODEL = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]
ROBOCASA_RAW_ACTION_NAMES = [
    "base_motion_x",
    "base_motion_y",
    "base_motion_yaw",
    "base_motion_torso",
    "control_mode",
    "end_effector_position_x",
    "end_effector_position_y",
    "end_effector_position_z",
    "end_effector_rotation_x",
    "end_effector_rotation_y",
    "end_effector_rotation_z",
    "gripper_close",
]
ROBOCASA_MODEL_ACTION_NAMES = [
    "end_effector_position_x",
    "end_effector_position_y",
    "end_effector_position_z",
    "end_effector_rotation_x",
    "end_effector_rotation_y",
    "end_effector_rotation_z",
    "gripper_close",
    "base_motion_x",
    "base_motion_y",
    "base_motion_yaw",
    "base_motion_torso",
    "control_mode",
]

ROBOCASA_ATOMIC_TASKS = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]
ROBOCASA_COMPOSITE_TASKS = [
    "DeliverStraw",
    "GetToastedBread",
    "KettleBoiling",
    "LoadDishwasher",
    "PackIdenticalLunches",
    "PreSoakPan",
    "PrepareCoffee",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SetUpCuttingStation",
    "StackBowlsInCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "WashLettuce",
]
ROBOCASA_TASK_DIR_ALIASES = {
    "StackBowlsInCabinet": "StackBowlsCabinet",
}

EBENCH_LONG_HORIZON_TASKS = [
    "bottle",
    "detergent",
    "dish",
    "dishwasher",
    "fruit",
    "make_sandwich",
    "microwave",
    "pen",
    "shop",
]
EBENCH_SIMPLE_PNP_TASKS = [
    "task1",
    "task2",
    "task3",
    "task4",
    "task5",
    "task6",
    "task7",
    "task8",
    "task9",
    "task10",
]
EBENCH_TELEOP_TASKS = [
    "collect_coffee_beans",
    "flip_cup_collect_cookies",
    "frame_against_pen_holder",
    "install_gear",
    "peg_in_hole",
    "put_glass_in_glassbox",
    "tighten_nut",
]


def _checkpoint_label_from_path(path: str | os.PathLike[str] | None) -> str:
    if not path:
        return ""
    checkpoint_path = pathlib.Path(path)
    parts = checkpoint_path.parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return checkpoint_path.name


def _checkpoint_step_from_path(path: str | os.PathLike[str] | None) -> str:
    if not path:
        return "checkpoint_unknown"
    checkpoint_path = pathlib.Path(path)
    label = _checkpoint_label_from_path(checkpoint_path)
    if "_step_" in label:
        return "step_" + label.rsplit("_step_", 1)[1]
    name = checkpoint_path.name
    if name.isdigit():
        return f"step_{name}"
    return name


def _safe_filename_part(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text)).strip("_")


def _find_robocasa_root(data_path: str | os.PathLike[str]) -> pathlib.Path:
    path = pathlib.Path(data_path).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "atomic").is_dir() and (candidate / "composite").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find RoboCasa root containing atomic/ and composite/ from: {data_path}")


def _latest_robocasa_lerobot_dir(task_dir: pathlib.Path) -> pathlib.Path | None:
    candidates = []
    direct = task_dir / "lerobot"
    if (direct / "meta" / "info.json").exists():
        candidates.append(direct)
    for child in sorted(task_dir.iterdir()) if task_dir.exists() else []:
        lerobot_dir = child / "lerobot"
        if (lerobot_dir / "meta" / "info.json").exists():
            candidates.append(lerobot_dir)
    return candidates[-1] if candidates else None


def _robocasa_task_specs(task_set: str) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    if task_set in {"atomic18", "official34"}:
        specs.extend(("atomic", task_name) for task_name in ROBOCASA_ATOMIC_TASKS)
    if task_set in {"composite16", "official34"}:
        specs.extend(("composite", task_name) for task_name in ROBOCASA_COMPOSITE_TASKS)
    return specs


def _resolve_robocasa_task_paths(data_path: str | os.PathLike[str], task_set: str) -> list[tuple[str, str, pathlib.Path]]:
    root = _find_robocasa_root(data_path)
    jobs: list[tuple[str, str, pathlib.Path]] = []
    missing = []
    for split_name, task_name in _robocasa_task_specs(task_set):
        task_dir_name = ROBOCASA_TASK_DIR_ALIASES.get(task_name, task_name)
        lerobot_dir = _latest_robocasa_lerobot_dir(root / split_name / task_dir_name)
        if lerobot_dir is None:
            missing.append(f"{split_name}/{task_name}")
            continue
        jobs.append((split_name, task_name, lerobot_dir))

    if missing:
        missing_text = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing RoboCasa task dataset(s):\n  {missing_text}")
    return jobs


def _find_ebench_root(data_path: str | os.PathLike[str]) -> pathlib.Path:
    path = pathlib.Path(data_path).resolve()
    for candidate in [path, *path.parents]:
        if (
            (candidate / "long_horizon").is_dir()
            and (candidate / "simple_pnp").is_dir()
            and (candidate / "teleop_tasks").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not find EBench root containing long_horizon/, simple_pnp/, and teleop_tasks/ from: {data_path}"
    )


def _ebench_task_specs(task_set: str) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    if task_set in {"long_horizon9", "all26"}:
        specs.extend(("long_horizon", task_name) for task_name in EBENCH_LONG_HORIZON_TASKS)
    if task_set in {"simple_pnp10", "all26"}:
        specs.extend(("simple_pnp", task_name) for task_name in EBENCH_SIMPLE_PNP_TASKS)
    if task_set in {"teleop7", "all26"}:
        specs.extend(("teleop_tasks", task_name) for task_name in EBENCH_TELEOP_TASKS)
    return specs


def _resolve_ebench_task_paths(data_path: str | os.PathLike[str], task_set: str) -> list[tuple[str, str, pathlib.Path]]:
    root = _find_ebench_root(data_path)
    jobs: list[tuple[str, str, pathlib.Path]] = []
    missing = []
    for split_name, task_name in _ebench_task_specs(task_set):
        task_dir = root / split_name / task_name
        if (task_dir / "meta" / "info.json").exists():
            jobs.append((split_name, task_name, task_dir))
        else:
            missing.append(f"{split_name}/{task_name}")

    if missing:
        missing_text = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing EBench task dataset(s):\n  {missing_text}")
    return jobs


def _ensure_policy_server_library_path() -> None:
    candidates: list[pathlib.Path] = []
    if os.environ.get("CONDA_PREFIX"):
        candidates.append(pathlib.Path(os.environ["CONDA_PREFIX"]) / "lib")
    candidates.append(pathlib.Path("/opt/conda/lib"))

    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    changed = False
    for candidate in reversed(candidates):
        candidate_str = str(candidate)
        if candidate.is_dir() and candidate_str not in parts:
            parts.insert(0, candidate_str)
            changed = True

    if changed:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)


# -----------------------------
# Optional in-process policy server
# -----------------------------


def _serve_policy_entry(
    *,
    policy_config: str,
    checkpoint_dir: str,
    port: int,
    default_prompt: str | None,
    record: bool,
) -> None:
    """Child-process entrypoint for serving an OpenPI checkpoint."""
    import logging
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    preferred_paths = (
        repo_root / "src",
        repo_root / "packages" / "openpi-client" / "src",
        repo_root,
    )
    for path in reversed(preferred_paths):
        path_str = str(path)
        if path.exists():
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass
            sys.path.insert(0, path_str)

    # The robocasa2 environment may have Robocasa/openpi installed editable.
    # Spawned workers inherit that import state, so purge it before importing
    # this repository's serve_policy.
    for module_name in list(sys.modules):
        if module_name == "openpi" or module_name.startswith("openpi."):
            del sys.modules[module_name]

    from scripts import serve_policy
    import openpi

    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s,%(msecs)03d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    openpi_file = pathlib.Path(getattr(openpi, "__file__", "")).resolve()
    expected_root = (repo_root / "src" / "openpi").resolve()
    logging.info("Using openpi package from %s", openpi_file)
    if expected_root not in openpi_file.parents:
        raise RuntimeError(
            f"Wrong openpi package imported: {openpi_file}. "
            f"Expected it under {expected_root}. "
            "Check PYTHONPATH / editable installs."
        )
    serve_policy.main(
        serve_policy.Args(
            default_prompt=default_prompt,
            port=port,
            record=record,
            policy=serve_policy.Checkpoint(config=policy_config, dir=checkpoint_dir),
        )
    )


def _wait_for_policy_server(
    *,
    host: str,
    port: int,
    api_key: str | None,
    timeout_s: float,
    process: Any | None = None,
) -> None:
    from openpi_client import websocket_client_policy

    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if isinstance(process, multiprocessing.Process):
            if not process.is_alive():
                raise RuntimeError(f"Policy server process exited early with code {process.exitcode}") from last_error
        elif isinstance(process, subprocess.Popen):
            returncode = process.poll()
            if returncode is not None:
                raise RuntimeError(f"Policy server process exited early with code {returncode}") from last_error

        try:
            client = websocket_client_policy.WebsocketClientPolicy(
                host=host,
                port=port,
                api_key=api_key,
            )
            metadata = client.get_server_metadata()
            print(f"Policy server is ready on {host}:{port}. Metadata: {metadata}", flush=True)
            return
        except Exception as exc:
            last_error = exc
            print(f"Waiting for policy server on {host}:{port} ... ({type(exc).__name__}: {exc})", flush=True)
            time.sleep(2.0)

    raise TimeoutError(f"Timed out after {timeout_s:.1f}s waiting for policy server on {host}:{port}") from last_error


def _start_policy_server_python(args: argparse.Namespace) -> multiprocessing.Process:
    if not args.policy_config:
        raise ValueError("--server-mode python requires --policy-config")
    if not args.checkpoint_dir:
        raise ValueError("--server-mode python requires --checkpoint-dir / --policy-dir")

    print("[001]server:"f"config={args.policy_config!r}, checkpoint={args.checkpoint_dir!r}, port={args.port}",flush=True,)
    _ensure_policy_server_library_path()
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(
        target=_serve_policy_entry,
        kwargs=dict(
            policy_config=args.policy_config,
            checkpoint_dir=args.checkpoint_dir,
            port=args.port,
            default_prompt=args.server_default_prompt,
            record=args.server_record,
        ),
        name="openpi_policy_server",
    )
    if args.keep_server:
        # Using a child process means the server cannot outlive the parent cleanly.
        # Use --server-mode script to keep a server running after this script exits.
        process.daemon = True
        print(
            "Warning: --keep-server with --server-mode python will stop the server when this script exits. "
            "Use --server-mode script to keep it running.",
            flush=True,
        )
    process.start()

    _wait_for_policy_server(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        timeout_s=args.server_startup_timeout,
        process=process,
    )
    return process


def _start_policy_server_script(args: argparse.Namespace) -> subprocess.Popen:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    script_path = pathlib.Path(args.server_script)
    if not script_path.is_absolute():
        script_path = repo_root / script_path
    if not script_path.exists():
        raise FileNotFoundError(f"Policy server script not found: {script_path}")

    print(f"Starting OpenPI policy server via {script_path} on port {args.port}", flush=True)
    process = subprocess.Popen(
        ["bash", str(script_path)],
        cwd=str(repo_root),
        env=os.environ.copy(),
        start_new_session=True,
    )
    _wait_for_policy_server(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        timeout_s=args.server_startup_timeout,
        process=process,
    )
    return process


def _start_policy_server_if_requested(args: argparse.Namespace) -> Any | None:
    if not args.start_server:
        return None
    if args.server_mode == "python":
        return _start_policy_server_python(args)
    if args.server_mode == "script":
        return _start_policy_server_script(args)
    raise ValueError(f"Unsupported --server-mode: {args.server_mode}")


def _stop_policy_server(process: Any | None, *, keep_server: bool) -> None:
    if process is None:
        return
    if keep_server:
        print(f"Keeping policy server alive with pid={process.pid}", flush=True)
        return
    if isinstance(process, multiprocessing.Process):
        if process.is_alive():
            print(f"Stopping policy server pid={process.pid}", flush=True)
            process.terminate()
            process.join(timeout=20)
        if process.is_alive():
            print(f"Policy server pid={process.pid} did not terminate cleanly; killing.", flush=True)
            process.kill()
            process.join(timeout=10)
        try:
            process.close()
        except Exception:
            pass
        return
    if isinstance(process, subprocess.Popen) and process.poll() is None:
        print(f"Stopping policy server pid={process.pid}", flush=True)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            print(f"Policy server pid={process.pid} did not terminate cleanly; killing.", flush=True)
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def _cleanup_multiprocessing_children(timeout_s: float = 10.0) -> None:
    children = multiprocessing.active_children()
    if not children:
        return
    print(f"Cleaning up {len(children)} multiprocessing child process(es)", flush=True)
    for child in children:
        if child.is_alive():
            child.terminate()
    for child in children:
        child.join(timeout=timeout_s)
    for child in children:
        if child.is_alive():
            print(f"Child pid={child.pid} did not terminate cleanly; killing.", flush=True)
            child.kill()
    for child in children:
        child.join(timeout=timeout_s)
        try:
            child.close()
        except Exception:
            pass


# -----------------------------
# Metadata / LeRobot helpers
# -----------------------------


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_info(data_path: pathlib.Path) -> dict[str, Any]:
    info_path = data_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Cannot find {info_path}")
    return _read_json(info_path)


def _episode_chunk(info: dict[str, Any], episode_index: int) -> int:
    chunks_size = int(info.get("chunks_size", 1000))
    return episode_index // chunks_size


def _episode_parquet_path(data_path: pathlib.Path, info: dict[str, Any], episode_index: int) -> pathlib.Path:
    chunk = _episode_chunk(info, episode_index)
    pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    rel = pattern.format(episode_chunk=chunk, episode_index=episode_index)
    path = data_path / rel
    if not path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {path}")
    return path


def _episode_video_path(data_path: pathlib.Path, info: dict[str, Any], episode_index: int, video_key: str) -> pathlib.Path:
    chunk = _episode_chunk(info, episode_index)
    pattern = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    rel = pattern.format(
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=video_key,
    )
    path = data_path / rel
    if path.exists():
        return path

    # Defensive fallback for datasets with slightly different folder naming.
    candidates = []
    video_root = data_path / "videos" / f"chunk-{chunk:03d}"
    if video_root.exists():
        candidates.extend(video_root.glob(f"*/episode_{episode_index:06d}.mp4"))
        candidates.extend(video_root.glob(f"**/episode_{episode_index:06d}.mp4"))
    suffix = video_key.split(".")[-1]
    for cand in candidates:
        parent = cand.parent.name
        if parent == video_key or parent.endswith(video_key) or parent.endswith(suffix):
            return cand

    raise FileNotFoundError(
        f"Video for key {video_key!r} not found. Tried {path}. "
        f"Searched under {video_root if video_root.exists() else '<missing video root>'}."
    )


def _read_task_map(data_path: pathlib.Path) -> dict[int, str]:
    task_map: dict[int, str] = {}
    for path in [data_path / "meta" / "tasks.jsonl", data_path / "meta" / "tasks.json"]:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    idx = row.get("task_index", row.get("index", row.get("id")))
                    text = row.get("task", row.get("prompt", row.get("name")))
                    if idx is not None and text is not None:
                        task_map[int(idx)] = str(text)
        else:
            raw = _read_json(path)
            rows: Iterable[Any]
            if isinstance(raw, dict):
                # Some versions store {'0': 'task text'}, others store {'tasks': [...]}.
                if "tasks" in raw and isinstance(raw["tasks"], list):
                    rows = raw["tasks"]
                else:
                    for k, v in raw.items():
                        if isinstance(v, str):
                            task_map[int(k)] = v
                    rows = raw.values()
            else:
                rows = raw
            for row in rows:
                if isinstance(row, str):
                    continue
                if not isinstance(row, dict):
                    continue
                idx = row.get("task_index", row.get("index", row.get("id")))
                text = row.get("task", row.get("prompt", row.get("name")))
                if idx is not None and text is not None:
                    task_map[int(idx)] = str(text)
    return task_map


def _scalar_int_from_cell(x: Any) -> int:
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot convert empty value to int")
    return int(arr[0])


def _stack_vector_column(df: pd.DataFrame, key: str, dtype: np.dtype | type = np.float32) -> np.ndarray:
    if key not in df.columns:
        raise KeyError(f"Column {key!r} not found. Available columns: {list(df.columns)}")
    values = df[key].to_numpy()
    try:
        arr = np.stack(values).astype(dtype)
    except Exception:
        arr = np.asarray([np.asarray(v, dtype=dtype) for v in values], dtype=dtype)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _stack_vector_columns(df: pd.DataFrame, keys: Iterable[str], dtype: np.dtype | type = np.float32) -> np.ndarray:
    arrays = [_stack_vector_column(df, key, dtype) for key in keys]
    if not arrays:
        raise ValueError("No columns provided")
    return np.concatenate(arrays, axis=-1)


def _action_names(info: dict[str, Any], action_dim: int) -> list[str]:
    names = info.get("features", {}).get("action", {}).get("names") or []
    names = [str(x) for x in names[:action_dim]]
    while len(names) < action_dim:
        names.append(f"action_{len(names)}")
    return names


# -----------------------------
# Robot/camera profile helpers
# -----------------------------


def _image_feature_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {}) or {}
    keys = []
    for key, value in features.items():
        key = str(key)
        if not (key.startswith("observation.images.") or key.startswith("video.")):
            continue
        if isinstance(value, dict) and value.get("dtype") not in {None, "video", "image"}:
            continue
        keys.append(key)
    return sorted(keys)


def _scan_video_keys(data_path: pathlib.Path, info: dict[str, Any], episode_index: int) -> list[str]:
    chunk = _episode_chunk(info, episode_index)
    root = data_path / "videos" / f"chunk-{chunk:03d}"
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _profile_image_map(profile: str) -> dict[str, str]:
    """Return output-image-name -> LeRobot dataset video key.

    The output names must match the names used in the OpenPI training config
    repack_transform. The dataset keys must match meta/info.json and the video
    folders on disk.
    """
    if profile == "agilex":
        return {
            "cam_high": "observation.images.cam_high",
            "cam_left_wrist": "observation.images.cam_left_wrist",
            "cam_right_wrist": "observation.images.cam_right_wrist",
        }
    if profile == "h01":
        return {
            "cam_high": "observation.images.cam_fisheye_front",
            "cam_left_wrist_up": "observation.images.cam_left_wrist_up",
            "cam_right_wrist_up": "observation.images.cam_right_wrist_up",
            "cam_left_wrist_down": "observation.images.cam_left_wrist_down",
            "cam_right_wrist_down": "observation.images.cam_right_wrist_down",
        }
    if profile == "h01_3v":
        return {
            "cam_high": "observation.images.cam_fisheye_front",
            "cam_left_wrist": "observation.images.cam_left_wrist_up",
            "cam_right_wrist": "observation.images.cam_right_wrist_up",
        }
    if profile == "ebench":
        return {
            "images/head": "video.overlook_camera_view",
            "images/hand_left": "video.left_camera_view",
            "images/hand_right": "video.right_camera_view",
        }
    if profile == "robocasa":
        return {
            "observation/image": "observation.images.robot0_agentview_left",
            "observation/wrist_image": "observation.images.robot0_eye_in_hand",
            "observation/right_image": "observation.images.robot0_agentview_right",
        }
    raise ValueError(f"Unknown robot profile: {profile!r}")


def _default_action_dim_for_profile(profile: str) -> int:
    if profile in {"h01", "h01_3v"}:
        return 16
    if profile == "ebench":
        return 19
    if profile == "robocasa":
        return 12
    return 14


def _take_last_dim(arr: np.ndarray, indices: list[int], *, name: str) -> np.ndarray:
    if arr.shape[1] <= max(indices):
        raise ValueError(
            f"{name} dim {arr.shape[1]} is too small for indices {indices}. "
            "This looks like a RoboCasa layout mismatch."
        )
    return arr[:, indices]


def _resolve_robocasa_layout(args: argparse.Namespace) -> str:
    layout = args.robocasa_layout
    if layout != "auto":
        return layout
    if args.policy_config == "pi05_robocasa_task300":
        return "raw"
    return "eef_first"


def _parse_image_key_map(text: str) -> dict[str, str]:
    """Parse 'out=dataset,out=dataset' into a dict.

    Example:
      cam_high=observation.images.cam_fisheye_front,cam_left_wrist_up=observation.images.cam_left_wrist_up
    """
    result: dict[str, str] = {}
    for raw_item in str(text).replace(";", ",").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "--image-key-map items must be output_name=dataset_key, "
                f"got {item!r}"
            )
        out_name, dataset_key = item.split("=", 1)
        out_name = out_name.strip()
        dataset_key = dataset_key.strip()
        if not out_name or not dataset_key:
            raise ValueError(f"Invalid --image-key-map item: {item!r}")
        result[out_name] = dataset_key
    if not result:
        raise ValueError("--image-key-map did not contain any valid mappings")
    return result


def _infer_robot_profile(info: dict[str, Any], data_path: pathlib.Path, episode_index: int) -> str:
    image_keys = set(_image_feature_keys(info))
    if not image_keys:
        image_keys = set(_scan_video_keys(data_path, info, episode_index))

    if "observation.images.cam_fisheye_front" in image_keys:
        return "h01"
    if {
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    }.issubset(image_keys):
        return "agilex"
    if {
        "video.overlook_camera_view",
        "video.left_camera_view",
        "video.right_camera_view",
    }.issubset(image_keys):
        return "ebench"
    if {
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_agentview_right",
    }.issubset(image_keys):
        return "robocasa"

    # Fallback heuristics for slightly different metadata.
    joined = " ".join(sorted(image_keys)).lower()
    if "fisheye" in joined or "wrist_up" in joined or "wrist_down" in joined:
        return "h01"
    if "cam_high" in joined and "cam_left_wrist" in joined and "cam_right_wrist" in joined:
        return "agilex"
    if "overlook_camera_view" in joined and "left_camera_view" in joined and "right_camera_view" in joined:
        return "ebench"
    if "robot0_agentview" in joined and "robot0_eye_in_hand" in joined:
        return "robocasa"

    available = ", ".join(sorted(image_keys)) or "<none>"
    raise ValueError(
        "Could not infer robot/camera profile from dataset metadata. "
        f"Available image/video keys: {available}. "
        "Pass --robot agilex, --robot h01, --robot h01_3v, --robot ebench, "
        "--robot robocasa, or --image-key-map output=dataset,..."
    )


def _resolve_image_key_map(
    *,
    args: argparse.Namespace,
    info: dict[str, Any],
    data_path: pathlib.Path,
    episode_index: int,
) -> tuple[str, dict[str, str]]:
    if args.image_key_map:
        return "custom", _parse_image_key_map(args.image_key_map)

    profile = args.robot
    if profile == "auto":
        profile = _infer_robot_profile(info, data_path, episode_index)
    elif profile == "manual3":
        return "manual3", {
            "cam_high": args.cam_high_key,
            "cam_left_wrist": args.cam_left_wrist_key,
            "cam_right_wrist": args.cam_right_wrist_key,
        }

    return profile, _profile_image_map(profile)



def _parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text).replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Empty integer list")
    return values



def _parse_optional_int_tokens(tokens: list[str] | None) -> list[int] | str | None:
    if tokens is None:
        return None
    values: list[int] = []
    saw_auto = False
    for token in tokens:
        for part in str(token).replace(",", " ").split():
            part = part.strip()
            if not part:
                continue
            if part.lower() == "auto":
                saw_auto = True
                continue
            values.append(int(part))
    if saw_auto:
        if values:
            raise ValueError("--dataset-index auto cannot be combined with explicit task indices")
        return "auto"
    if not values:
        return None
    return values


def _auto_task_indices(data_path: pathlib.Path) -> list[int]:
    task_map = _read_task_map(data_path)
    if task_map:
        return sorted(int(idx) for idx in task_map)

    info = _load_info(data_path)
    total_tasks = int(info.get("total_tasks", 0) or 0)
    if total_tasks > 0:
        return list(range(total_tasks))

    raise ValueError(
        "--dataset-index auto requires task metadata. "
        f"Could not read tasks from {data_path / 'meta' / 'tasks.jsonl'} or total_tasks from meta/info.json."
    )

def _task_index_from_name(task_map: dict[int, str], task_name: str) -> int:
    needle = task_name.strip().lower()
    if not needle:
        raise ValueError("--task-name cannot be empty")
    matches = [(idx, text) for idx, text in task_map.items() if needle in text.lower()]
    if not matches:
        available = "; ".join(f"{idx}: {text}" for idx, text in sorted(task_map.items()))
        raise ValueError(f"No task prompt contains {task_name!r}. Available tasks: {available}")
    if len(matches) > 1:
        print("Warning: --task-name matched multiple tasks; using the first one:")
        for idx, text in matches:
            print(f"  task_index={idx}: {text}")
    return int(sorted(matches, key=lambda x: x[0])[0][0])


def _episode_task_index(parquet_path: pathlib.Path) -> int | None:
    try:
        df = pd.read_parquet(parquet_path, columns=["task_index"])
    except Exception:
        df = pd.read_parquet(parquet_path)
    if "task_index" not in df.columns or len(df) == 0:
        return None
    return _scalar_int_from_cell(df["task_index"].iloc[0])


def _find_episode_for_task(
    *,
    data_path: pathlib.Path,
    info: dict[str, Any],
    task_index: int,
    episode_rank_in_task: int,
) -> int:
    if episode_rank_in_task < 0:
        raise ValueError("--episode-rank-in-task must be non-negative")
    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes <= 0:
        # Fallback: infer episode indices from parquet files.
        candidates = sorted((data_path / "data").glob("chunk-*/episode_*.parquet"))
        episode_indices = [int(p.stem.split("_")[-1]) for p in candidates]
    else:
        episode_indices = list(range(total_episodes))

    seen = 0
    for ep in episode_indices:
        try:
            parquet_path = _episode_parquet_path(data_path, info, ep)
        except FileNotFoundError:
            continue
        ep_task = _episode_task_index(parquet_path)
        if ep_task == task_index:
            if seen == episode_rank_in_task:
                return ep
            seen += 1
    raise ValueError(
        f"Could not find episode_rank_in_task={episode_rank_in_task} for task_index={task_index}. "
        f"Found {seen} matching episodes."
    )


def _resolve_episode_index(
    *,
    data_path: pathlib.Path,
    info: dict[str, Any],
    task_map: dict[int, str],
    explicit_episode_index: int | None,
    task_index: int | None,
    task_name: str | None,
    episode_rank_in_task: int,
) -> tuple[int, int | None, str | None]:
    """Resolve which episode to evaluate.

    If --episode-index is provided, it wins. Otherwise the script selects the first
    episode for the requested task/dataset. If no task is specified, it defaults to
    the smallest task index found in tasks metadata, usually 0.
    """
    if task_name is not None:
        task_index = _task_index_from_name(task_map, task_name)

    if explicit_episode_index is not None:
        ep = int(explicit_episode_index)
        parquet_path = _episode_parquet_path(data_path, info, ep)
        ep_task = _episode_task_index(parquet_path)
        return ep, ep_task, task_map.get(ep_task) if ep_task is not None else None

    if task_index is None:
        if task_map:
            task_index = int(sorted(task_map.keys())[0])
        else:
            task_index = 0

    ep = _find_episode_for_task(
        data_path=data_path,
        info=info,
        task_index=int(task_index),
        episode_rank_in_task=episode_rank_in_task,
    )
    return ep, int(task_index), task_map.get(int(task_index))


# -----------------------------
# Optional local episode cache
# -----------------------------


def _copy_if_needed(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return dst


def _maybe_cache_episode(
    *,
    local_cache_dir: str | None,
    episode_index: int,
    parquet_path: pathlib.Path,
    video_paths: dict[str, pathlib.Path],
) -> tuple[pathlib.Path, dict[str, pathlib.Path]]:
    if not local_cache_dir:
        return parquet_path, video_paths

    cache_root = pathlib.Path(local_cache_dir) / f"episode_{episode_index:06d}"
    t0 = time.time()
    cached_parquet = _copy_if_needed(parquet_path, cache_root / parquet_path.name)
    cached_videos: dict[str, pathlib.Path] = {}
    for name, src in video_paths.items():
        safe = name.replace("/", "_").replace(".", "_")
        cached_videos[name] = _copy_if_needed(src, cache_root / f"{safe}_{src.name}")
    print(f"Cached episode files to {cache_root} in {time.time() - t0:.1f}s")
    return cached_parquet, cached_videos


# -----------------------------
# Fast video frame readers
# -----------------------------


class SequentialVideoReader:
    """Read increasing frame indices from a video without LeRobotDataset random access.

    Backends:
      - auto: use ffmpeg for AV1 videos, otherwise OpenCV.
      - opencv: fastest for common H.264/H.265 videos, but often fails on AV1 builds.
      - ffmpeg: robust fallback; decodes requested frames through system ffmpeg.
      - imageio: fallback if imageio.v3 has a working video plugin in your environment.
    """

    def __init__(self, path: pathlib.Path, backend: str = "auto") -> None:
        self.path = pathlib.Path(path)
        self.backend = self._resolve_backend(backend)
        self._cur = -1
        self._cache: dict[int, np.ndarray] = {}
        self._cap = None
        self._iter = None
        self.cv2 = None
        self.iio = None

        if self.backend == "opencv":
            try:
                import cv2  # type: ignore
            except Exception as exc:
                raise ImportError("OpenCV backend requested but cv2 is not available") from exc
            self.cv2 = cv2
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open video with OpenCV: {self.path}")
        elif self.backend == "imageio":
            try:
                import imageio.v3 as iio  # type: ignore
            except Exception as exc:
                raise ImportError("imageio backend requested but imageio.v3 is not available") from exc
            self.iio = iio
            self._iter = iter(iio.imiter(self.path))
        elif self.backend == "ffmpeg":
            if shutil.which("ffmpeg") is None:
                raise RuntimeError("ffmpeg backend requested, but the `ffmpeg` command was not found in PATH")
        else:
            raise ValueError(f"Unsupported video backend after resolution: {self.backend}")

        print(f"Video backend for {self.path.name}: {self.backend}")

    def _resolve_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend
        codec = _probe_video_codec(self.path)
        if codec == "av1":
            # Many OpenCV wheels are built without reliable software AV1 decode support.
            return "ffmpeg"
        return "opencv"

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _reopen(self) -> None:
        self.close()
        self._cur = -1
        self._iter = None
        if self.backend == "opencv":
            assert self.cv2 is not None
            self._cap = self.cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to reopen video with OpenCV: {self.path}")
        elif self.backend == "imageio":
            assert self.iio is not None
            self._iter = iter(self.iio.imiter(self.path))

    def get(self, frame_idx: int) -> np.ndarray:
        if frame_idx in self._cache:
            return self._cache[frame_idx]
        if frame_idx < self._cur and self.backend in {"opencv", "imageio"}:
            # Rare case: support re-reading an earlier frame by reopening.
            self._reopen()

        try:
            if self.backend == "opencv":
                frame = self._get_opencv(frame_idx)
            elif self.backend == "imageio":
                frame = self._get_imageio(frame_idx)
            elif self.backend == "ffmpeg":
                frame = self._get_ffmpeg(frame_idx)
            else:
                raise ValueError(f"Unsupported video backend: {self.backend}")
        except Exception as exc:
            if self.backend == "opencv":
                # Common path for AV1 videos in OpenCV builds. Fall back to ffmpeg once.
                print(f"Warning: OpenCV failed to decode {self.path} frame {frame_idx}: {exc}")
                print("Falling back to ffmpeg backend for this video.")
                self.close()
                self.backend = "ffmpeg"
                frame = self._get_ffmpeg(frame_idx)
            else:
                raise

        self._cache[frame_idx] = frame
        return frame

    def _get_opencv(self, frame_idx: int) -> np.ndarray:
        assert self._cap is not None
        assert self.cv2 is not None
        # If the gap is very large, seeking is faster. For query_stride=50, sequential grabs are usually fast.
        gap = frame_idx - self._cur
        if gap > 300:
            self._cap.set(self.cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = self._cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"Could not read frame {frame_idx} from {self.path}")
            self._cur = frame_idx
        else:
            bgr = None
            while self._cur < frame_idx:
                ok, bgr = self._cap.read()
                if not ok or bgr is None:
                    raise RuntimeError(f"Could not read frame {frame_idx} from {self.path}; stopped at {self._cur}")
                self._cur += 1
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        return np.asarray(rgb, dtype=np.uint8)

    def _get_imageio(self, frame_idx: int) -> np.ndarray:
        assert self._iter is not None
        frame = None
        while self._cur < frame_idx:
            try:
                frame = next(self._iter)
            except StopIteration as exc:
                raise RuntimeError(f"Could not read frame {frame_idx} from {self.path}; stopped at {self._cur}") from exc
            self._cur += 1
        if frame is None:
            raise RuntimeError(f"Could not read frame {frame_idx} from {self.path}")
        return np.asarray(frame, dtype=np.uint8)

    def _get_ffmpeg(self, frame_idx: int) -> np.ndarray:
        """Extract one exact frame as RGB using system ffmpeg.

        This is slower than OpenCV sequential decode, but it is much more robust for AV1 videos
        in containers where OpenCV lacks AV1 software decoding support.
        """
        from PIL import Image
        import io

        # The comma in eq(n,FRAME) must be escaped for ffmpeg's filter parser.
        vf = f"select=eq(n\\,{int(frame_idx)})"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-hwaccel",
            "none",
            "-i",
            str(self.path),
            "-vf",
            vf,
            "-vsync",
            "0",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not proc.stdout:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ffmpeg failed to decode frame {frame_idx} from {self.path}\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{stderr}"
            )
        img = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)


def _probe_video_codec(path: pathlib.Path) -> str | None:
    """Return codec name via ffprobe when available, e.g. 'av1' or 'h264'."""
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    codec = proc.stdout.decode("utf-8", errors="replace").strip().splitlines()
    return codec[0].strip().lower() if codec else None


# -----------------------------
# Image / observation helpers
# -----------------------------


def _hwc_to_chw_uint8(image_hwc: np.ndarray) -> np.ndarray:
    img = np.asarray(image_hwc)
    if img.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {img.shape}")
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got shape {img.shape}")
    if np.issubdtype(img.dtype, np.floating):
        if float(np.nanmax(img)) <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.moveaxis(img, -1, 0)


def _resize_chw_with_pad_uint8(image_chw: np.ndarray, size: int | None) -> np.ndarray:
    if size is None or image_chw.shape[1:] == (size, size):
        return image_chw
    from PIL import Image

    hwc = np.moveaxis(image_chw, 0, -1)
    h, w = hwc.shape[:2]
    scale = min(size / h, size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    pil = Image.fromarray(hwc)
    pil = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = np.asarray(pil, dtype=np.uint8)
    return np.moveaxis(canvas, -1, 0)


def _build_observation(
    *,
    states: np.ndarray,
    raw_state_columns: dict[str, np.ndarray] | None,
    frame_idx: int,
    prompt: str,
    readers: dict[str, SequentialVideoReader],
    image_size: int | None,
    robot_profile: str,
) -> tuple[dict[str, Any], float]:
    t0 = time.time()
    images: dict[str, np.ndarray] = {}
    for out_name, reader in readers.items():
        img_hwc = reader.get(frame_idx)
        img_chw = _hwc_to_chw_uint8(img_hwc)
        img_chw = _resize_chw_with_pad_uint8(img_chw, image_size)
        images[out_name] = img_chw
    read_ms = 1000.0 * (time.time() - t0)
    if robot_profile == "ebench":
        if raw_state_columns is None:
            raise ValueError("EBench profile requires raw_state_columns")
        obs = {
            "states/joint": np.asarray(raw_state_columns["state.joints"][frame_idx], dtype=np.float32),
            "states/gripper": np.asarray(raw_state_columns["state.gripper"][frame_idx], dtype=np.float32),
            "images/head": images["images/head"],
            "images/hand_left": images["images/hand_left"],
            "images/hand_right": images["images/hand_right"],
            "prompt": prompt,
        }
        if "state.base" in raw_state_columns:
            obs["state.base"] = np.asarray(raw_state_columns["state.base"][frame_idx], dtype=np.float32)
    elif robot_profile == "robocasa":
        obs = {
            "observation/state": np.asarray(states[frame_idx], dtype=np.float32),
            "prompt": prompt,
        }
        obs.update(images)
    else:
        state = np.asarray(states[frame_idx], dtype=np.float32)
        if robot_profile == "h01_3v":
            state = state[:16]
        obs = {
            "state": state,
            "images": images,
            "prompt": prompt,
        }
    return obs, read_ms


@dataclass
class EvalResult:
    pred: np.ndarray
    gt: np.ndarray
    frame_index: np.ndarray
    query_frame: np.ndarray
    relative_step: np.ndarray
    action_names: list[str]
    prompt: str | None = None
    task_index: int | None = None
    episode_index: int | None = None
    checkpoint_label: str = ""


# -----------------------------
# Evaluation
# -----------------------------


def run_eval(args: argparse.Namespace) -> EvalResult:
    from openpi_client import websocket_client_policy

    data_path = pathlib.Path(args.data_path)
    info = _load_info(data_path)
    task_map = _read_task_map(data_path)

    selected_episode_index, selected_task_index, selected_task_prompt = _resolve_episode_index(
        data_path=data_path,
        info=info,
        task_map=task_map,
        explicit_episode_index=args.episode_index,
        task_index=args.task_index,
        task_name=args.task_name,
        episode_rank_in_task=args.episode_rank_in_task,
    )
    args.episode_index = selected_episode_index
    print(
        f"Selected episode_index={selected_episode_index} "
        f"for task_index={selected_task_index}, task_prompt={selected_task_prompt!r}"
    )

    parquet_path = _episode_parquet_path(data_path, info, selected_episode_index)
    robot_profile, video_keys = _resolve_image_key_map(
        args=args,
        info=info,
        data_path=data_path,
        episode_index=selected_episode_index,
    )
    if args.action_dim is None:
        args.action_dim = _default_action_dim_for_profile(robot_profile)
    robocasa_layout = _resolve_robocasa_layout(args) if robot_profile == "robocasa" else None
    print(f"Robot/camera profile: {robot_profile}")
    print(f"Action dim for comparison: {args.action_dim}")
    if robocasa_layout is not None:
        print(f"RoboCasa layout: {robocasa_layout}")
    print("Image key map, policy input name -> dataset video key:")
    for out_name, dataset_key in video_keys.items():
        print(f"  {out_name}: {dataset_key}")

    try:
        video_paths = {
            out_name: _episode_video_path(data_path, info, selected_episode_index, dataset_key)
            for out_name, dataset_key in video_keys.items()
        }
    except FileNotFoundError as exc:
        available_features = _image_feature_keys(info)
        available_video_dirs = _scan_video_keys(data_path, info, selected_episode_index)
        details = [str(exc)]
        details.append("Available image keys from meta/info.json: " + (", ".join(available_features) if available_features else "<none>"))
        details.append("Available video folders for this episode chunk: " + (", ".join(available_video_dirs) if available_video_dirs else "<none>"))
        details.append(
            "For H01, try: --robot h01 or --robot h01_3v. "
            "For AgileX, try: --robot agilex. For EBench, try: --robot ebench. "
            "For RoboCasa, try: --robot robocasa. "
            "For custom data, use --image-key-map out=dataset,..."
        )
        raise FileNotFoundError("\n".join(details)) from exc

    parquet_path, video_paths = _maybe_cache_episode(
        local_cache_dir=args.local_cache_dir,
        episode_index=selected_episode_index,
        parquet_path=parquet_path,
        video_paths=video_paths,
    )

    print(f"Reading episode parquet: {parquet_path}")
    t0 = time.time()
    df = pd.read_parquet(parquet_path)
    print(f"Loaded parquet rows={len(df)} in {time.time() - t0:.2f}s")

    raw_state_columns: dict[str, np.ndarray] | None = None
    if robot_profile == "ebench":
        ebench_action_base_key = "action.base" if "action.base" in df.columns else "action.base_delta"
        raw_state_columns = {
            "state.joints": _stack_vector_column(df, "state.joints", np.float32),
            "state.gripper": _stack_vector_column(df, "state.gripper", np.float32),
        }
        state_keys = ["state.joints", "state.gripper"]
        if "state.base" in df.columns:
            raw_state_columns["state.base"] = _stack_vector_column(df, "state.base", np.float32)
            state_keys.append("state.base")
        states = _stack_vector_columns(df, state_keys, np.float32)
        actions_all = _stack_vector_columns(df, ["action.joints", "action.gripper", ebench_action_base_key], np.float32)
    else:
        states = _stack_vector_column(df, args.state_key, np.float32)
        actions_all = _stack_vector_column(df, args.action_key, np.float32)
        if robot_profile == "robocasa" and robocasa_layout == "eef_first":
            states = _take_last_dim(states, ROBOCASA_STATE_RAW_TO_MODEL, name="RoboCasa state")
            actions_all = _take_last_dim(actions_all, ROBOCASA_ACTION_RAW_TO_MODEL, name="RoboCasa action")
            print(
                "RoboCasa layout adapter: converted parquet base-first state/action "
                "to training/eval eef-first order."
            )
        elif robot_profile == "robocasa":
            print("RoboCasa layout adapter: keeping parquet base-first state/action order.")
    if states.shape[0] != actions_all.shape[0]:
        raise ValueError(f"State rows {states.shape[0]} != action rows {actions_all.shape[0]}")
    if actions_all.shape[1] < args.action_dim:
        raise ValueError(f"Dataset action dim {actions_all.shape[1]} < --action-dim {args.action_dim}")

    task_index: int | None = None
    if "task_index" in df.columns:
        task_index = _scalar_int_from_cell(df["task_index"].iloc[0])
    prompt = args.prompt or (task_map.get(task_index) if task_index is not None else None) or selected_task_prompt or "do the task"
    print(f"Episode {args.episode_index}, frames={len(df)}, state_dim={states.shape[1]}, action_dim={actions_all.shape[1]}")
    print(f"task_index={task_index}, prompt={prompt!r}")
    print("Video paths:")
    for name, path in video_paths.items():
        print(f"  {name}: {path}")

    readers = {
        name: SequentialVideoReader(path, backend=args.video_backend)
        for name, path in video_paths.items()
    }

    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    try:
        print("Server metadata:", client.get_server_metadata())
    except Exception as exc:
        print(f"Warning: failed to get server metadata: {exc}")

    episode_len = len(df)
    if args.start_frame < 0 or args.start_frame >= episode_len:
        raise ValueError(f"--start-frame {args.start_frame} outside episode length {episode_len}")
    query_starts = list(range(args.start_frame, episode_len, args.query_stride))
    if args.max_queries is not None:
        query_starts = query_starts[: args.max_queries]
    if not query_starts:
        raise RuntimeError("No query frames selected")

    # Warm up the server. This also warms video decoding for the first frame.
    if args.warmup_queries > 0:
        warm_frame = query_starts[0]
        obs0, read_ms = _build_observation(
            states=states,
            raw_state_columns=raw_state_columns,
            frame_idx=warm_frame,
            prompt=prompt,
            readers=readers,
            image_size=args.image_size,
            robot_profile=robot_profile,
        )
        for i in range(args.warmup_queries):
            wt0 = time.time()
            _ = client.infer(obs0)
            print(f"warmup {i + 1}/{args.warmup_queries}: frame={warm_frame}, video_read={read_ms:.1f} ms, infer={1000*(time.time()-wt0):.1f} ms")

    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    frame_chunks: list[np.ndarray] = []
    query_frame_chunks: list[np.ndarray] = []
    rel_step_chunks: list[np.ndarray] = []

    for query_i, start in enumerate(query_starts):
        obs, read_ms = _build_observation(
            states=states,
            raw_state_columns=raw_state_columns,
            frame_idx=start,
            prompt=prompt,
            readers=readers,
            image_size=args.image_size,
            robot_profile=robot_profile,
        )

        t_infer = time.time()
        out = client.infer(obs)
        infer_ms = 1000.0 * (time.time() - t_infer)
        if "actions" not in out:
            raise KeyError(f"Policy response does not contain 'actions'. Keys: {list(out.keys())}")
        pred_raw = np.asarray(out["actions"], dtype=np.float32)
        if pred_raw.ndim == 1:
            pred_raw = pred_raw[None, :]

        horizon_cap = args.compare_horizon or pred_raw.shape[0]
        horizon = min(pred_raw.shape[0], horizon_cap, episode_len - start)
        if args.skip_last_incomplete and horizon < min(pred_raw.shape[0], horizon_cap):
            break

        dim = min(args.action_dim, pred_raw.shape[1], actions_all.shape[1])
        pred = pred_raw[:horizon, :dim]
        gt = actions_all[start : start + horizon, :dim]

        pred_chunks.append(pred)
        gt_chunks.append(gt)
        frame_chunks.append(np.arange(start, start + horizon, dtype=np.int64))
        query_frame_chunks.append(np.full((horizon,), start, dtype=np.int64))
        rel_step_chunks.append(np.arange(horizon, dtype=np.int64))

        mae = float(np.mean(np.abs(pred - gt)))
        print(
            f"query {query_i + 1:04d}/{len(query_starts):04d}: "
            f"frame={start}, pred_shape={tuple(pred_raw.shape)}, compare=({horizon},{dim}), "
            f"mae={mae:.6f}, video_read={read_ms:.1f} ms, infer={infer_ms:.1f} ms"
        )

    for reader in readers.values():
        reader.close()

    if not pred_chunks:
        raise RuntimeError("No predictions were collected. Check start_frame/query_stride/episode length.")

    pred_all = np.concatenate(pred_chunks, axis=0)
    gt_all = np.concatenate(gt_chunks, axis=0)
    frame_all = np.concatenate(frame_chunks, axis=0)
    query_frame_all = np.concatenate(query_frame_chunks, axis=0)
    rel_step_all = np.concatenate(rel_step_chunks, axis=0)
    if robot_profile == "robocasa":
        source_names = ROBOCASA_RAW_ACTION_NAMES if robocasa_layout == "raw" else ROBOCASA_MODEL_ACTION_NAMES
        names = list(source_names[: pred_all.shape[1]])
        while len(names) < pred_all.shape[1]:
            names.append(f"action_{len(names)}")
    else:
        names = _action_names(info, pred_all.shape[1])

    return EvalResult(
        pred=pred_all,
        gt=gt_all,
        frame_index=frame_all,
        query_frame=query_frame_all,
        relative_step=rel_step_all,
        action_names=names,
        prompt=prompt,
        task_index=task_index,
        episode_index=args.episode_index,
        checkpoint_label=_checkpoint_label_from_path(
            args.checkpoint_dir or os.environ.get("checkpoint") or os.environ.get("CHECKPOINT")
        ),
    )


# -----------------------------
# Output / plotting
# -----------------------------


def save_outputs(result: EvalResult, args: argparse.Namespace) -> None:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    err, abs_err, overall_mae, overall_rmse, mae_per_dim, rmse_per_dim = _error_statistics(result)

    rows = []
    for t in range(result.pred.shape[0]):
        for d, name in enumerate(result.action_names):
            rows.append(
                {
                    "frame_index": int(result.frame_index[t]),
                    "query_frame": int(result.query_frame[t]),
                    "relative_step": int(result.relative_step[t]),
                    "dim": int(d),
                    "action_name": name,
                    "gt": float(result.gt[t, d]),
                    "pred": float(result.pred[t, d]),
                    "error": float(result.pred[t, d] - result.gt[t, d]),
                    "abs_error": float(abs(result.pred[t, d] - result.gt[t, d])),
                }
            )
    output_prefix = getattr(args, "output_prefix", None)
    file_prefix = f"{_safe_filename_part(output_prefix)}_" if output_prefix else ""

    csv_path = output_dir / f"{file_prefix}per_step_actions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    saved_plot_paths: list[pathlib.Path] = []
    for horizon in args.plot_horizons:
        plot_result = _filter_result_by_relative_horizon(result, horizon)
        if plot_result.pred.shape[0] == 0:
            print(f"Warning: no samples for plot horizon {horizon}; skipping")
            continue
        summary_path = output_dir / f"{file_prefix}openloop_summary_h{horizon}.png"
        _plot_combined_summary(plot_result, summary_path, title_suffix=f"first {horizon} steps of each chunk")
        saved_plot_paths.append(summary_path)

    print("\nSaved outputs:")
    print(f"  {csv_path}")
    for path in saved_plot_paths:
        print(f"  {path}")
    print("\nNo openloop_arrays.npz or metrics.json is written in this version.")
    print("Separate action_curves/abs_error/mae images are not written; they are combined into openloop_summary_h*.png.")
    print(f"Overall MAE:  {overall_mae:.6f}")
    print(f"Overall RMSE: {overall_rmse:.6f}")
    print("Per-dimension MAE/RMSE:")
    for i, name in enumerate(result.action_names):
        print(f"  {i:02d} {name}: mae={mae_per_dim[i]:.6f}, rmse={rmse_per_dim[i]:.6f}")


def _result_metric_row(
    *,
    split_name: str,
    task_name: str,
    data_path: pathlib.Path,
    output_prefix: str,
    result: EvalResult,
) -> dict[str, Any]:
    _, _, overall_mae, overall_rmse, _, _ = _error_statistics(result)
    return {
        "split": split_name,
        "task": task_name,
        "data_path": str(data_path),
        "output_prefix": output_prefix,
        "episode_index": result.episode_index,
        "task_index": result.task_index,
        "prompt": result.prompt,
        "checkpoint": result.checkpoint_label,
        "mae": overall_mae,
        "rmse": overall_rmse,
    }


def _save_robocasa_task_set_summary(rows: list[dict[str, Any]], output_dir: pathlib.Path, task_set: str) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"robocasa_{task_set}_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    mae_values = [float(row["mae"]) for row in rows]
    rmse_values = [float(row["rmse"]) for row in rows]
    print("\n" + "=" * 100, flush=True)
    print(f"RoboCasa {task_set} summary: {len(rows)} task(s)", flush=True)
    print(f"Mean task MAE:  {np.mean(mae_values):.6f}", flush=True)
    print(f"Mean task RMSE: {np.mean(rmse_values):.6f}", flush=True)
    print(f"Saved task-set summary: {path}", flush=True)
    print("=" * 100, flush=True)


def _save_ebench_task_set_summary(rows: list[dict[str, Any]], output_dir: pathlib.Path, task_set: str) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"ebench_{task_set}_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    mae_values = [float(row["mae"]) for row in rows]
    rmse_values = [float(row["rmse"]) for row in rows]
    print("\n" + "=" * 100, flush=True)
    print(f"EBench {task_set} summary: {len(rows)} task(s)", flush=True)
    print(f"Mean task MAE:  {np.mean(mae_values):.6f}", flush=True)
    print(f"Mean task RMSE: {np.mean(rmse_values):.6f}", flush=True)
    print(f"Saved task-set summary: {path}", flush=True)
    print("=" * 100, flush=True)


def _filter_result_by_relative_horizon(result: EvalResult, horizon: int) -> EvalResult:
    mask = result.relative_step < int(horizon)
    return EvalResult(
        pred=result.pred[mask],
        gt=result.gt[mask],
        frame_index=result.frame_index[mask],
        query_frame=result.query_frame[mask],
        relative_step=result.relative_step[mask],
        action_names=result.action_names,
        prompt=result.prompt,
        task_index=result.task_index,
        episode_index=result.episode_index,
        checkpoint_label=result.checkpoint_label,
    )


def _iter_query_masks(result: EvalResult) -> Iterable[np.ndarray]:
    # Split lines by query chunk so prediction curves do not connect across independent server calls.
    for q in np.unique(result.query_frame):
        idx = np.flatnonzero(result.query_frame == q)
        if idx.size == 0:
            continue
        order = np.argsort(result.frame_index[idx])
        yield idx[order]


def _mae_vs_relative_step_values(result: EvalResult) -> tuple[list[int], list[float]]:
    _, abs_err, _, _, _, _ = _error_statistics(result)
    xs: list[int] = []
    ys: list[float] = []
    for step in sorted(np.unique(result.relative_step)):
        mask = result.relative_step == step
        if not np.any(mask):
            continue
        xs.append(int(step))
        ys.append(float(np.mean(abs_err[mask])))
    return xs, ys


def _error_statistics(result: EvalResult) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    err = result.pred - result.gt
    abs_err = np.abs(err)
    overall_mae = float(np.mean(abs_err))
    overall_rmse = float(np.sqrt(np.mean(err ** 2)))
    mae_per_dim = np.mean(abs_err, axis=0)
    rmse_per_dim = np.sqrt(np.mean(err ** 2, axis=0))
    return err, abs_err, overall_mae, overall_rmse, mae_per_dim, rmse_per_dim


def _mark_chunk_start(ax: Any, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) == 0 or len(y) == 0:
        return
    ax.annotate("", xy=(x[0], y[0]), xytext=(CHUNK_START_LINE_POINTS, 0),
        xycoords="data", textcoords="offset points",
        arrowprops=dict( arrowstyle="-",color=CHUNK_START_COLOR, linestyle=(0, (1, 3)), linewidth=CHUNK_START_LINE_WIDTH, alpha=CHUNK_START_ALPHA,),
        zorder=10,
    )


def _plot_combined_summary(result: EvalResult, path: pathlib.Path, title_suffix: str = "") -> None:
    """Save one combined figure.

    Top block:
      - Left half: GT vs policy action curves for every action dimension.
      - Right half: absolute error curves for every action dimension.
    Bottom block:
      - MAE as a function of relative step inside the predicted chunk.
    """
    import matplotlib.pyplot as plt

    dim = result.pred.shape[1]
    top_rows = math.ceil(dim / 2)
    fig_height = max(3.0 * top_rows + 4.0, 8.0)
    fig = plt.figure(figsize=(28, fig_height))
    gs = fig.add_gridspec(
        nrows=top_rows + 1,
        ncols=4,
        height_ratios=[1.0] * top_rows + [1.25],
        hspace=0.55,
        wspace=0.28,
    )

    chunk_indices = list(_iter_query_masks(result))
    _, abs_err, overall_mae, overall_rmse, _, _ = _error_statistics(result)

    # Left side: action curves. Right side: absolute error curves.
    for d in range(dim):
        row = d // 2
        col = d % 2

        ax_action = fig.add_subplot(gs[row, col])
        first_gt = True
        first_pred = True
        for idx in chunk_indices:
            x = result.frame_index[idx]
            gt_y = result.gt[idx, d]
            pred_y = result.pred[idx, d]
            ax_action.plot(
                x,
                gt_y,
                label="dataset action" if first_gt else None,
                linewidth=1.2,
                color=GT_COLOR,
            )
            ax_action.plot(
                x,
                pred_y,
                label="policy prediction" if first_pred else None,
                linewidth=1.0,
                alpha=0.85,
                color=PRED_COLOR,
            )
            _mark_chunk_start(ax_action, x, gt_y)
            first_gt = False
            first_pred = False
        ax_action.set_title(f"action {d}: {result.action_names[d]}")
        ax_action.set_xlabel("dataset frame")
        ax_action.set_ylabel("action value")
        ax_action.grid(True, alpha=0.3)
        if d == 0:
            ax_action.legend(loc="best")

        ax_error = fig.add_subplot(gs[row, 2 + col])
        for chunk_i, idx in enumerate(chunk_indices):
            ax_error.plot(
                result.frame_index[idx],
                abs_err[idx, d],
                linewidth=1.0,
                color=ABS_ERROR_COLORS[chunk_i % len(ABS_ERROR_COLORS)],
            )
        ax_error.set_title(f"abs error {d}: {result.action_names[d]}")
        ax_error.set_xlabel("dataset frame")
        ax_error.set_ylabel("|pred - gt|")
        ax_error.grid(True, alpha=0.3)

    # Hide unused top cells if action dim is odd.
    if dim % 2 == 1:
        fig.add_subplot(gs[top_rows - 1, 1]).axis("off")
        fig.add_subplot(gs[top_rows - 1, 3]).axis("off")

    ax_mae = fig.add_subplot(gs[top_rows, :])
    xs, ys = _mae_vs_relative_step_values(result)
    mae_failed = any(y > OPENLOOP_MAE_FAIL_THRESHOLD for y in ys)
    mae_color = MAE_FAIL_COLOR if mae_failed else MAE_COLOR
    ax_mae.plot(xs, ys, linewidth=1.6, color=mae_color)
    ax_mae.set_xlabel("relative step inside predicted chunk")
    ax_mae.set_ylabel("mean absolute error")
    ax_mae.set_title("Open-loop MAE vs prediction horizon")
    ax_mae.grid(True, alpha=0.3)
    if mae_failed:
        ax_mae.text(
            0.01,
            0.98,
            "F",
            transform=ax_mae.transAxes,
            ha="left",
            va="top",
            fontsize=42,
            fontweight="bold",
            color=MAE_FAIL_COLOR,
        )

    title = "OpenPI open-loop evaluation summary"
    if result.checkpoint_label:
        title += f" | checkpoint={result.checkpoint_label}"
    title += f" | MAE={overall_mae:.6f} | RMSE={overall_rmse:.6f}"
    if title_suffix:
        title += f" ({title_suffix})"

    prompt = (result.prompt or "").strip()
    meta_parts = []
    if result.task_index is not None:
        meta_parts.append(f"task_index={result.task_index}")
    if result.episode_index is not None:
        meta_parts.append(f"episode_index={result.episode_index}")

    # Keep the prompt directly under the figure title.  Long prompts are wrapped so they
    # remain readable in the saved PNG.
    subtitle = ""
    if prompt:
        subtitle = f"Prompt: {prompt}"
    if meta_parts:
        meta = " | ".join(meta_parts)
        subtitle = f"{subtitle}    ({meta})" if subtitle else meta
    if subtitle:
        subtitle = "\n".join(textwrap.wrap(subtitle, width=160))

    fig.suptitle(title, fontsize=16, y=0.992)
    if subtitle:
        fig.text(0.5, 0.977, subtitle, ha="center", va="top", fontsize=11)
        fig.subplots_adjust(top=0.955)
    else:
        fig.subplots_adjust(top=0.975)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="Local LeRobot dataset root, containing meta/data/videos.")
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Force a specific LeRobot episode index. If omitted, choose the first episode from the selected task/dataset.",
    )
    parser.add_argument(
        "--task-index",
        "--dataset-index",
        dest="task_indices",
        nargs="+",
        default=None,
        help=(
            "Select task/dataset index or indices in a merged LeRobot dataset. "
            "Examples: --dataset-index 1, --dataset-index 1,2,3,4, or --dataset-index auto. "
            "Default: first task, usually 0."
        ),
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Select a task/dataset by substring match on the task prompt.",
    )
    parser.add_argument(
        "--episode-rank-in-task",
        type=int,
        default=0,
        help="When selecting by task/dataset, choose the N-th episode within that task. Default: 0, the first one.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8999)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output-dir", default="openloop_eval_out_fast")

    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start a policy server before running open-loop evaluation.",
    )
    parser.add_argument(
        "--server-mode",
        choices=["auto", "script", "python"],
        default="auto",
        help=(
            "How to start the server with --start-server. "
            "auto uses --server-script unless --policy-config/--checkpoint-dir is provided."
        ),
    )
    parser.add_argument(
        "--server-script",
        default="run_policy.sh",
        help="Shell script used when --server-mode script. Relative paths are resolved from the 1-Pi05 repo root.",
    )
    parser.add_argument(
        "--policy-config",
        default=None,
        help="Training config name passed to serve_policy Checkpoint(config=...). Used with --server-mode python.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--policy-dir",
        dest="checkpoint_dir",
        default=None,
        help="Checkpoint directory passed to serve_policy Checkpoint(dir=...). Used with --server-mode python.",
    )
    parser.add_argument(
        "--server-startup-timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for an internally started policy server to become ready.",
    )
    parser.add_argument(
        "--server-default-prompt",
        default=None,
        help="Default prompt passed to serve_policy when starting the server internally.",
    )
    parser.add_argument(
        "--server-record",
        action="store_true",
        help="Enable serve_policy recording when starting the server internally.",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Do not terminate the internally started server after evaluation.",
    )

    parser.add_argument("--query-stride", type=int, default=50, help="Query the policy every N dataset frames.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--compare-horizon", type=int, default=50, help="Max actions from each predicted chunk to compare.")
    parser.add_argument(
        "--plot-horizons",
        default="50",
        help="Comma/space separated horizons to plot from each predicted chunk. Default saves only the 50-step plot.",
    )
    parser.add_argument("--skip-last-incomplete", action="store_true")

    parser.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help=(
            "Compare only the first N action dimensions. "
            "Default depends on --robot: agilex/manual3=14, h01/h01_3v=16, ebench=19."
        ),
    )
    parser.add_argument("--image-size", type=int, default=224, help="Resize/pad images to this square size before sending. Use 0 to disable resizing.")
    parser.add_argument("--prompt", default=None, help="Override prompt. If omitted, read from LeRobot task_index metadata.")
    parser.add_argument("--warmup-queries", type=int, default=1)

    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument(
        "--robot",
        choices=["auto", "agilex", "h01", "h01_3v", "ebench", "robocasa", "manual3"],
        default="auto",
        help=(
            "Camera/profile preset. auto detects from meta/info.json. "
            "agilex uses 3 cameras: cam_high/left_wrist/right_wrist. "
            "h01 uses 5 cameras: fisheye_front plus wrist up/down. "
            "h01_3v uses fisheye_front plus left/right wrist up and sends the first 16 state dimensions. "
            "ebench uses video.overlook/left/right camera view and EBench state/action columns. "
            "robocasa maps robot0_agentview_left/eye_in_hand/agentview_right to official observation/* inputs. "
            "manual3 keeps the legacy --cam-*-key options."
        ),
    )
    parser.add_argument(
        "--robocasa-layout",
        choices=["auto", "raw", "eef_first"],
        default="auto",
        help=(
            "RoboCasa state/action layout. auto uses raw/base-first for pi05_robocasa_task300 "
            "and eef_first for official RoboCasa configs."
        ),
    )
    parser.add_argument(
        "--robocasa-task-set",
        choices=["none", "official34", "atomic18", "composite16"],
        default="none",
        help="Evaluate fixed RoboCasa task sets. Use --data-path as the RoboCasa-Dataset root or any path under it.",
    )
    parser.add_argument(
        "--ebench-task-set",
        choices=["none", "all26", "long_horizon9", "simple_pnp10", "teleop7"],
        default="none",
        help="Evaluate fixed EBench task sets. Use --data-path as the EBench-Dataset root or any path under it.",
    )
    parser.add_argument(
        "--image-key-map",
        default=None,
        help=(
            "Custom policy_input_name=dataset_video_key mapping, comma separated. "
            "Example: cam_high=observation.images.cam_fisheye_front,"
            "cam_left_wrist_up=observation.images.cam_left_wrist_up"
        ),
    )
    # Legacy/manual 3-camera options. Only used when --robot manual3 is set.
    parser.add_argument("--cam-high-key", default="observation.images.cam_high")
    parser.add_argument("--cam-left-wrist-key", default="observation.images.cam_left_wrist")
    parser.add_argument("--cam-right-wrist-key", default="observation.images.cam_right_wrist")

    parser.add_argument(
        "--video-backend",
        choices=["auto", "opencv", "ffmpeg", "imageio"],
        default="auto",
        help="auto uses ffmpeg for AV1 and OpenCV otherwise. Use ffmpeg if OpenCV shows AV1 decode errors.",
    )
    parser.add_argument(
        "--local-cache-dir",
        default=None,
        help="Optional local cache directory. Copies this episode's parquet and videos before reading, useful for /shared_disk.",
    )

    args = parser.parse_args()
    args.task_indices = _parse_optional_int_tokens(args.task_indices)
    if args.server_mode == "auto":
        args.server_mode = "python" if (args.policy_config or args.checkpoint_dir) else "script"
    if (args.policy_config or args.checkpoint_dir) and not args.start_server:
        args.start_server = True
    if args.start_server and args.server_mode == "python" and (not args.policy_config or not args.checkpoint_dir):
        raise ValueError("--server-mode python requires both --policy-config and --checkpoint-dir")
    if args.start_server and args.server_mode == "script" and (args.policy_config or args.checkpoint_dir):
        print(
            "Warning: --server-mode script uses --server-script and ignores --policy-config/--checkpoint-dir. "
            "Use --server-mode python if you want to pass them directly.",
            flush=True,
        )
    args.task_index = None
    if args.task_indices and args.task_name is not None:
        raise ValueError("Use either --dataset-index/--task-index or --task-name, not both")
    if isinstance(args.task_indices, list) and len(args.task_indices) > 1 and args.episode_index is not None:
        raise ValueError("--episode-index selects one fixed episode and cannot be combined with multiple --dataset-index values")
    if args.task_indices == "auto" and args.episode_index is not None:
        raise ValueError("--episode-index selects one fixed episode and cannot be combined with --dataset-index auto")
    if args.image_size == 0:
        args.image_size = None
    args.plot_horizons = _parse_int_list(args.plot_horizons)
    if args.query_stride <= 0:
        raise ValueError("--query-stride must be positive")
    if args.action_dim is not None and args.action_dim <= 0:
        raise ValueError("--action-dim must be positive")
    if args.episode_index is not None and args.episode_index < 0:
        raise ValueError("--episode-index must be non-negative")
    if args.robocasa_task_set != "none":
        if args.robot not in {"auto", "robocasa"}:
            raise ValueError("--robocasa-task-set requires --robot auto or --robot robocasa")
        if args.episode_index is not None:
            raise ValueError("--episode-index selects one fixed episode and cannot be combined with --robocasa-task-set")
    if args.ebench_task_set != "none":
        if args.robot not in {"auto", "ebench"}:
            raise ValueError("--ebench-task-set requires --robot auto or --robot ebench")
        if args.episode_index is not None:
            raise ValueError("--episode-index selects one fixed episode and cannot be combined with --ebench-task-set")
    if args.robocasa_task_set != "none" and args.ebench_task_set != "none":
        raise ValueError("--robocasa-task-set and --ebench-task-set cannot be used together")
    if args.episode_rank_in_task < 0:
        raise ValueError("--episode-rank-in-task must be non-negative")
    if args.compare_horizon is not None and args.compare_horizon <= 0:
        raise ValueError("--compare-horizon must be positive")
    for h in args.plot_horizons:
        if h <= 0:
            raise ValueError("--plot-horizons values must be positive")
    return args


def main() -> None:
    args = parse_args()
    server_process = _start_policy_server_if_requested(args)
    try:
        if args.robocasa_task_set != "none":
            task_jobs = _resolve_robocasa_task_paths(args.data_path, args.robocasa_task_set)
            base_output_dir = pathlib.Path(args.output_dir)
            checkpoint_step = _checkpoint_step_from_path(
                args.checkpoint_dir or os.environ.get("checkpoint") or os.environ.get("CHECKPOINT")
            )
            print(f"Resolved RoboCasa task set {args.robocasa_task_set}: {len(task_jobs)} task(s)", flush=True)
            for split_name, task_name, data_path in task_jobs:
                print(f"  {split_name}/{task_name}: {data_path}", flush=True)

            summary_rows = []
            for task_i, (split_name, task_name, data_path) in enumerate(task_jobs, start=1):
                eval_args = copy.copy(args)
                eval_args.robot = "robocasa"
                eval_args.data_path = str(data_path)
                eval_args.output_dir = str(base_output_dir)
                eval_args.output_prefix = f"{task_name}_{checkpoint_step}"
                eval_args.task_indices = None
                eval_args.task_index = None
                eval_args.task_name = None
                eval_args.episode_index = None
                print("\n" + "=" * 100)
                print(f"Evaluating RoboCasa task {task_i}/{len(task_jobs)}: {split_name}/{task_name}")
                print(f"data_path={eval_args.data_path}")
                print(f"output_dir={eval_args.output_dir}")
                print(f"output_prefix={eval_args.output_prefix}")
                print("=" * 100)

                result = run_eval(eval_args)
                save_outputs(result, eval_args)
                summary_rows.append(
                    _result_metric_row(
                        split_name=split_name,
                        task_name=task_name,
                        data_path=data_path,
                        output_prefix=eval_args.output_prefix,
                        result=result,
                    )
                )

            _save_robocasa_task_set_summary(summary_rows, base_output_dir, args.robocasa_task_set)
            return

        if args.ebench_task_set != "none":
            task_jobs = _resolve_ebench_task_paths(args.data_path, args.ebench_task_set)
            base_output_dir = pathlib.Path(args.output_dir)
            checkpoint_step = _checkpoint_step_from_path(
                args.checkpoint_dir or os.environ.get("checkpoint") or os.environ.get("CHECKPOINT")
            )
            print(f"Resolved EBench task set {args.ebench_task_set}: {len(task_jobs)} task(s)", flush=True)
            for split_name, task_name, data_path in task_jobs:
                print(f"  {split_name}/{task_name}: {data_path}", flush=True)

            summary_rows = []
            for task_i, (split_name, task_name, data_path) in enumerate(task_jobs, start=1):
                eval_args = copy.copy(args)
                eval_args.robot = "ebench"
                eval_args.data_path = str(data_path)
                eval_args.output_dir = str(base_output_dir)
                eval_args.output_prefix = f"{task_name}_{checkpoint_step}"
                eval_args.task_indices = None
                eval_args.task_index = None
                eval_args.task_name = None
                eval_args.episode_index = None
                print("\n" + "=" * 100)
                print(f"Evaluating EBench task {task_i}/{len(task_jobs)}: {split_name}/{task_name}")
                print(f"data_path={eval_args.data_path}")
                print(f"output_dir={eval_args.output_dir}")
                print(f"output_prefix={eval_args.output_prefix}")
                print("=" * 100)

                result = run_eval(eval_args)
                save_outputs(result, eval_args)
                summary_rows.append(
                    _result_metric_row(
                        split_name=split_name,
                        task_name=task_name,
                        data_path=data_path,
                        output_prefix=eval_args.output_prefix,
                        result=result,
                    )
                )

            _save_ebench_task_set_summary(summary_rows, base_output_dir, args.ebench_task_set)
            return

        if args.task_indices == "auto":
            task_indices = _auto_task_indices(pathlib.Path(args.data_path))
            print(f"--dataset-index auto resolved to task indices: {','.join(str(x) for x in task_indices)}", flush=True)
        else:
            task_indices = args.task_indices if args.task_indices is not None else [None]
        multi_task = len(task_indices) > 1
        base_output_dir = pathlib.Path(args.output_dir)

        for task_idx in task_indices:
            eval_args = copy.copy(args)
            eval_args.task_index = task_idx
            if multi_task:
                if task_idx is None:
                    subdir_name = "dataset_default"
                else:
                    subdir_name = f"dataset_index_{int(task_idx):03d}"
                eval_args.output_dir = str(base_output_dir / subdir_name)
                print("\n" + "=" * 100)
                print(f"Evaluating dataset/task index {task_idx} -> {eval_args.output_dir}")
                print("=" * 100)

            result = run_eval(eval_args)
            save_outputs(result, eval_args)
    except KeyboardInterrupt:
        print("Interrupted; shutting down.", flush=True)
    finally:
        _stop_policy_server(server_process, keep_server=args.keep_server)
        if not args.keep_server:
            _cleanup_multiprocessing_children()


if __name__ == "__main__":
    main()
