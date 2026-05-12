"""Fast norm stats computation - self-contained, no config dependency.

Reads parquet files directly with multi-threading and applies the same transforms
as the Aloha pipeline inline (pad, zero mask, delta actions).
"""

import json
import pathlib
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
import tqdm
import tyro

import openpi.shared.normalize as normalize


# ---------------------------------------------------------------------------
# Helpers copied from transforms.py / aloha_policy.py
# ---------------------------------------------------------------------------

def _pad_to_dim(x: np.ndarray, target_dim: int) -> np.ndarray:
    if x.shape[-1] >= target_dim:
        return x[..., :target_dim]
    pad = np.zeros((*x.shape[:-1], target_dim - x.shape[-1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=-1)


def _make_bool_mask(*dims: int) -> tuple[bool, ...]:
    result = []
    for d in dims:
        if d > 0:
            result.extend([True] * d)
        else:
            result.extend([False] * (-d))
    return tuple(result)


def _window_actions(actions_2d: np.ndarray, horizon: int) -> np.ndarray:
    num_steps, action_dim = actions_2d.shape
    windows = np.empty((num_steps, horizon, action_dim), dtype=actions_2d.dtype)
    last = actions_2d[-1]
    for t in range(num_steps):
        end = min(t + horizon, num_steps)
        valid = actions_2d[t:end]
        windows[t, : len(valid)] = valid
        if len(valid) < horizon:
            windows[t, len(valid) :] = last
    return windows


def _parse_value(v) -> np.ndarray:
    if isinstance(v, bytes):
        return np.frombuffer(v, dtype=np.float64).copy()
    return np.asarray(v, dtype=np.float64)


# ---------------------------------------------------------------------------
# adapt_to_pi conversion (copied from aloha_policy.py)
# ---------------------------------------------------------------------------

_JOINT_FLIP_MASK = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)
    arm_length, horn_radius = 0.036, 0.022
    cos_val = (horn_radius**2 + value**2 - arm_length**2) / (2 * horn_radius * value)
    value = np.arcsin(np.clip(cos_val, -1.0, 1.0))
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular_inv(value):
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_state_batch(state: np.ndarray) -> np.ndarray:
    """Apply _decode_state(adapt_to_pi=True) to batched state [N, 14]."""
    state = state * _JOINT_FLIP_MASK
    state[:, [6, 13]] = _gripper_to_angular(state[:, [6, 13]])
    return state


def _encode_actions_inv_batch(actions: np.ndarray) -> np.ndarray:
    """Apply _encode_actions_inv(adapt_to_pi=True) to batched actions [N, H, 14]."""
    actions = actions * _JOINT_FLIP_MASK
    actions[:, :, [6, 13]] = _gripper_from_angular_inv(actions[:, :, [6, 13]])
    return actions


# ---------------------------------------------------------------------------
# Inline transforms matching AlohaInputs (adapt_to_pi=False) + DeltaActions
# ---------------------------------------------------------------------------

def _apply_transforms(
    states_raw: np.ndarray,       # [N, raw_dim]
    actions_windowed: np.ndarray,  # [N, H, raw_dim]
    action_dim: int,
    delta_mask: tuple[bool, ...] | None,
    zero_mask: tuple[bool, ...] | None,
    adapt_to_pi: bool,
) -> tuple[np.ndarray, np.ndarray]:
    # _decode_aloha / _decode_state — applies only to the first 14 dims
    # (standard Aloha joint space). Extra dims (e.g. waist joints) pass
    # through unchanged and are preserved after the conversion.
    if adapt_to_pi:
        front_state = _decode_state_batch(states_raw[..., :14])
        front_actions = _encode_actions_inv_batch(actions_windowed[..., :14])
        if states_raw.shape[-1] > 14:
            states_raw = np.concatenate([front_state, states_raw[..., 14:]], axis=-1)
        else:
            states_raw = front_state
        if actions_windowed.shape[-1] > 14:
            actions_windowed = np.concatenate([front_actions, actions_windowed[..., 14:]], axis=-1)
        else:
            actions_windowed = front_actions

    # AlohaInputs: pad_to_dim
    state = _pad_to_dim(states_raw, action_dim)
    actions = _pad_to_dim(actions_windowed, action_dim)

    # AlohaInputs: zero masking
    if zero_mask is not None:
        zm = np.asarray(zero_mask, dtype=bool)
        state[:, zm] = 0
        actions[:, :, zm] = 0

    # DeltaActions
    if delta_mask is not None:
        dims = len(delta_mask)
        mask = np.asarray(delta_mask, dtype=bool)
        state_base = np.where(mask, state[:, :dims], 0)
        actions[:, :, :dims] -= state_base[:, np.newaxis, :]

    return state, actions


# ---------------------------------------------------------------------------
# Parquet processing
# ---------------------------------------------------------------------------

def _collect_parquet_files(data_paths: list[str]) -> list[str]:
    parquet_files: list[str] = []
    for data_path in data_paths:
        base = pathlib.Path(data_path)
        info_file = base / "meta" / "info.json"
        if info_file.is_file():
            info = json.loads(info_file.read_text())
            total_episodes = info["total_episodes"]
            chunks_size = info.get("chunks_size", 1000)
            candidates = [
                base / f"data/chunk-{ep // chunks_size:03d}/episode_{ep:06d}.parquet"
                for ep in range(total_episodes)
            ]
            with ThreadPoolExecutor(max_workers=32) as ex:
                results = list(tqdm.tqdm(
                    ex.map(lambda f: f if f.is_file() else None, candidates),
                    total=len(candidates),
                    desc=f"Listing episodes in {base.name}",
                ))
            parquet_files.extend(str(f) for f in results if f is not None)
        else:
            # Fallback for non-LeRobot datasets: scan directory tree
            data_subdir = base / "data"
            search_dir = data_subdir if data_subdir.is_dir() else base
            it = search_dir.rglob("*.parquet")
            parquet_files.extend(str(p) for p in tqdm.tqdm(it, desc=f"Scanning {base.name}"))
    parquet_files.sort()
    return parquet_files


def _process_single_parquet(
    parquet_file: str,
    action_horizon: int,
    action_dim: int,
    delta_mask: tuple[bool, ...] | None,
    zero_mask: tuple[bool, ...] | None,
    adapt_to_pi: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"Skip unreadable parquet {parquet_file}: {e}")
        return None

    state_col = None
    for key in ["observation.state", "observation.state_tcp_endpose_quat", "observation.state_vr_endpose_quat"]:
        if key in df.columns:
            state_col = key
            break

    action_col = None
    for key in ["action", "action_tcp_endpose_quat", "action_vr_endpose_quat"]:
        if key in df.columns:
            action_col = key
            break

    if state_col is None or action_col is None:
        return None

    try:
        states_raw = np.stack([_parse_value(v) for v in df[state_col].to_list()])
        actions_raw = np.stack([_parse_value(v) for v in df[action_col].to_list()])

        # Group by episode to respect boundaries during action windowing
        if "episode_index" in df.columns:
            ep_indices = df["episode_index"].values
            unique_eps = np.unique(ep_indices)
        else:
            unique_eps = [None]

        all_states = []
        all_actions = []

        for ep_id in unique_eps:
            if ep_id is not None:
                ep_mask = ep_indices == ep_id
                ep_states = states_raw[ep_mask]
                ep_actions = actions_raw[ep_mask]
            else:
                ep_states = states_raw
                ep_actions = actions_raw

            ep_actions_windowed = _window_actions(ep_actions, action_horizon)
            ep_state, ep_actions = _apply_transforms(
                ep_states, ep_actions_windowed, action_dim, delta_mask, zero_mask, adapt_to_pi,
            )
            all_states.append(ep_state)
            all_actions.append(ep_actions)

        if not all_states:
            return None

        return np.concatenate(all_states), np.concatenate(all_actions)
    except Exception as e:
        print(f"Skip malformed parquet {parquet_file}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_path: str,
    output_path: str,
    action_horizon: int = 50,
    action_dim: int = 32,
    mask_list: list[int] | None = None,
    zero_mask_list: list[int] | None = None,
    adapt_to_pi: bool = False,
    num_workers: int = 16,
):
    """Compute norm stats directly from parquet files.

    Args:
        data_path: Path to LeRobot dataset directory.
        output_path: Directory to write norm_stats.json into.
        action_horizon: Number of future action frames per window.
        action_dim: Padding dimension for state and actions.
        mask_list: Delta mask dims, e.g. [7,-1,7,-1,4].
        zero_mask_list: Zero-out mask dims, same format. True dims are zeroed.
        adapt_to_pi: If true, convert Aloha joint/gripper space to pi internal space.
        num_workers: Number of threads for parallel parquet reading.
    """
    delta_mask = _make_bool_mask(*mask_list) if mask_list else None
    zero_mask = _make_bool_mask(*zero_mask_list) if zero_mask_list else None

    parquet_files = _collect_parquet_files([data_path])
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_path}")

    print(f"Found {len(parquet_files)} parquet files")
    print(f"action_horizon={action_horizon}, action_dim={action_dim}, num_workers={num_workers}")

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    total_steps = 0
    processed = 0
    pbar = tqdm.tqdm(total=len(parquet_files), desc="Processing parquet files")

    def consume(result):
        nonlocal total_steps, processed
        pbar.update(1)
        if result is None:
            return
        state, actions = result
        stats["state"].update(state)
        stats["actions"].update(actions)
        total_steps += state.shape[0]
        processed += 1

    if num_workers <= 1:
        try:
            for pf in parquet_files:
                consume(_process_single_parquet(pf, action_horizon, action_dim, delta_mask, zero_mask, adapt_to_pi))
        finally:
            pbar.close()
    else:
        max_in_flight = max(num_workers * 2, num_workers + 1)
        executor = ThreadPoolExecutor(max_workers=num_workers)
        pending: set = set()
        order: deque = deque()
        results_by_future: dict = {}
        file_iter = iter(parquet_files)

        def submit_next() -> bool:
            try:
                pf = next(file_iter)
            except StopIteration:
                return False
            fut = executor.submit(
                _process_single_parquet, pf, action_horizon, action_dim, delta_mask, zero_mask, adapt_to_pi
            )
            pending.add(fut)
            order.append(fut)
            return True

        try:
            for _ in range(max_in_flight):
                if not submit_next():
                    break

            while order:
                if pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        pending.discard(fut)
                        results_by_future[fut] = fut.result()

                while order and order[0] in results_by_future:
                    head = order.popleft()
                    consume(results_by_future.pop(head))
                    submit_next()
        finally:
            executor.shutdown(wait=True)
            pbar.close()

    if total_steps < 2:
        raise ValueError("Insufficient valid samples (<2) to compute normalization stats")

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    out = pathlib.Path(output_path)
    print(f"Writing stats to: {out}")
    normalize.save(out, norm_stats)
    print(f"Processed {processed}/{len(parquet_files)} files, {total_steps} total steps")


if __name__ == "__main__":
    tyro.cli(main)
