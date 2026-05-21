#!/usr/bin/env python3
from time import perf_counter

import numpy as np
from openpi_client import websocket_client_policy


state_dim = 19      # "ebench"
state_dim = 16      # "robocasa"


def make_obs() -> dict:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    print(f"[states] state_dim={state_dim}，image_shape={image.shape}")
    return {
        "image": {
            "base_0_rgb": image,
            "left_wrist_0_rgb": image,
            "right_wrist_0_rgb": image,
        },
        "image_mask": {
            "base_0_rgb": np.asarray(True),
            "left_wrist_0_rgb": np.asarray(True),
            "right_wrist_0_rgb": np.asarray(True),
        },
        "state": np.zeros((state_dim,), dtype=np.float32),
        "prompt": "pick up the object",
    }


def print_result(result: dict) -> None:
    if not isinstance(result, dict) or "actions" not in result:
        print(f"[error] bad result: {type(result)}, keys={list(result.keys()) if isinstance(result, dict) else None}")
        return
    actions = np.asarray(result["actions"])
    print(f"[action] shape={actions.shape}")


def main() -> None:
    obs = make_obs()
    policy = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8000)
    _ = policy.infer(obs)

    start = perf_counter()
    result = policy.infer(obs)
    elapsed_ms = (perf_counter() - start) * 1000
    print(f"[timing] {elapsed_ms:.3f} ms")
    print_result(result)


if __name__ == "__main__":
    main()
