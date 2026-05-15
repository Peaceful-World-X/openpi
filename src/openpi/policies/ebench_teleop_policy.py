import dataclasses
import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _image_to_numpy_hwc(x):
    x = _to_numpy(x)

    # LeRobot image: [C, H, W] -> OpenPI image: [H, W, C]
    if x.ndim == 3 and x.shape[0] in (1, 3, 4):
        x = np.transpose(x, (1, 2, 0))

    if x.dtype != np.uint8:
        if np.max(x) <= 1.0:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def _check_last_dim(name, x, expected):
    if x.shape[-1] != expected:
        raise ValueError(f"{name} last dim should be {expected}, got {x.shape}.")
    return x


@dataclasses.dataclass(frozen=True)
class EBenchTeleopInputs:
    # 19-dim state/action:
    # state  = state.joints(12) + state.gripper(4) + state.base(3)
    # action = action.joints(12) + action.gripper(4) + action.base_delta(3)
    use_base: bool = True

    right_camera_key: str = "video.right_camera_view"
    left_camera_key: str = "video.left_camera_view"
    overlook_camera_key: str = "video.overlook_camera_view"
    top_camera_fallback_key: str = "video.top_camera_view"

    def __call__(self, data):
        state_joints = _check_last_dim("state.joints", _to_numpy(data["state.joints"]).astype(np.float32), 12)
        state_gripper = _check_last_dim("state.gripper", _to_numpy(data["state.gripper"]).astype(np.float32), 4)
        action_joints = _check_last_dim("action.joints", _to_numpy(data["action.joints"]).astype(np.float32), 12)
        action_gripper = _check_last_dim("action.gripper", _to_numpy(data["action.gripper"]).astype(np.float32), 4)

        state_parts = [state_joints, state_gripper]
        action_parts = [action_joints, action_gripper]

        if self.use_base:
            state_base = _check_last_dim("state.base", _to_numpy(data["state.base"]).astype(np.float32), 3)
            action_base_delta = _check_last_dim("action.base_delta", _to_numpy(data["action.base_delta"]).astype(np.float32), 3)
            state_parts.append(state_base)
            action_parts.append(action_base_delta)

        state = np.concatenate(state_parts, axis=-1).astype(np.float32)
        actions = np.concatenate(action_parts, axis=-1).astype(np.float32)

        image = {
            "base_0_rgb": _image_to_numpy_hwc(data[self.overlook_camera_key]),
            "left_wrist_0_rgb": _image_to_numpy_hwc(data[self.left_camera_key]),
            "right_wrist_0_rgb": _image_to_numpy_hwc(data[self.right_camera_key]),
        }

        # OpenPI Observation.from_dict 需要 image_mask。
        # True 表示这一路相机有效。
        image_mask = {
            "base_0_rgb": np.asarray(True),
            "left_wrist_0_rgb": np.asarray(True),
            "right_wrist_0_rgb": np.asarray(True),
        }

        # 关键：只返回 OpenPI 训练需要的字段，不保留原始 task 字段，否则字符串会进入 JAX batch。
        result = {
            "image": image,
            "image_mask": image_mask,
            "state": state,
            "actions": actions,
        }

        prompt = data.get("prompt", data.get("task", None))
        if prompt is not None:
            result["prompt"] = prompt

        return result


@dataclasses.dataclass(frozen=True)
class EBenchTeleopOutputs:
    use_base: bool = True

    def __call__(self, data):
        data = dict(data)
        action_dim = 19 if self.use_base else 16
        data["actions"] = data["actions"][..., :action_dim]
        return data
