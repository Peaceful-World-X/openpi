"""Policy transforms for Agilex bag dataset with three camera inputs."""

import dataclasses
from typing import ClassVar

import numpy as np
import torch

import openpi.models.model as _model
import openpi.transforms as transforms


def _parse_image(image) -> np.ndarray:
    """Normalize image layout/type to uint8 HWC."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class AgilexBagImageInputs(transforms.DataTransformFn):
    """Convert Agilex bag 3-camera data format into model input format.

    Three-camera mode:
    - base camera: global_camera
    - left wrist camera: pikaGripperFisheyeCamera
    - right wrist camera: pikaGripperDepthCamera (RGB stream from gripper-side RealSense)
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05

    BASE_CAMERA_ALIASES: ClassVar[tuple[str, ...]] = ("global_camera", "top_head", "base_0_rgb")
    LEFT_CAMERA_ALIASES: ClassVar[tuple[str, ...]] = (
        "pikaGripperFisheyeCamera",
        "hand_left",
        "left_wrist_0_rgb",
    )
    RIGHT_CAMERA_ALIASES: ClassVar[tuple[str, ...]] = (
        "pikaGripperDepthCamera",
        "hand_right",
        "right_wrist_0_rgb",
    )

    def _pick_image(self, in_images: dict, aliases: tuple[str, ...]) -> tuple[np.ndarray | None, bool]:
        for key in aliases:
            if key in in_images:
                return _parse_image(in_images[key]), True
        return None, False

    def __call__(self, data: dict) -> dict:
        if "images" not in data:
            raise ValueError("Expected key 'images' in input data.")

        in_images = data["images"]
        if not isinstance(in_images, dict):
            raise ValueError(f"Expected 'images' to be dict, got {type(in_images)}")

        base_image, base_found = self._pick_image(in_images, self.BASE_CAMERA_ALIASES)
        if not base_found:
            raise ValueError(f"Missing base camera image. Expected one of: {self.BASE_CAMERA_ALIASES}")

        left_image, left_found = self._pick_image(in_images, self.LEFT_CAMERA_ALIASES)
        if not left_found:
            raise ValueError(f"Missing fisheye camera image. Expected one of: {self.LEFT_CAMERA_ALIASES}")

        right_image, right_found = self._pick_image(in_images, self.RIGHT_CAMERA_ALIASES)
        if not right_found:
            raise ValueError(f"Missing depth camera image. Expected one of: {self.RIGHT_CAMERA_ALIASES}")

        parsed_images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": left_image,
            "right_wrist_0_rgb": right_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }

        inputs = {
            "state": data["state"],
            "image": parsed_images,
            "image_mask": image_masks,
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AgilexBagImageOutputs(transforms.DataTransformFn):
    """Convert model output actions back to environment action dimension."""

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
