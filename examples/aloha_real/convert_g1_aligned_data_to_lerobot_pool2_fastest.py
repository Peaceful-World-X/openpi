"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal
from PIL import Image
import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import cv2
import os
import imageio.v3 as iio
from concurrent.futures import ThreadPoolExecutor


camera_name_map = {
    "head_color": "cam_high", 
    "hand_left_color": "cam_left_wrist", 
    "hand_right_color": "cam_right_wrist"
}

# camera_name_map = {
#     "head_center_fisheye_color": "cam_high", 
#     "hand_left_color": "cam_left_wrist", 
#     "hand_right_color": "cam_right_wrist"
# }


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    out_dir: Path,
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_7",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_7",
        "left_gripper",
    ]
    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        root=out_dir,
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: Path, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_images_per_camera_for_g1(ep: Path, camera_name_map: dict[str, str]) -> dict[str, np.ndarray]:
    """
    从 episode 目录加载所有相机的时序帧图像
    返回: {camera_name: np.ndarray (T, H, W, C)}
    """
    imgs_per_cam: dict[str, list[np.ndarray]] = {}

    camera_dir = ep / "camera"
    frame_dirs = sorted([p for p in camera_dir.iterdir() if p.is_dir() and p.name.isdigit()],
                        key=lambda x: int(x.name))

    for frame_dir in frame_dirs:
        for img_file in sorted(frame_dir.iterdir()):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            cam_name = img_file.stem
            if cam_name not in ["head_color", "hand_left_color", "hand_right_color"]:
                continue
            print(img_file)
            img = np.array(Image.open(img_file).convert("RGB"))

            camera_name = camera_name_map[cam_name]

            if camera_name not in imgs_per_cam:
                imgs_per_cam[camera_name] = []
            imgs_per_cam[camera_name].append(img)

    # 转成 np.ndarray
    for camera_name in imgs_per_cam:
        imgs_per_cam[camera_name] = np.stack(imgs_per_cam[camera_name], axis=0)  # (T, H, W, C)

    return imgs_per_cam


def load_raw_images_per_camera_for_g1_pool(ep: Path, camera_name_map: dict[str, str], worker_num: int = 8) -> dict[str, np.ndarray]:
    imgs_per_cam: dict[str, list[np.ndarray]] = {}

    camera_dir = ep / "camera"
    frame_dirs = sorted([p for p in camera_dir.iterdir() if p.is_dir() and p.name.isdigit()],
                        key=lambda x: int(x.name))

    img_files = []
    for frame_dir in frame_dirs:
        for img_file in frame_dir.iterdir():
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                cam_name = img_file.stem
                if cam_name in ["head_color", "hand_left_color", "hand_right_color"]:
                    img_files.append((img_file, camera_name_map[cam_name]))

    # def _load_one(args):
    #     img_file, camera_name = args
    #     return camera_name, np.array(Image.open(img_file).convert("RGB"))

    def _load_one(args):
        img_file, camera_name = args
        img = cv2.cvtColor(cv2.imread(str(img_file), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return camera_name, img

    with ThreadPoolExecutor(max_workers=worker_num) as ex:
        results = list(ex.map(_load_one, img_files))

    # 按照 camera_name 聚合
    for camera_name, img in results:
        if camera_name not in imgs_per_cam:
            imgs_per_cam[camera_name] = []
        imgs_per_cam[camera_name].append(img)

    # stack
    for camera_name in imgs_per_cam:
        imgs_per_cam[camera_name] = np.stack(imgs_per_cam[camera_name], axis=0)

    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    ep_path_h5 = ep_path / "aligned_joints.h5"
    with h5py.File(ep_path_h5, "r") as ep:
        # state = torch.from_numpy(ep["/observations/qpos"][:])
        # action = torch.from_numpy(ep["/action"][:])

        state_joint = ep["/state/joint/position"][:]                 # (T, 14)
        state_left = ep["/state/left_effector/position"][:]          # (T, 1)
        state_right = ep["/state/right_effector/position"][:]        # (T, 1)

        action_joint = ep["/action/joint/position"][:]               # (T, 14)
        action_left = ep["/action/left_effector/position"][:]        # (T, 1)
        action_right = ep["/action/right_effector/position"][:]      # (T, 1)

        state_joint_aug = np.concatenate(
            [state_joint[:, :7], state_left, state_joint[:, 7:], state_right],
            axis=1
        )  # (T, 16)

        action_joint_aug = np.concatenate(
            [action_joint[:, :7], action_left, action_joint[:, 7:], action_right],
            axis=1
        )  # (T, 16)

        state = torch.from_numpy(state_joint_aug)
        action = torch.from_numpy(action_joint_aug)

        velocity = None
        # if "/observations/qvel" in ep:
        #     velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        # if "/observations/effort" in ep:
        #     effort = torch.from_numpy(ep["/observations/effort"][:])

    # imgs_per_cam = load_raw_images_per_camera(
    #     ep_path,
    #     [
    #         "cam_high",
    #         "cam_left_wrist",
    #         "cam_right_wrist",
    #     ],
    # )
    print("ffffffff")
    # imgs_per_cam = load_raw_images_per_camera_for_g1(
    #     ep_path,
    #     camera_name_map
    # )
    imgs_per_cam = load_raw_images_per_camera_for_g1_pool(
        ep_path,
        camera_name_map,
        worker_num=128
    )

    return imgs_per_cam, state, action, velocity, effort


# 获取h5数据集工具
def _get_h5_dset(file_handle, candidates):
    for path in candidates:
        try:
            if path in file_handle:
                return file_handle[path][:]
        except Exception:
            continue
    raise KeyError(f"None of candidate datasets found: {candidates}")


# 读取已对齐的关节与夹爪state数据（支持多种键名）
def read_aligned_joint_state_data(file_path):
    """读取已对齐的关节与夹爪数据（适配 aligned_joints.h5）。
    优先尝试与原始结构一致的路径，不存在则尝试常见备选键。
    """
    with h5py.File(file_path, 'r') as f:
        joint_positions = _get_h5_dset(f, [
            'state/joint/position',
            'joint/position',
            'position',
        ])
        # 关节速度：可能不存在或为空，若为空则创建零数组
        try:
            joint_velocities = _get_h5_dset(f, [
                'state/joint/velocity',
                'joint/velocity',
                'velocity',
            ])
            if not hasattr(joint_velocities, 'shape') or joint_velocities.shape == () or joint_velocities.shape[0] == 0:
                raise ValueError('empty velocity')
        except Exception:
            joint_velocities = np.zeros_like(joint_positions, dtype=np.float64)
        # 对齐场景下时间戳可选，缺失时用等间隔索引代替
        try:
            joint_timestamps = _get_h5_dset(f, [
                'state/joint/timestamp',
                'joint/timestamp',
                'timestamp',
            ])
        except Exception:
            joint_timestamps = np.arange(len(joint_positions), dtype=np.int64)

        # 夹爪位置
        left_effector_pos = _get_h5_dset(f, [
            'state/left_effector/position',
            'left_effector/position',
            'left_gripper/position',
            'left_gripper_position',
        ])
        right_effector_pos = _get_h5_dset(f, [
            'state/right_effector/position',
            'right_effector/position',
            'right_gripper/position',
            'right_gripper_position',
        ])
        # 夹爪时间戳可选
        try:
            left_effector_timestamps = _get_h5_dset(f, [
                'state/left_effector/timestamp',
                'left_effector/timestamp',
                'left_gripper/timestamp',
            ])
        except Exception:
            left_effector_timestamps = np.arange(len(left_effector_pos), dtype=np.int64)
        try:
            right_effector_timestamps = _get_h5_dset(f, [
                'state/right_effector/timestamp',
                'right_effector/timestamp',
                'right_gripper/timestamp',
            ])
        except Exception:
            right_effector_timestamps = np.arange(len(right_effector_pos), dtype=np.int64)

        # 关节名称可选
        try:
            joint_names = f['state/joint'].attrs.get('name', [])
            joint_names = joint_names.astype(str)
        except Exception:
            joint_names = []

        # 夹爪名称可选
        try:
            left_gripper_name = f['state/left_effector'].attrs['name'].astype(str)[0]
        except Exception:
            left_gripper_name = 'left_gripper'
        try:
            right_gripper_name = f['state/right_effector'].attrs['name'].astype(str)[0]
        except Exception:
            right_gripper_name = 'right_gripper'

        return {
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'joint_timestamps': joint_timestamps,
            'joint_names': joint_names,
            'left_effector_pos': left_effector_pos,
            'right_effector_pos': right_effector_pos,
            'left_effector_timestamps': left_effector_timestamps,
            'right_effector_timestamps': right_effector_timestamps,
            'left_gripper_name': left_gripper_name,
            'right_gripper_name': right_gripper_name,
        }


# 读取已对齐的关节与夹爪action数据
def read_aligned_joint_action_data(file_path):
    """读取已对齐的关节与夹爪数据（适配 aligned_joints.h5）。

    """
    with h5py.File(file_path, 'r') as f:
        joint_positions = _get_h5_dset(f, [
            'action/joint/position',
        ])
        # 关节速度：可能不存在或为空，若为空则创建零数组
        try:
            joint_velocities = _get_h5_dset(f, [
                'action/joint/velocity',
            ])
            if not hasattr(joint_velocities, 'shape') or joint_velocities.shape == () or joint_velocities.shape[0] == 0:
                raise ValueError('empty velocity')
        except Exception:
            joint_velocities = np.zeros_like(joint_positions, dtype=np.float64)
        # 对齐场景下时间戳可选，缺失时用等间隔索引代替
        try:
            joint_timestamps = _get_h5_dset(f, [
                'action/joint/timestamp',
            ])
        except Exception:
            joint_timestamps = np.arange(len(joint_positions), dtype=np.int64)

        # 夹爪位置
        left_effector_pos = _get_h5_dset(f, [
            'action/left_effector/position',
        ])
        right_effector_pos = _get_h5_dset(f, [
            'action/right_effector/position',
        ])
        # 夹爪时间戳可选
        try:
            left_effector_timestamps = _get_h5_dset(f, [
                'action/left_effector/timestamp',
            ])
        except Exception:
            left_effector_timestamps = np.arange(len(left_effector_pos), dtype=np.int64)
        try:
            right_effector_timestamps = _get_h5_dset(f, [
                'action/right_effector/timestamp',
            ])
        except Exception:
            right_effector_timestamps = np.arange(len(right_effector_pos), dtype=np.int64)

        # 关节名称可选
        try:
            joint_names = f['action/joint'].attrs.get('name', [])
            joint_names = joint_names.astype(str)
        except Exception:
            joint_names = []

        # 夹爪名称可选
        try:
            left_gripper_name = f['action/left_effector'].attrs['name'].astype(str)[0]
        except Exception:
            left_gripper_name = 'left_gripper'
        try:
            right_gripper_name = f['action/right_effector'].attrs['name'].astype(str)[0]
        except Exception:
            right_gripper_name = 'right_gripper'

        return {
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'joint_timestamps': joint_timestamps,
            'joint_names': joint_names,
            'left_effector_pos': left_effector_pos,
            'right_effector_pos': right_effector_pos,
            'left_effector_timestamps': left_effector_timestamps,
            'right_effector_timestamps': right_effector_timestamps,
            'left_gripper_name': left_gripper_name,
            'right_gripper_name': right_gripper_name,
        }


# 按帧索引目录列举（仅数字目录, 从小到大，时序）
def get_frame_indices_from_camera_root(camera_root):
    if not os.path.exists(camera_root):
        return []
    indices = []
    for name in os.listdir(camera_root):
        if name.isdigit() and os.path.isdir(os.path.join(camera_root, name)):
            indices.append(int(name))
    indices.sort()
    return indices


# 基于帧索引构造三路相机图片路径
def get_frame_image_paths(camera_root, frame_index):
    frame_dir = os.path.join(camera_root, str(frame_index))
    head_path = os.path.join(frame_dir, 'head_color.jpg')
    left_path = os.path.join(frame_dir, 'hand_left_color.jpg')
    right_path = os.path.join(frame_dir, 'hand_right_color.jpg')
    return head_path, left_path, right_path


def load_image_from_path(image_path):
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found: {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to load image: {image_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def create_16d_state_action(joint_positions, left_gripper_pos, right_gripper_pos, idx, scale=1):
    """创建16维的state和action数据
    
    Args:
        joint_positions: 关节位置数据
        left_gripper_pos: 左夹爪位置数据
        right_gripper_pos: 右夹爪位置数据
        idx: 时间索引
    
    Returns:
        16维的numpy数组: [左臂关节(7) + 左夹爪(1) + 右臂关节(7) + 右夹爪(1)]
    """
    # 左臂关节 (0-6)
    left_arm_joints = joint_positions[idx, :7]
    # 右臂关节 (7-13) - 修复：应该是7:14而不是7:
    right_arm_joints = joint_positions[idx, 7:14]
    # 左夹爪 (7)
    left_gripper = left_gripper_pos[idx]
    # 右夹爪 (15)
    right_gripper = right_gripper_pos[idx]
    
    # 组合成16维数组
    state_16d = np.concatenate([
        left_arm_joints,            # 0-6: 左臂关节
        [left_gripper / scale],       # 7: 左夹爪
        right_arm_joints,           # 8-14: 右臂关节 (正确：7个值)
        [right_gripper / scale]       # 15: 右夹爪
    ])
    
    return state_16d


def preload_all_images(camera_root, frame_indices):
    """一次性多线程读取所有帧的三路相机图片到内存"""
    all_images = {}

    tasks = []
    for frame_idx in frame_indices:
        head_img_path, left_img_path, right_img_path = get_frame_image_paths(camera_root, frame_idx)
        tasks.append((frame_idx, 'head', head_img_path))
        tasks.append((frame_idx, 'left', left_img_path))
        tasks.append((frame_idx, 'right', right_img_path))

    def _load_one(task):
        frame_idx, cam_type, img_path = task
        img = load_image_from_path(img_path)
        return frame_idx, cam_type, img

    worker_num = os.cpu_count() or 8
    with ThreadPoolExecutor(max_workers=worker_num) as ex:
        for frame_idx, cam_type, img in ex.map(_load_one, tasks):
            if frame_idx not in all_images:
                all_images[frame_idx] = {}
            all_images[frame_idx][cam_type] = img

    return all_images


def preload_all_images_from_videos(videos_dir):
    imgs_per_cam: dict[str, np.ndarray] = {}
    for cam_name in camera_name_map:
        video_path = os.path.join(videos_dir, f'{cam_name}.mp4')
        print(f"Loading video: {video_path}")
        frames = iio.imread(video_path)
        camera_name = camera_name_map[cam_name]
        imgs_per_cam[camera_name] = frames
        print(f"Loaded {camera_name} with imageio: shape {frames.shape}")
    return imgs_per_cam


def populate_dataset(
    dataset: LeRobotDataset,
    # hdf5_files: list[Path],
    raw_dirs: Path,
    task: str,
    log_path: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "w")
    if episodes is None:
        episodes = range(len(raw_dirs))

    for ep_idx in tqdm.tqdm(episodes):
        # if ep_idx > 0:
        #     break
        ep_dir = raw_dirs[ep_idx]
        print("Processing data path:", ep_dir)
        aligned_h5 = os.path.join(ep_dir, "aligned_joints.h5")
        joint_state_data = read_aligned_joint_state_data(aligned_h5)
        joint_action_data = read_aligned_joint_action_data(aligned_h5)

        videos_dir = os.path.join(ep_dir, 'observations/videos')
        all_images = preload_all_images_from_videos(videos_dir)

        if all_images['cam_high'].shape[0] != all_images['cam_left_wrist'].shape[0] != all_images['cam_right_wrist'].shape[0]:
            log.write(f"[ERROR] Mismatch between the number of images. cam_high_num={all_images['cam_high'].shape[0]}, cam_left_wrist_num={all_images['cam_left_wrist'].shape[0]}, cam_right_wrist_num={all_images['cam_right_wrist'].shape[0]}\n")
            continue

        worker_num = os.cpu_count() or 8
        print("worker_num:", worker_num)
        frame_indices = list(range(all_images['cam_high'].shape[0]))

        if len(joint_state_data['joint_positions']) != len(frame_indices):
            log.write(f"[ERROR] Mismatch between the number of joints and images. joints_num={len(joint_state_data['joint_positions'])}, images_num={len(frame_indices)}, episode_dir={ep_dir}\n")
            continue

        def _prepare_one(local_i: int):
            try:
                frame_idx = frame_indices[local_i]
                next_local_i = local_i + 1

                # head_img_path, left_img_path, right_img_path = get_frame_image_paths(camera_root, frame_idx)
                # head_img = load_image_from_path(head_img_path)
                # hand_left_img = load_image_from_path(left_img_path)
                # hand_right_img = load_image_from_path(right_img_path)

                head_img = all_images['cam_high'][frame_idx]
                hand_left_img = all_images['cam_left_wrist'][frame_idx]
                hand_right_img = all_images['cam_right_wrist'][frame_idx]

                # TODO
                if head_img is None or hand_left_img is None or hand_right_img is None:
                    warn_info = f"Reading image error at index {frame_idx} (local {local_i}) in {ep_dir}."
                    log.write(f"[WARN] {warn_info}\n")
                    return local_i, None

                current_state = create_16d_state_action(
                    joint_state_data['joint_positions'],
                    joint_state_data['left_effector_pos'].flatten(),
                    joint_state_data['right_effector_pos'].flatten(),
                    local_i,
                    scale=120,
                )
                next_state = create_16d_state_action(
                    joint_state_data['joint_positions'],
                    joint_state_data['left_effector_pos'].flatten(),
                    joint_state_data['right_effector_pos'].flatten(),
                    next_local_i,
                    scale=120,
                )
                current_action = create_16d_state_action(
                    joint_action_data['joint_positions'],
                    joint_action_data['left_effector_pos'].flatten(),
                    joint_action_data['right_effector_pos'].flatten(),
                    local_i,
                    scale=1,
                )
                current_velocity = np.concatenate([
                    joint_state_data['joint_velocities'][local_i, :7],
                    [0.0],
                    joint_state_data['joint_velocities'][local_i, 7:14],
                    [0.0],
                ])
                effort = np.zeros(16)

                payload = (
                    frame_idx,
                    head_img,
                    hand_left_img,
                    hand_right_img,
                    current_state,
                    next_state,
                    current_action,
                    current_velocity,
                    effort,
                )
                return local_i, payload
            except Exception as e:
                warn_info = f"Maybe reading joint error at index {local_i} (local {local_i}) in {ep_dir}: {str(e)}. You can ignore this issue if it is the last index."
                log.write(f"[WARN] {warn_info}\n")
                return local_i, None

        with ThreadPoolExecutor(max_workers=worker_num) as ex:
            for local_i, payload in ex.map(_prepare_one, frame_indices):
                # TODO
                if payload is None:
                    continue
                frame_idx, head_img, hand_left_img, hand_right_img, current_state, next_state, current_action, current_velocity, effort = payload
                try:
                    frame_data = {
                        "action": torch.from_numpy(next_state.astype(np.float32)),
                        "observation.images.cam_high": head_img,
                        "observation.images.cam_left_wrist": hand_left_img,
                        "observation.images.cam_right_wrist": hand_right_img,
                        "observation.state": torch.from_numpy(current_state.astype(np.float32))
                    }
                    dataset.add_frame(frame_data)
                except Exception as e:
                    err_info = f"Error processing frame index {frame_idx} (local {local_i}) in {ep_dir}: {str(e)}."
                    log.write(f"[ERROR] {err_info}\n")
                    print(err_info)
                    continue

        # print('dddddd')
        # imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        # num_frames = state.shape[0]

        # for i in range(num_frames):
        #     frame = {
        #         "observation.state": state[i],
        #         "action": action[i],
        #     }

        #     for camera, img_array in imgs_per_cam.items():
        #         frame[f"observation.images.{camera}"] = img_array[i]

        #     if velocity is not None:
        #         frame["observation.velocity"] = velocity[i]
        #     if effort is not None:
        #         frame["observation.effort"] = effort[i]

        #     dataset.add_frame(frame)

        dataset.save_episode(task=task)

    log.close()

    return dataset


def port_aloha(
    out_dir: Path,
    raw_dir: Path,
    repo_id: str,
    log_path: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        download_raw(raw_dir, repo_id=raw_repo_id)

    # hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    # raw_dirs = sorted([p for p in raw_dir.glob("*") if p.is_dir()])

    # raw_dir = "/mnt/pfs/users/yang.wang/data/giga_g1_data/fold_shirt/episodes_173/obs-shucai/framing/GENIE/52"
    # */*/* = 872/A2D0013B200333/29791
    # raw_dirs = sorted([p for p in raw_dir.glob("*/*/*") if p.is_dir()])

    # 获取所有三级目录
    all_dirs = sorted([p for p in raw_dir.glob("*/*/*") if p.is_dir()])
    raw_dirs = all_dirs
    # 862 这一组
    # raw_dirs = [p for p in all_dirs if p.parts[-3] == "862"]
    # 872 和 1118 这一组
    # raw_dirs_2 = [p for p in all_dirs if p.parts[-3] in {"872", "1118"}]

    dataset = create_empty_dataset(
        out_dir,
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        # has_effort=has_effort(hdf5_files),
        # has_velocity=has_velocity(hdf5_files),
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )

    dataset = populate_dataset(
        dataset,
        # hdf5_files,
        raw_dirs,
        task=task,
        episodes=episodes,
        log_path=log_path
    )
    # dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)
