"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
import logging.handlers
from pathlib import Path
import shutil
import time
from typing import Literal
from multiprocessing import Process

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import logging

log_dir = Path('log')
if not log_dir.is_dir():
    log_dir.mkdir(parents=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 16
    image_writer_threads: int = 8
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
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
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


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
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


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    try:
        with h5py.File(ep_path, "r") as ep:
            state = torch.from_numpy(ep["/observations/qpos"][:])
            action = torch.from_numpy(ep["/action"][:])

            velocity = None
            # if "/observations/qvel" in ep:
            #     velocity = torch.from_numpy(ep["/observations/qvel"][:])

            effort = None
            # if "/observations/effort" in ep:
            #     effort = torch.from_numpy(ep["/observations/effort"][:])

            imgs_per_cam = load_raw_images_per_camera(
                ep,
                [
                    "cam_high",
                    "cam_left_wrist",
                    "cam_right_wrist",
                ],
            )

        return imgs_per_cam, state, action, velocity, effort
    except Exception as e:
        print(f"Error loading episode {ep_path}: {e}")
        return None, None, None, None, None

def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        if imgs_per_cam is None and state is None and action is None and velocity is None and effort is None:
            logger.error(f"Error loading episode {ep_path}")
            continue
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)

        dataset.save_episode(task=task)

    return dataset


def port_aloha(
    out_dir: Path,
    raw_dirs: list[Path],
    repo_id: str,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    num_processes: int = 4
):

    logfilename = f"convert_{task}_{out_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logfilepath = str(log_dir / logfilename)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    rotatfile = logging.handlers.RotatingFileHandler(logfilepath, 'a', 20971520, 2, 'utf-8')
    rotatfile.setLevel(logging.INFO)

    logger.addHandler(console)
    logger.addHandler(rotatfile)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    rotatfile.setFormatter(formatter)
                        

    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)
    
    hdf5_files = []
    for raw_dir in raw_dirs:
        hdf5_files.extend(sorted(raw_dir.glob("episode_*.hdf5")))
    
    files_len = len(hdf5_files)
    logger.info('start to process')
    logger.info(f'Total hdf5 files: {files_len}')
    
    # 处理单个任务组的函数
    def process_one_group(start_idx: int, end_idx: int, group_idx: int):
        try:
            hdf5_files_group = hdf5_files[start_idx:end_idx]
            logger.info(f'Processing group {group_idx}: files {start_idx} to {end_idx-1}')
            logger.info(f'files : {hdf5_files_group}')
            save_dir = out_dir / f"group_{group_idx}"
            logger.info(f"Save directory: {save_dir}")
            
            dataset = create_empty_dataset(
                save_dir,
                repo_id,
                robot_type="mobile_aloha" if is_mobile else "aloha",
                mode=mode,
                has_effort=False,
                has_velocity=False,
                dataset_config=dataset_config,
            )

            dataset = populate_dataset(
                dataset,
                hdf5_files_group,
                task=task,
                episodes=episodes,
            )

            # dataset.consolidate()
            logger.info(f"Successfully processed group {group_idx}")
        except Exception as e:
            logger.error(f"Error processing group {group_idx}: {str(e)}")
    
    # 创建任务列表
    processes = []

    # files_len : 10
    # num_processes : 4
    # group_size : 3
    group_size = (files_len + num_processes - 1) // num_processes  # 向上取整确保处理所有文件
    
    for group_idx in range(1, num_processes + 1):
        start_idx = (group_idx - 1) * group_size
        end_idx = min(start_idx + group_size, files_len)
        
        if start_idx >= files_len:
            break
            
        # 创建并启动进程
        p = Process(target=process_one_group, args=(start_idx, end_idx, group_idx))
        processes.append(p)
        p.start()
        logger.info(f"Started process for group {group_idx} (PID: {p.pid})")
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    logger.info("All processes completed")


if __name__ == "__main__":
    tyro.cli(port_aloha)
