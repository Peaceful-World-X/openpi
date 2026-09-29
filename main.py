#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import os
from pathlib import Path
import shlex
import sys

# Log SDK imports and submission errors too; no training environment is needed.
if __name__ == "__main__" and os.environ.get("_OPENPI_LOG_SUBMIT") != "1":
    os.execv(
        sys.executable,
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "scripts/run_logged.py"),
            "submit",
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
    )

from pypai.conf import ExecConf
from pypai.conf import GpuType
from pypai.conf import KMConf
from pypai.conf import NasStoreConf
from pypai.job import PythonJobBuilder

ROOT_DIR = Path("/ossfs/workspace/openpi")
SUBMIT_ROOT = Path("/tmp/openpi_submit")
IMAGE = "reg.docker.alibaba-inc.com/aii/aistudio:12550155-20250530161247"


@dataclass(frozen=True)
class TrainingProfile:
    gpu_type: GpuType | str
    config_name: str
    gpu_num: int = 4


PROFILES = {
    "h20-8gpu": TrainingProfile(
        gpu_type=getattr(GpuType, "H203e", "h20-3e"),
        config_name="pi05_bund_20260917_n906_v21_new_80G_8gpu_full",
        gpu_num=8,
    ),
    "h20-4gpu": TrainingProfile(
        gpu_type=getattr(GpuType, "H203e", "h20-3e"),
        config_name="pi05_bund_20260917_n906_v21_new_80G_4gpu_full",
        gpu_num=4,
    ),
    "h20-2gpu": TrainingProfile(
        gpu_type=getattr(GpuType, "H203e", "h20-3e"),
        config_name="pi05_bund_20260917_n906_v21_new_80G_2gpu_full",
        gpu_num=2,
    ),
    "h20": TrainingProfile(
        # aii-pypai <=0.1.41 does not expose H203e in GpuType, but ExecConf accepts its wire value.
        gpu_type=getattr(GpuType, "H203e", "h20-3e"),
        # Backward-compatible alias for the new 4-GPU full-fine-tuning profile.
        config_name="pi05_bund_20260917_n906_v21_new_80G_4gpu_full",
    ),
}
RESOURCE_GROUPS = ("digitaltechnology", "zolozasiagputrain")


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without adding a submit-time dependency."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        values = shlex.split(value, comments=True, posix=True)
        if key and len(values) == 1:
            os.environ.setdefault(key, values[0])


def _require_environment_value(name: str, *aliases: str) -> None:
    if not any(os.environ.get(key) for key in (name, *aliases)):
        accepted_names = ", ".join((name, *aliases))
        raise RuntimeError(f"Missing required environment variable (accepted names: {accepted_names})")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit OpenPI pi0.5 training to AIStudio")
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default=None,
        help="Optional preset override; by default use GPU_TYPE, GPU_NUM and TRAIN_CONFIG from .env",
    )
    parser.add_argument(
        "--resource-group",
        choices=RESOURCE_GROUPS,
        default=os.environ.get("OPENPI_RESOURCE_GROUP", "digitaltechnology"),
    )
    parser.add_argument("--exp-name", default=os.environ.get("OPENPI_EXP_NAME"))
    parser.add_argument("--project-name", default=os.environ.get("OPENPI_PROJECT_NAME"))
    parser.add_argument("--config-name", help="Override the training config selected by .env / --profile")
    return parser.parse_args()


def submit_job(args: argparse.Namespace) -> None:
    if args.profile:
        profile = PROFILES[args.profile]
    else:
        for key in ("OPENPI_GPU_TYPE", "OPENPI_GPU_NUM", "OPENPI_TRAIN_CONFIG"):
            _require_environment_value(key)
        gpu_type = os.environ["OPENPI_GPU_TYPE"].lower()
        profile = TrainingProfile(
            gpu_type=getattr(GpuType, "H203e" if gpu_type == "h20-3e" else gpu_type.upper(), gpu_type),
            gpu_num=int(os.environ["OPENPI_GPU_NUM"]),
            config_name=os.environ["OPENPI_TRAIN_CONFIG"],
        )
    if profile.gpu_num < 1:
        raise ValueError("OPENPI_GPU_NUM must be a positive integer")
    config_name = getattr(args, "config_name", None) or profile.config_name
    for count in (2, 4, 8):
        if config_name.endswith(f"_80G_{count}gpu_full") and profile.gpu_num != count:
            raise ValueError(f"Config {config_name} requires {count} GPUs, requested {profile.gpu_num}")
    gpu_name = getattr(profile.gpu_type, "value", profile.gpu_type)
    if os.environ.get("OPENPI_RESUME") == "1" and not args.exp_name:
        raise ValueError("Set OPENPI_EXP_NAME or --exp-name to the existing experiment when OPENPI_RESUME=1")
    timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"pi05_{gpu_name}_{profile.gpu_num}cards_{timestamp}"
    project_name = args.project_name or config_name

    # Validate credentials before reserving GPUs. The worker reads their values from the NAS-mounted .env file.
    _require_environment_value("OPENPI_OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_ID")
    _require_environment_value("OPENPI_OSS_ACCESS_KEY_SECRET", "OSS_ACCESS_KEY_SECRET")

    master = ExecConf(
        cpu=96,
        memory=1024 * 800,
        gpu_num=profile.gpu_num,
        num=1,
        gpu_type=profile.gpu_type,
        disk_m=1024 * 500,
    )
    km_conf = KMConf(image=IMAGE)

    # The submit package is intentionally tiny so pypai does not scan openpi/.venv.
    SUBMIT_ROOT.mkdir(parents=True, exist_ok=True)
    (SUBMIT_ROOT / "README.txt").write_text("OpenPI K8S submit files\n", encoding="utf-8")

    envs = {
        "OPENPI_TRAIN_CONFIG": config_name,
        "OPENPI_EXP_NAME": exp_name,
        "OPENPI_PROJECT_NAME": project_name,
        "OPENPI_GPU_TYPE": gpu_name,
        "OPENPI_GPU_NUM": str(profile.gpu_num),
        "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(profile.gpu_num)),
    }

    # Carry .env / submit-shell settings without overriding the selected profile.
    # Workers read credentials from the NAS .env, not job arguments.
    excluded = {"OPENPI_OSS_ACCESS_KEY_ID", "OPENPI_OSS_ACCESS_KEY_SECRET", "OPENPI_ENV_FILE"}
    for key, value in os.environ.items():
        if (key.startswith("OPENPI_") and key not in excluded) or key in {
            "UV_DEFAULT_INDEX",
            "UV_SYSTEM_CERTS",
            "XLA_FLAGS",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "TOKENIZERS_PARALLELISM",
            "PYTHONUNBUFFERED",
            "WANDB_MODE",
        }:
            envs.setdefault(key, value)

    job = PythonJobBuilder(
        source_root=str(SUBMIT_ROOT),
        command=f"bash {ROOT_DIR}/k8s_job.sh",
        main_file="",
        km_conf=km_conf,
        master=master,
        k8s_app_name=args.resource_group,
        k8s_priority="high",
        nas_storage=NasStoreConf(mount_path="/ossfs/workspace"),
        tag=f"type=train,model=pi05,gpu={gpu_name}",
        platform="kubemaker",
        envs=envs,
    )

    record_id = job.run(enable_wait=False)
    print(
        f"submitted, record_id={record_id}, gpu={gpu_name}x{profile.gpu_num}, "
        f"resource_group={args.resource_group}, config={config_name}, exp_name={exp_name}"
    )


if __name__ == "__main__":
    _load_dotenv(Path(os.environ.get("OPENPI_ENV_FILE", ROOT_DIR / ".env")))
    submit_job(_parse_args())
