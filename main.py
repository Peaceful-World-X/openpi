#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import os
from pathlib import Path
import shlex
import shutil
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

# Resolve the project from this file.  The local submitter lives under
# /16T-2/... while the worker source is mounted under /ossfs/workspace.
# Keeping this configurable also makes the submission testable from a clone.
ROOT_DIR = Path(os.environ.get("OPENPI_ROOT_DIR", Path(__file__).resolve().parent)).resolve()
SUBMIT_ROOT = Path("/tmp/openpi_submit")
IMAGE = "reg.docker.alibaba-inc.com/aii/aistudio:12550155-20250530161247"
NAS_MOUNT_PATH = "/input"
NAS_WORKSPACE_PATH = f"{NAS_MOUNT_PATH}/workspace"


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


def _worker_nas_path(value: str) -> str:
    """Map the interactive container's NAS path to the submitted task mount."""
    return value.replace("/ossfs/workspace/", f"{NAS_WORKSPACE_PATH}/")


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
        if config_name.removesuffix("_ttrtc").endswith(f"_80G_{count}gpu_full") and profile.gpu_num != count:
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

    # Follow the AIStudio tutorial: source_root is the job working directory
    # and command must be relative to it.  /ossfs/workspace is a separate NAS
    # mount, so an absolute command there can hide the submitted source tree.
    # Keep large local environments, datasets and checkpoints out of the
    # submission package; those are supplied by NAS or prepared by init_cfg.
    SUBMIT_ROOT.mkdir(parents=True, exist_ok=True)
    for entry in SUBMIT_ROOT.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    files = (".env", "README.md", "k8s_job.sh", "init_cfg.sh", "init_env.sh", "pyproject.toml", "uv.lock")
    # LeRobot and dlimp are downloaded from OSS during worker initialization;
    # keep them out of the SDK source package because they are large.
    directories = ("src", "scripts", "packages")
    for name in files:
        source = ROOT_DIR / name
        if source.is_file():
            destination = SUBMIT_ROOT / name
            if name == ".env":
                # The source container exposes NAS at /ossfs, while the SDK
                # task uses /input to keep NAS separate from source_root.
                text = source.read_text(encoding="utf-8")
                destination.write_text(_worker_nas_path(text), encoding="utf-8")
                shutil.copystat(source, destination)
            else:
                shutil.copy2(source, destination)
    for name in directories:
        source = ROOT_DIR / name
        if source.is_dir():
            shutil.copytree(
                source,
                SUBMIT_ROOT / name,
                symlinks=True,
                ignore=shutil.ignore_patterns(".git", "__pycache__", ".cache", "*.pyc"),
            )
    # The worker may need ossutil before its NAS is available. Include only
    # this small bootstrap binary; never include the dataset archive itself.
    ossutil_source = os.environ.get("OPENPI_OSSUTIL_SOURCE", "")
    if ossutil_source and Path(ossutil_source).is_file():
        destination = SUBMIT_ROOT / "ossutil64"
        shutil.copy2(ossutil_source, destination)
        destination.chmod(0o755)
    # uv is needed before the worker environment exists. Carry the local
    # standalone binary when available so a fresh worker does not depend on
    # the internal PyPI mirror's TLS certificate during bootstrap. init_env.sh
    # prepends the submitted source root to PATH and keeps pip as a fallback.
    uv_source = os.environ.get("OPENPI_UV_SOURCE") or shutil.which("uv")
    if uv_source and Path(uv_source).is_file():
        destination = SUBMIT_ROOT / "uv"
        if Path(uv_source).resolve() != destination.resolve():
            shutil.copy2(uv_source, destination)
        destination.chmod(0o755)
    missing = [name for name in files if not (SUBMIT_ROOT / name).is_file()]
    missing += [name for name in directories if not (SUBMIT_ROOT / name).is_dir()]
    if missing:
        raise RuntimeError(f"Submission package is incomplete; missing: {', '.join(missing)}")
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
            envs.setdefault(key, _worker_nas_path(value) if key.startswith("OPENPI_") else value)

    job = PythonJobBuilder(
        source_root=str(SUBMIT_ROOT),
        command="bash k8s_job.sh",
        main_file="",
        km_conf=km_conf,
        master=master,
        k8s_app_name=args.resource_group,
        k8s_priority="high",
        # Keep NAS separate from the SDK source workspace. The tutorial uses
        # /input for this reason; the NAS root contains its workspace/ folder.
        nas_storage=NasStoreConf(mount_path=NAS_MOUNT_PATH),
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
