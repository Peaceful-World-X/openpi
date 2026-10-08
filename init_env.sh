#!/usr/bin/env bash
# Install and verify the training environment; all settings come from .env.
set -euo pipefail
# shellcheck source=init_cfg.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/init_cfg.sh"

PYTHON_INSTALL_DIR="${OPENPI_PYTHON_INSTALL_DIR:-/python}"
PYTHON_ARCHIVE="${OPENPI_PYTHON_ARCHIVE:?}"
VENV_DIR="${PYTHON%/bin/python}"
export UV_CACHE_DIR="${OPENPI_UV_CACHE_DIR:-/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${PYTHON_INSTALL_DIR}"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export UV_PYTHON_DOWNLOADS=never
export DEBIAN_FRONTEND=noninteractive
export GIT_LFS_SKIP_SMUDGE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PATH="${ROOT_DIR}:/usr/local/bin:${HOME}/.local/bin:${PATH}"
unset VIRTUAL_ENV

# The internal package mirror uses a private CA that is not present in some
# fresh training images. Keep certificate verification for every other host,
# but allow this explicitly configured internal mirror to bootstrap uv.
PIP_NETWORK_ARGS=()
UV_NETWORK_ARGS=()
if [[ "${UV_DEFAULT_INDEX}" == https://artifacts.antgroup-inc.cn/* || "${UV_DEFAULT_INDEX}" == https://artifacts.antgroup-inc.cn ]]; then
  PIP_NETWORK_ARGS+=(--trusted-host artifacts.antgroup-inc.cn)
  UV_NETWORK_ARGS+=(--allow-insecure-host artifacts.antgroup-inc.cn)
  export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-artifacts.antgroup-inc.cn}"
  export UV_INSECURE_HOST="${UV_INSECURE_HOST:-artifacts.antgroup-inc.cn}"
fi

die() { echo "init_env.sh: $*" >&2; exit 1; }

if [[ -n "${OPENPI_CA_BUNDLE:-}" ]]; then
  [[ -f "${OPENPI_CA_BUNDLE}" ]] || die "OPENPI_CA_BUNDLE 不存在: ${OPENPI_CA_BUNDLE}"
  export SSL_CERT_FILE="${OPENPI_CA_BUNDLE}"
fi

have() {
  command -v "$1" >/dev/null 2>&1
}

if [[ "${EUID}" -ne 0 ]]; then
  case "${VENV_DIR}:${PYTHON_INSTALL_DIR}:${UV_CACHE_DIR}" in
    /openpi-venv:*|*:/python:*|*:*:/uv-cache)
      die "默认安装目录位于根目录，必须用 root 执行；非 root 测试请同时设置 OPENPI_VENV_DIR、OPENPI_PYTHON_INSTALL_DIR 和 OPENPI_UV_CACHE_DIR 到可写目录。"
      ;;
  esac
fi

echo "[1/5] Installing OS prerequisites"
if have apt-get && [[ "${EUID}" -eq 0 ]]; then
  apt-get update
  apt-get install -y --no-install-recommends \
    ca-certificates \
    clang \
    curl \
    git \
    git-lfs \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libva-drm2 \
    libva-x11-2
  rm -rf /var/lib/apt/lists/*
else
  echo "Skipping apt packages (not root or apt-get is unavailable)"
fi

BOOTSTRAP_PYTHON="$(find_bootstrap_python || true)"
[[ -n "${BOOTSTRAP_PYTHON}" ]] || die "找不到可用于安装环境的系统 Python"

python_version_ok() {
  "$1" - <<'PY'
# Check the stdlib as well: an interrupted extraction can leave a runnable
# interpreter with missing libraries. Such an installation must be extracted again.
import bz2
import ctypes
import lzma
import sqlite3
import ssl
import sys
import venv
# gym-aloha pins MuJoCo 2.3.7, whose Linux wheels do not cover Python 3.12.
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY
}

echo "[2/5] Selecting Python 3.11 (required by the locked training dependencies)"
if [[ -n "${OPENPI_BASE_PYTHON:-}" ]]; then
  PYTHON_SPEC="${OPENPI_BASE_PYTHON}"
else
  # The standalone archive contains a top-level python/ directory. Keep its
  # libraries and interpreter together on container-local storage.
  PYTHON_SPEC="${PYTHON_INSTALL_DIR}/python/bin/python3.11"
  if [[ -x "${PYTHON_SPEC}" ]] && python_version_ok "${PYTHON_SPEC}"; then
    echo "Reusing container-local Python: ${PYTHON_SPEC}"
  else
    [[ -r "${PYTHON_ARCHIVE}" ]] || die "Python 安装包不存在或不可读: ${PYTHON_ARCHIVE}；请上传安装包或设置 OPENPI_PYTHON_ARCHIVE。"
    echo "Extracting ${PYTHON_ARCHIVE} to ${PYTHON_INSTALL_DIR}"
    mkdir -p "${PYTHON_INSTALL_DIR}"
    tar -tzf "${PYTHON_ARCHIVE}" >/dev/null \
      || die "Python 安装包损坏或格式不正确: ${PYTHON_ARCHIVE}"
    tar -xzf "${PYTHON_ARCHIVE}" -C "${PYTHON_INSTALL_DIR}" --no-same-owner \
      || die "Python 安装包解压失败: ${PYTHON_ARCHIVE}"
  fi
fi
[[ -x "${PYTHON_SPEC}" ]] && python_version_ok "${PYTHON_SPEC}" \
  || die "解压后未找到可用的 Python 3.11: ${PYTHON_SPEC}"
"${PYTHON_SPEC}" - <<'PY'
import bz2
import ctypes
import lzma
import sqlite3
import ssl
import sys
import venv

print(f"python_spec={sys.executable} ({sys.version.split()[0]})")
PY

echo "[3/5] Installing uv"
if ! have uv; then
  PIP_INSTALL_ARGS=(--no-cache-dir --index-url "${UV_DEFAULT_INDEX}" "${PIP_NETWORK_ARGS[@]}" uv)
  if [[ "${EUID}" -eq 0 ]]; then
    PIP_INSTALL_ARGS=(--break-system-packages "${PIP_INSTALL_ARGS[@]}")
  else
    PIP_INSTALL_ARGS=(--user "${PIP_INSTALL_ARGS[@]}")
  fi
  "${BOOTSTRAP_PYTHON}" -m pip install "${PIP_INSTALL_ARGS[@]}" || die "无法从 ${UV_DEFAULT_INDEX} 安装 uv"
fi
have uv || die "uv 安装失败"
UV_COMMAND="$(command -v uv)"
echo "uv=${UV_COMMAND} ($("${UV_COMMAND}" --version))"

mkdir -p "${PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}"

echo "[4/5] Refreshing dependency sources and installing OpenPI"
cd "${ROOT_DIR}"
# Refresh wheel URLs/local sources before touching the existing venv.
# No --upgrade: prefer locked versions; all local source directories must exist.
uv lock --python "${PYTHON_SPEC}" \
  --default-index "${UV_DEFAULT_INDEX}" --system-certs \
  "${UV_NETWORK_ARGS[@]}"

if [[ "${OPENPI_RECREATE_ENV:-0}" == "1" ]] || {
  [[ -x "${VENV_DIR}/bin/python" ]] && ! python_version_ok "${VENV_DIR}/bin/python";
}; then
  echo "Rebuilding environment: ${VENV_DIR}"
  rm -rf -- "${VENV_DIR}"
fi

SYNC_ARGS=(
  sync
  --locked
  --python "${PYTHON_SPEC}"
  --default-index "${UV_DEFAULT_INDEX}"
  --system-certs
  --no-build-package mujoco
  # LeRobot declares pynput for interactive robot controls. OpenPI training
  # only reads LeRobot datasets, so skip pynput and its native evdev input
  # backend; evdev has no wheel in this image and would require clang/root.
  --no-install-package pynput
  --no-install-package evdev
)
SYNC_ARGS+=("${UV_NETWORK_ARGS[@]}")
if [[ "${OPENPI_INSTALL_DEV:-0}" != "1" ]]; then
  SYNC_ARGS+=(--no-dev)
fi
uv "${SYNC_ARGS[@]}"

PYTHON="${VENV_DIR}/bin/python"
[[ -x "${PYTHON}" ]] || die "虚拟环境创建失败: ${PYTHON}"

prepare_torchcodec_ffmpeg() {
  local site_packages source_dir runtime_dir env_file
  site_packages="$(${PYTHON} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  source_dir=""
  for candidate in "${site_packages}/opencv_python_headless.libs" "${site_packages}/opencv_python.libs"; do
    if [[ -d "${candidate}" ]]; then
      source_dir="${candidate}"
      break
    fi
  done

  # System FFmpeg is preferred when present. The OpenCV wheel fallback is
  # needed in non-root images where apt packages cannot be installed.
  local system_ffmpeg=0
  for candidate in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu; do
    if compgen -G "${candidate}/libavutil.so.5[6-9]" >/dev/null; then
      system_ffmpeg=1
      break
    fi
  done
  if (( system_ffmpeg )); then
    echo "FFmpeg runtime found in system library paths"
    return 0
  fi
  [[ -n "${source_dir}" ]] || {
    echo "FFmpeg runtime not found; torchcodec verification may fail" >&2
    return 0
  }

  runtime_dir="${OPENPI_FFMPEG_RUNTIME_DIR:-${VENV_DIR}/lib/openpi-ffmpeg}"
  mkdir -p "${runtime_dir}" || die "无法创建 FFmpeg 运行库目录: ${runtime_dir}"
  "${BOOTSTRAP_PYTHON}" - "${source_dir}" "${runtime_dir}" <<'PY'
import re
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
supported = {"avutil", "avcodec", "avformat", "avfilter", "swscale", "swresample", "postproc", "avdevice"}
for library in source.iterdir():
    match = re.match(r"lib([a-z0-9]+)-[^/]+\.so\.([0-9]+)", library.name)
    if not match or match.group(1) not in supported:
        continue
    link = target / f"lib{match.group(1)}.so.{match.group(2)}"
    link.unlink(missing_ok=True)
    link.symlink_to(library.resolve())
PY
  export LD_LIBRARY_PATH="${runtime_dir}:${source_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  env_file="${VENV_DIR}/openpi-runtime.env"
  printf 'export LD_LIBRARY_PATH=%q\n' "${LD_LIBRARY_PATH}" >"${env_file}"
  echo "FFmpeg runtime prepared from ${source_dir}"
}

prepare_torchcodec_ffmpeg

echo "[5/5] Verifying the installed environment"
"${PYTHON}" - "${OPENPI_TOKENIZER_PATH:?}" <<'PY'
import hashlib
import os
from pathlib import Path
import sys
import tempfile

import sentencepiece as spm
from openpi.shared import download

# Seed the standard OpenPI cache so training never needs to fetch this file
# from Google Storage. Respect the same OPENPI_DATA_HOME as training.
source = Path(sys.argv[1])
if not source.is_file():
    raise SystemExit(f"Tokenizer not found: {source}; upload it or set OPENPI_TOKENIZER_PATH")
data = source.read_bytes()
expected_sha256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
if hashlib.sha256(data).hexdigest() != expected_sha256:
    raise SystemExit(f"Tokenizer checksum mismatch: {source}; upload the complete official file again")
tokenizer = spm.SentencePieceProcessor(model_proto=data)
target = download.get_cache_dir() / "big_vision/paligemma_tokenizer.model"
target.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".tokenizer-", delete=False) as stream:
    temporary = Path(stream.name)
    try:
        stream.write(data)
        stream.flush()
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
print(f"tokenizer={target} (sha256 verified, vocab_size={tokenizer.vocab_size()})")

required = ("jax", "flax", "torch", "lerobot", "openpi")
print(f"python={sys.executable}")
print(f"version={sys.version.split()[0]}")
print(f"prefix={sys.prefix}")
if sys.version_info[:2] != (3, 11):
    raise SystemExit("This training environment requires Python 3.11 (MuJoCo 2.3.7 wheels)")
for name in required:
    module = __import__(name)
    print(f"{name}={getattr(module, '__version__', 'ok')}")

# OpenPI's pinned LeRobot 0.1.0 dataset reader uses PyAV for video decoding.
# TorchCodec is an optional LeRobot dependency and requires libavdevice, which
# is absent from some non-root images; importing it here would reject an
# otherwise usable training environment.
import av
from lerobot.common.datasets import video_utils
print(f"av={av.__version__}")
print(f"lerobot video backend={video_utils.__name__}")

import jax

print("jax_devices=" + ",".join(str(device) for device in jax.devices()))
import flax.nnx

print("flax.nnx import ok")
if os.environ.get("OPENPI_RUN_SMOKE") == "1":
    from openpi.training import config

    name = os.environ["OPENPI_TRAIN_CONFIG"]
    config.get_config(name)
    print(f"training_config={name}")
PY

echo "OpenPI environment ready: python=${PYTHON}, cache=${UV_CACHE_DIR}"
