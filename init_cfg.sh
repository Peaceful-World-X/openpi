#!/usr/bin/env bash
# Shared configuration and container preparation. Safe to source interactively.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

load_config() {
  local key allexport_was_set=0 status=0
  local -A inherited=()
  # Explicit exported values (including main.py's job settings) override .env.
  while IFS= read -r key; do inherited["${key}"]="${!key}"; done < <(compgen -e)
  [[ $- == *a* ]] && allexport_was_set=1
  set -a
  # shellcheck disable=SC1090
  source "${OPENPI_ENV_FILE:-${ROOT_DIR}/.env}" || status=$?
  (( allexport_was_set )) || set +a
  for key in "${!inherited[@]}"; do export "${key}=${inherited[${key}]}"; done
  return "${status}"
}
load_config || return $? 2>/dev/null || exit $?

find_bootstrap_python() {
  local candidate
  if [[ -n "${OPENPI_BOOTSTRAP_PYTHON:-}" ]]; then
    [[ -x "${OPENPI_BOOTSTRAP_PYTHON}" ]] && { printf '%s\n' "${OPENPI_BOOTSTRAP_PYTHON}"; return 0; }
  fi
  for candidate in python3 python /usr/local/bin/python3 /opt/conda/bin/python /usr/bin/python3; do
    if [[ "${candidate}" == */* ]]; then
      [[ -x "${candidate}" ]] || continue
      printf '%s\n' "${candidate}"
    elif command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
    else
      continue
    fi
    return 0
  done
  return 1
}

BOOTSTRAP_PYTHON="$(find_bootstrap_python || true)"

# The interactive notebook and a submitted K8S task can mount the same NAS at
# different roots.  In particular, the NAS shown in the console contains a
# `workspace/` directory, so mounting the NAS root at /ossfs/workspace makes
# the visible files /ossfs/workspace/workspace/.data/... . Resolve only known
# NAS-backed inputs; /data and /result remain container-local paths.
resolve_nas_path() {
  local configured="$1" candidate relative root
  if [[ -e "${configured}" ]]; then
    printf '%s\n' "${configured}"
    return 0
  fi
  case "${configured}" in
    /ossfs/workspace/*) relative="${configured#/ossfs/workspace/}" ;;
    /input/*) relative="${configured#/input/}" ;;
    *) printf '%s\n' "${configured}"; return 0 ;;
  esac
  local -a roots=(
    "/ossfs/workspace"
    "/ossfs/workspace/workspace"
    "/input"
    "/input/workspace"
  )
  for root in "${roots[@]}"; do
    candidate="${root}/${relative}"
    if [[ -e "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    # Some NAS configurations mount the workspace directory itself rather
    # than the NAS root. Accept both /input/.data and /input/workspace/.data.
    if [[ "${relative}" == workspace/* ]]; then
      candidate="${root}/${relative#workspace/}"
      if [[ -e "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    fi
  done
  # Last resort for platform-specific NAS layouts. Restrict the search to
  # mount roots and use the basename, so a nested workflow directory is fine.
  local name="$(basename "${configured}")"
  for root in /input /ossfs /mnt; do
    [[ -d "${root}" ]] || continue
    candidate="$(find "${root}" -maxdepth 8 -type f -name "${name}" -print -quit 2>/dev/null || true)"
    if [[ -n "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' "${configured}"
}

DATA_ARCHIVE="${OPENPI_DATA_ARCHIVE:?Set OPENPI_DATA_ARCHIVE in .env}"
DATA_DIR="${OPENPI_DATA_DIR:?Set OPENPI_DATA_DIR in .env}"
ASSETS_DIR="${OPENPI_ASSETS_DIR:-${DATA_DIR}}"
PARAMS_DIR="${OPENPI_PARAMS_DIR:?}"
CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:?}"
PYTHON="${OPENPI_VENV_DIR:-/openpi-venv}/bin/python"
OSSUTIL="${OPENPI_OSSUTIL:-/ossutil64}"
OSS_ROOT="${OPENPI_OSS_ROOT:?}"
OSS_DATA_ROOT="${OPENPI_OSS_DATA_ROOT:-oss://tipalgog/sudu/.data}"
OSS_MODEL_ROOT="${OPENPI_OSS_MODEL_ROOT:-${OSS_DATA_ROOT}}"
OSS_DEPS_ROOT="${OPENPI_OSS_DEPS_ROOT:-${OSS_ROOT%/}/deps}"
OSS_ENDPOINT="${OPENPI_OSS_ENDPOINT:?}"
OSS_CONFIG="${OPENPI_OSS_CONFIG:-${HOME}/.config/openpi/ossutil.conf}"
UPLOAD_INTERVAL_SECONDS="${OPENPI_UPLOAD_INTERVAL_SECONDS:-300}"

DATA_ARCHIVE="$(resolve_nas_path "${DATA_ARCHIVE}")"
if [[ "${ASSETS_DIR}" != "${DATA_DIR}" ]]; then
  ASSETS_DIR="$(resolve_nas_path "${ASSETS_DIR}")"
fi
PARAMS_DIR="$(resolve_nas_path "${PARAMS_DIR}")"
if [[ -n "${OPENPI_PYTHON_ARCHIVE:-}" ]]; then
  OPENPI_PYTHON_ARCHIVE="$(resolve_nas_path "${OPENPI_PYTHON_ARCHIVE}")"
fi
if [[ -n "${OPENPI_TOKENIZER_PATH:-}" ]]; then
  OPENPI_TOKENIZER_PATH="$(resolve_nas_path "${OPENPI_TOKENIZER_PATH}")"
fi
if [[ -n "${OPENPI_OSSUTIL_SOURCE:-}" ]]; then
  OPENPI_OSSUTIL_SOURCE="$(resolve_nas_path "${OPENPI_OSSUTIL_SOURCE}")"
fi
echo "Resolved inputs: archive=${DATA_ARCHIVE}, params=${PARAMS_DIR}"

prepare_dataset() {
  [[ -n "${BOOTSTRAP_PYTHON}" ]] || { echo "找不到可用于解压数据集的系统 Python" >&2; return 1; }
  if [[ ! -f "${DATA_ARCHIVE}" ]]; then
    echo "Dataset archive is not visible at ${DATA_ARCHIVE}" >&2
    for root in /input /ossfs /mnt; do
      if [[ -d "${root}" ]]; then
        echo "Mount listing for ${root}:" >&2
        find "${root}" -maxdepth 3 -type d -print 2>/dev/null | head -80 >&2 || true
      fi
    done
  fi
  "${BOOTSTRAP_PYTHON}" - "${DATA_ARCHIVE}" "${DATA_DIR}" <<'PY'
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

import sys

archive, target = map(Path, sys.argv[1:])
target = target.resolve()
if target == Path('/ossfs') or Path('/ossfs') in target.parents:
    raise SystemExit(f'Dataset target must be on container-local storage, not /ossfs: {target}')

def validate(path):
    for name in ('meta/info.json', 'assets/norm_stats.json'):
        with (path / name).open() as stream:
            json.load(stream)
    if not (path / 'data').is_dir():
        raise ValueError(f'Missing data directory: {path}')

# A new image may already contain an empty /data directory.
if target.is_dir() and not any(target.iterdir()):
    target.rmdir()
if target.exists():
    validate(target)
    print(f'Reusing local dataset: {target}')
else:
    if not archive.is_file():
        raise SystemExit(f'Dataset archive not found: {archive}; set OPENPI_DATA_ARCHIVE')
    target.parent.mkdir(parents=True, exist_ok=True)
    # Extract to a sibling temporary directory. Interrupted extraction must
    # never be mistaken for a complete training dataset on the next run.
    with tempfile.TemporaryDirectory(prefix='.openpi-extract-', dir=target.parent) as temporary:
        staging = Path(temporary)
        print(f'Extracting {archive} to {target}', flush=True)
        with tarfile.open(archive, 'r|*') as stream:
            for member in stream:
                name = PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or not (member.isdir() or member.isfile()):
                    raise ValueError(f'Unsupported or unsafe archive entry: {member.name}')
                stream.extract(member, staging, set_attrs=False)
        # Accept either a top-level dataset folder or meta/data/assets at root.
        if (staging / 'meta/info.json').is_file():
            payload = staging
        else:
            candidates = [p for p in staging.iterdir() if p.is_dir() and (p / 'meta/info.json').is_file()]
            if len(candidates) != 1:
                raise ValueError('Archive must contain exactly one dataset with meta/info.json')
            payload = candidates[0]
        validate(payload)
        payload.rename(target)
    print(f'Local dataset ready: {target}')
PY
}

is_supported_ossutil() {
  [[ -x "$1" ]] && "$1" --version 2>&1 | grep -q "ossutil version: v1.7.18"
}

configure_oss() {
  local access_id access_secret config_tmp source_tool
  access_id="${OPENPI_OSS_ACCESS_KEY_ID:-${OSS_ACCESS_KEY_ID:-}}"
  access_secret="${OPENPI_OSS_ACCESS_KEY_SECRET:-${OSS_ACCESS_KEY_SECRET:-}}"
  : "${access_id:?Set OPENPI_OSS_ACCESS_KEY_ID in .env or the environment}"
  : "${access_secret:?Set OPENPI_OSS_ACCESS_KEY_SECRET in .env or the environment}"
  # An explicitly supplied, working local executable can be reused.
  if ! is_supported_ossutil "${OSSUTIL}"; then
    source_tool="${OPENPI_OSSUTIL_SOURCE:-}"
    if [[ -z "${source_tool}" || ! -f "${source_tool}" ]]; then
      source_tool="${ROOT_DIR}/ossutil64"
    fi
    local tool_tmp
    [[ -f "${source_tool}" ]] || { echo "OSS tool not found: ${source_tool}" >&2; return 1; }
    mkdir -p "$(dirname "${OSSUTIL}")" || return $?
    tool_tmp="$(mktemp "${OSSUTIL}.XXXXXX")" || return $?
    # Unique temporary file + rename avoids overwriting an executing binary.
    if ! cp "${source_tool}" "${tool_tmp}" || ! chmod 755 "${tool_tmp}" || ! is_supported_ossutil "${tool_tmp}"; then
      rm -f "${tool_tmp}"
      echo "Cannot prepare ossutil v1.7.18 from ${source_tool}" >&2
      return 1
    fi
    if ! mv "${tool_tmp}" "${OSSUTIL}"; then
      rm -f "${tool_tmp}"
      return 1
    fi
  fi
  echo "Using OSS tool: ${OSSUTIL}"
  mkdir -p "$(dirname "${OSS_CONFIG}")" || return $?
  config_tmp="$(mktemp "${OSS_CONFIG}.XXXXXX")" || return $?
  if ! chmod 600 "${config_tmp}"; then
    rm -f "${config_tmp}"
    return 1
  fi
  if ! cat >"${config_tmp}" <<EOF
[Credentials]
language=EN
endpoint=${OSS_ENDPOINT}
accessKeyID=${access_id}
accessKeySecret=${access_secret}
EOF
  then
    rm -f "${config_tmp}"
    return 1
  fi
  if ! mv "${config_tmp}" "${OSS_CONFIG}"; then
    rm -f "${config_tmp}"
    return 1
  fi
  echo "OSS configured: ${OSS_CONFIG}; upload root: ${OSS_ROOT}"
}

# Equivalent to ossutil --retry-times 20 -e ENDPOINT -i ID -k SECRET ...,
# with credentials read from a private config file instead of CLI arguments.
# Example after initialization:
#   source ./init_cfg.sh
#   ossutil_upload cp -r /local/directory/ oss://tipalgog/sudu/your-path/ -j 4 --update
ossutil_upload() {
  "${OSSUTIL}" --retry-times 20 -e "${OSS_ENDPOINT}" -c "${OSS_CONFIG}" "$@"
}

download_oss_file_if_missing() {
  local target="$1" uri="$2"
  [[ -f "${target}" ]] && return 0
  echo "Downloading ${uri} to ${target}"
  mkdir -p "$(dirname "${target}")" || return $?
  ossutil_upload cp "${uri}" "${target}" --update || {
    echo "OSS download failed: ${uri}" >&2
    return 1
  }
}

download_oss_dir_if_missing() {
  local target="$1" uri="$2"
  if [[ -d "${target}" ]] && find "${target}" -type f -print -quit 2>/dev/null | grep -q .; then
    return 0
  fi
  echo "Downloading ${uri} to ${target}"
  mkdir -p "${target}" || return $?
  ossutil_upload cp -r "${uri%/}/" "${target%/}/" --update || {
    echo "OSS directory download failed: ${uri}" >&2
    return 1
  }
}

download_dataset_archive() {
  local archive_uri="${OPENPI_DATA_ARCHIVE_URI:-${OSS_DATA_ROOT%/}/$(basename "${DATA_ARCHIVE}")}"
  download_oss_file_if_missing "${DATA_ARCHIVE}" "${archive_uri}"
  if [[ -n "${OPENPI_PYTHON_ARCHIVE:-}" ]]; then
    local python_uri="${OPENPI_PYTHON_ARCHIVE_URI:-${OSS_DATA_ROOT%/}/$(basename "${OPENPI_PYTHON_ARCHIVE}")}"
    download_oss_file_if_missing "${OPENPI_PYTHON_ARCHIVE}" "${python_uri}"
  fi
  if [[ -n "${OPENPI_TOKENIZER_PATH:-}" ]]; then
    local tokenizer_uri="${OPENPI_TOKENIZER_URI:-${OSS_MODEL_ROOT%/}/$(basename "${OPENPI_TOKENIZER_PATH}")}"
    download_oss_file_if_missing "${OPENPI_TOKENIZER_PATH}" "${tokenizer_uri}"
  fi
  download_oss_dir_if_missing "${PARAMS_DIR}" "${OPENPI_PARAMS_URI:-${OSS_MODEL_ROOT%/}/openpi_base/params}"
}

download_local_project_dependencies() {
  # pyproject.toml deliberately uses local sources. Download them beside the
  # project before init_env.sh runs uv lock/sync, so no GitHub checkout is
  # attempted inside the training worker.
  download_oss_dir_if_missing "${ROOT_DIR}/lerobot" "${OPENPI_LEROBOT_URI:-${OSS_DEPS_ROOT%/}/lerobot}"
  download_oss_dir_if_missing "${ROOT_DIR}/dlimp" "${OPENPI_DLIMP_URI:-${OSS_DEPS_ROOT%/}/dlimp}"
}

init_cfg_main() {
  if [[ "${OPENPI_ENV_ONLY:-0}" == "1" ]]; then
    bash "${ROOT_DIR}/init_env.sh"
    return
  fi
  configure_oss || return $?
  download_dataset_archive || return $?
  download_local_project_dependencies || return $?
  prepare_dataset || return $?
  bash "${ROOT_DIR}/init_env.sh" || return $?
  [[ "${ASSETS_DIR}" == "${DATA_DIR}" || -f "${ASSETS_DIR}/assets/norm_stats.json" ]] || { echo "Missing norm stats: ${ASSETS_DIR}/assets/norm_stats.json" >&2; return 1; }
  [[ -d "${PARAMS_DIR}" ]] || { echo "Base checkpoint not found: ${PARAMS_DIR}" >&2; return 1; }
  mkdir -p "${CHECKPOINT_DIR}" || return $?
  echo "Ready: dataset=${DATA_DIR}, python=${PYTHON}, checkpoints=${CHECKPOINT_DIR}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  init_cfg_main
fi
