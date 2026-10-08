#!/usr/bin/env bash
# Upload all assets required by OpenPI training; jobs download missing assets.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../init_cfg.sh
source "${ROOT_DIR}/init_cfg.sh"

configure_oss
remote_root="${OPENPI_OSS_DATA_ROOT:-oss://tipalgog/sudu/.data}"
deps_root="${OPENPI_OSS_DEPS_ROOT:-${OPENPI_OSS_ROOT%/}/deps}"

upload_tree() {
  local local_dir="$1" remote_dir="$2" entry remote_uri
  [[ -d "${local_dir}" ]] || { echo "Directory not found: ${local_dir}" >&2; exit 1; }
  shopt -s nullglob dotglob
  local entries=("${local_dir}"/*)
  ((${#entries[@]} > 0)) || { echo "No files in ${local_dir}" >&2; exit 1; }
  for entry in "${entries[@]}"; do
    remote_uri="${remote_dir%/}/$(basename "${entry}")"
    echo "Uploading ${entry} to ${remote_uri}"
    if [[ -d "${entry}" ]]; then
      ossutil_upload cp -r "${entry}/" "${remote_uri}/" --update
    else
      ossutil_upload cp "${entry}" "${remote_uri}" --update
    fi
  done
}

upload_project_dependencies() {
  local name source target
  for name in lerobot dlimp; do
    source="${ROOT_DIR}/${name}"
    target="${deps_root%/}/${name}/"
    [[ -d "${source}" ]] || { echo "Project dependency not found: ${source}" >&2; exit 1; }
    echo "Uploading ${source} to ${target}"
    ossutil_upload cp -r "${source%/}/" "${target}" --update
  done
}

if [[ "${1:-}" == "--all" || $# -eq 0 ]]; then
  data_dir="${OPENPI_LOCAL_DATA_DIR:-$(dirname "${OPENPI_OSSUTIL_SOURCE}")}"
  upload_tree "${data_dir}" "${remote_root}"
  upload_project_dependencies
  echo "Uploaded all assets from .data"
elif [[ "${1:-}" == "--data" ]]; then
  data_dir="${OPENPI_LOCAL_DATA_DIR:-$(dirname "${OPENPI_OSSUTIL_SOURCE}")}"
  upload_tree "${data_dir}" "${remote_root}"
elif [[ "${1:-}" == "--code-deps" ]]; then
  upload_project_dependencies
elif [[ "${1:-}" == "--models" ]]; then
  echo "--models is deprecated; openpi_base and tokenizer are already under .data" >&2
  data_dir="${OPENPI_LOCAL_DATA_DIR:-$(dirname "${OPENPI_OSSUTIL_SOURCE}")}"
  upload_tree "${data_dir}" "${remote_root}"
else
  archive="$1"
  [[ -f "${archive}" ]] || { echo "File not found: ${archive}" >&2; exit 1; }
  remote_uri="${remote_root%/}/$(basename "${archive}")"
  echo "Uploading ${archive} to ${remote_uri}"
  ossutil_upload cp "${archive}" "${remote_uri}" --update
fi
echo "Upload complete"
echo "OSS data root: ${remote_root}"
echo "OSS dependency root: ${deps_root}"
