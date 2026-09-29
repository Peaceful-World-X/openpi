#!/usr/bin/env bash
# CPU-container entry point: prepare the submission SDK, then request GPUs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${_OPENPI_LOG_SUBMIT:-0}" != 1 ]]; then
  exec /usr/bin/python3 "${ROOT_DIR}/scripts/run_logged.py" submit bash "${ROOT_DIR}/submit.sh" "$@"
fi

BASE_PYTHON="${OPENPI_SUBMIT_PYTHON:-/usr/bin/python3}"
SUBMIT_ENV="${OPENPI_SUBMIT_ENV:-/openpi-submit-venv}"
SUBMIT_INDEX="${OPENPI_SUBMIT_INDEX:-https://pypi.antfin-inc.com/simple/}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash submit.sh [--check-only | main.py arguments]"
  echo "Missing/broken pypai is installed into ${SUBMIT_ENV}; --check-only never submits a job."
  exit 0
fi
check_only=0
if [[ "${1:-}" == "--check-only" ]]; then
  check_only=1
  shift
  [[ "$#" -eq 0 ]] || { echo "--check-only does not accept submission arguments" >&2; exit 1; }
fi

[[ -x "${BASE_PYTHON}" ]] || { echo "Submission Python not found: ${BASE_PYTHON}" >&2; exit 1; }
"${BASE_PYTHON}" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit('Submission requires Python >=3.11; set OPENPI_SUBMIT_PYTHON to a suitable interpreter')
PY

sdk_ready() {
  "$1" - <<'PY'
from pypai.conf import ExecConf, GpuType, KMConf, NasStoreConf
from pypai.job import PythonJobBuilder
PY
}

if sdk_ready "${BASE_PYTHON}"; then
  SUBMIT_PYTHON="${BASE_PYTHON}"
else
  echo "Submission SDK missing or unusable; preparing isolated environment: ${SUBMIT_ENV}"
  SUBMIT_PYTHON="${SUBMIT_ENV}/bin/python"
  if [[ ! -x "${SUBMIT_PYTHON}" ]]; then
    # Ubuntu minimal images often omit ensurepip/python3-venv.
    if ! "${BASE_PYTHON}" -m venv "${SUBMIT_ENV}"; then
      if [[ "${EUID}" -ne 0 ]] || ! command -v apt-get >/dev/null 2>&1; then
        echo "Cannot create ${SUBMIT_ENV}; install the matching python3-venv package and use a writable directory." >&2
        exit 1
      fi
      python_minor="$("${BASE_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "python${python_minor}-venv"
      "${BASE_PYTHON}" -m venv "${SUBMIT_ENV}"
    fi
  fi
  if ! sdk_ready "${SUBMIT_PYTHON}"; then
    if ! "${SUBMIT_PYTHON}" -m pip --version; then
      "${SUBMIT_PYTHON}" -m ensurepip --upgrade
    fi
    # Follow the AIStudio tutorial. The couler distribution conflicts with the
    # couler-core/ant-couler modules used by the submission SDK.
    "${SUBMIT_PYTHON}" -m pip install --index-url "${SUBMIT_INDEX}" aii-pypai
    "${SUBMIT_PYTHON}" -m pip uninstall -y couler
    "${SUBMIT_PYTHON}" -m pip install --index-url "${SUBMIT_INDEX}" --force-reinstall couler-core ant-couler
    sdk_ready "${SUBMIT_PYTHON}" || {
      echo "Submission SDK verification failed; no GPU job was submitted." >&2
      exit 1
    }
  fi
fi

echo "Submission SDK ready: ${SUBMIT_PYTHON}"
if (( check_only )); then
  exit 0
fi
cd "${ROOT_DIR}"
exec "${SUBMIT_PYTHON}" "${ROOT_DIR}/main.py" "$@"
