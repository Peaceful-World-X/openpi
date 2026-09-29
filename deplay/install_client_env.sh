#!/usr/bin/env bash
set -euo pipefail

# Create an isolated environment for deplay without installing OpenPI training dependencies.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${1:-${ROOT_DIR}/.venv}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${ENV_DIR}" <<'PY'
import sys
from pathlib import Path

target = Path(sys.argv[1])
if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    raise SystemExit(
        f"deplay requires Python 3.10, 3.11, or 3.12, found {sys.version.split()[0]}"
    )
print(f"Creating client environment at {target}")
PY

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
fi

"${ENV_DIR}/bin/python" -m pip install --upgrade pip
"${ENV_DIR}/bin/python" -m pip install -e "${ROOT_DIR}/packages/openpi-client"
"${ENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/deplay/requirements-client.txt"

cat <<EOF

Client environment is ready:
  ${ENV_DIR}/bin/python

From the repository root, run for example:
  ${ENV_DIR}/bin/python -m deplay.ur7_client --help
EOF
