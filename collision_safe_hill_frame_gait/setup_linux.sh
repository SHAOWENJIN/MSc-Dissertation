#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PACKAGE_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

command -v python3 >/dev/null 2>&1 || {
  echo "python3 is required. Install Python 3.11 or 3.12 first." >&2
  exit 1
}

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${PACKAGE_DIR}/requirements.txt"

echo "Setup complete: ${VENV_DIR}"
echo "Run: bash collision_safe_hill_frame_gait/run_linux.sh"
