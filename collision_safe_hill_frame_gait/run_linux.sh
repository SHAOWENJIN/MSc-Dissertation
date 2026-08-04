#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PACKAGE_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Virtual environment not found." >&2
  echo "First run: bash collision_safe_hill_frame_gait/setup_linux.sh" >&2
  exit 1
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -c \
  'import mujoco; mujoco.MjModel.from_xml_path("collision_safe_hill_frame_gait/scene_hill_frame.xml"); print("Scene check passed.")'

exec "${PYTHON_BIN}" collision_safe_hill_frame_gait/run_simulation.py \
  --gravity-scale 0.0 \
  --cycles 1 \
  --sequence RR,RL,FR,FL \
  --real-time \
  --hold-viewer-on-exit \
  --trial-name collision_safe_linux_demo
