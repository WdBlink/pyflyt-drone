#!/usr/bin/env bash
set -euo pipefail

# Use bash so ROS2 setup scripts behave consistently even if the user's login shell is zsh.

if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found at ~/miniforge3/etc/profile.d/conda.sh" >&2
  exit 1
fi

conda activate ardupilot

# Avoid mixing user-site packages (~/.local) and any custom PYTHONPATH when running conda.
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# ROS2 Humble (system install)
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash

# If you have a mavros workspace overlay, source it here (optional):
# source "$HOME/your_ros_ws/install/setup.bash"

# Run from repo root
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

exec python eval/eval_objlock_sitl.py "$@"
