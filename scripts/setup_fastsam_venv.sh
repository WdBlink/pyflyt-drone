#!/usr/bin/env bash
set -euo pipefail

# Create a dedicated venv to avoid polluting system Python (ROS installs can be sensitive).
# This venv is only needed if you want to run FastSAM segmentation inside ArdupilotGazeboObjLockEnv.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT}/.venv-fastsam"

python3 -m venv "${VENV_DIR}"

source "${VENV_DIR}/bin/activate"

python -m pip install -U pip wheel

# Install CPU-only PyTorch wheels (avoids downloading CUDA/NVIDIA dependencies).
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.9.1 torchvision==0.24.1

# FastSAM is provided by ultralytics. OpenCV is needed for many Ultralytics ops.
python -m pip install ultralytics opencv-python pillow

echo ""
echo "Done."
echo "Activate with:"
echo "  source \"${VENV_DIR}/bin/activate\""
