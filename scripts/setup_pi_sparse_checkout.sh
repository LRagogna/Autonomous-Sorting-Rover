#!/usr/bin/env bash
set -euo pipefail

# Configure this checkout for Raspberry Pi runtime use.
#
# The Pi should only pull files needed for rover operation. Training data,
# training code, desktop-only detector tools, generated model run artifacts,
# documentation, and CAD files stay out of the Pi working tree.
#
# Sparse checkout keeps the working tree allowlisted to runtime paths. The LFS
# fetch exclusion is a second guard so normal Git LFS operations do not download
# dataset, documentation media, CAD, or model-training artifacts on the Pi.

git lfs install --local --skip-smudge
git config --local lfs.fetchexclude 'data/**,docs/**,solidworks/**,ml/**,models/**,tests/**,yolov8n.pt'
git sparse-checkout init --no-cone
git sparse-checkout set \
  '/.gitattributes' \
  '/.gitignore' \
  '/README.md' \
  '/requirements.txt' \
  '/scripts/setup_pi_sparse_checkout.sh' \
  '/src/main.py' \
  '/src/serial_drive_turns.ino' \
  '/ros2_ws/README.md' \
  '/ros2_ws/rover_env.sh' \
  '/ros2_ws/src/'

echo "Configured sparse checkout for Raspberry Pi runtime files only."
echo "Excluded data, docs, CAD, ML training, model artifacts, tests, and desktop tools from Git LFS fetches."
echo "Use 'git pull' normally after this; non-runtime files will stay off the Pi."
