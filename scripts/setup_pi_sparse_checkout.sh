#!/usr/bin/env bash
set -euo pipefail

# Configure this checkout for Raspberry Pi runtime use.
#
# The Pi should only pull files needed for rover operation and on-device
# hardware tests. Training data, training code, desktop-only detector tools,
# generated model run artifacts, documentation, and CAD files stay out of the Pi
# working tree. The one exception is models/active_model.pt: the current active
# detector is small (~6 MB, a plain git blob, not LFS) and is needed to run the
# detector on the Pi, so it is pulled. Other model versions stay off the Pi.
#
# Sparse checkout keeps the working tree allowlisted to runtime paths. The LFS
# fetch exclusion is a second guard so normal Git LFS operations do not download
# dataset, documentation media, CAD, or model-training artifacts on the Pi.

git lfs install --local --skip-smudge
git config --local lfs.fetchexclude 'data/**,docs/**,solidworks/**,ml/**,models/**,requirements.txt,yolov8n.pt'
git sparse-checkout init --no-cone
git sparse-checkout set \
  '/.gitattributes' \
  '/.gitignore' \
  '/README.md' \
  '/requirements-pi.txt' \
  '/scripts/setup_pi_sparse_checkout.sh' \
  '/scripts/run_pi_detector.sh' \
  '/models/active_model.pt' \
  '/src/main.py' \
  '/src/pi_yolo_detector.py' \
  '/src/record_video.py' \
  '/src/serial_drive_turns.ino' \
  '/tests/' \
  '/ros2_pi/'

# NOTE: the Pi pulls the ros2_pi/ workspace (Pi camera defaults + real Arduino
# motor bridge), NOT ros2_ws/. ros2_ws/ is the development-computer workspace
# (it includes the OpenCV overlay viz window, which needs a display) and stays
# off the headless Pi.

echo "Configured sparse checkout for Raspberry Pi runtime files and hardware tests."
echo "Excluded data, docs, CAD, ML training, model artifacts, and desktop tools from Git LFS fetches."
echo "Use 'git pull' normally after this; non-runtime files will stay off the Pi."
