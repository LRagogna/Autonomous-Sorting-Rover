#!/usr/bin/env bash
# One command to retrain the detector from scratch.
#
# It does the two steps for you:
#   1. Rebuild the dataset from ALL your source folders
#      (data/raw/photos + data/hand_labeled). This picks up anything new you
#      added, together with everything old.
#   2. Train the detector and save it to models/yolo_detector.pt
#
# USAGE
#   ./scripts/retrain.sh                # normal retrain (uses the Apple GPU)
#   ./scripts/retrain.sh --epochs 60    # pass extra training options through
#   ./scripts/retrain.sh --scale 0.9    # e.g. more size variety for distance
#   DEVICE=cpu ./scripts/retrain.sh     # train on the CPU instead of the GPU
#
# When it finishes, test it with:
#   ./scripts/run_desktop_detector.sh
set -euo pipefail

# Move to the project folder no matter where this is run from.
cd "$(dirname "$0")/.."

# Use the project's Python if the virtual environment exists.
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

# The Apple GPU ("mps") is much faster on a Mac. Set DEVICE=cpu to force the CPU.
DEVICE="${DEVICE:-mps}"

echo "==> Step 1/2: building the dataset from all your source folders..."
"$PY" data/auto_label_frames.py --overwrite

echo ""
echo "==> Step 2/2: training the detector on device '$DEVICE'..."
"$PY" ml/train_yolo.py --device "$DEVICE" "$@"

echo ""
echo "==> All done. Try it live with:  ./scripts/run_desktop_detector.sh"
