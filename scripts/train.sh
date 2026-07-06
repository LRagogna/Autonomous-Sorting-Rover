#!/usr/bin/env bash
# Train the detector from the processed YOLO dataset.
#
# USAGE
#   ./scripts/train.sh                # normal training (uses the Apple GPU)
#   ./scripts/train.sh --epochs 60    # pass extra training options through
#   ./scripts/train.sh --scale 0.9    # e.g. more size variety for distance
#   DEVICE=cpu ./scripts/train.sh     # train on the CPU instead of the GPU
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

echo "==> Training the detector on device '$DEVICE'..."
"$PY" ml/train_yolo.py --device "$DEVICE" "$@"

echo ""
echo "==> All done. Try it live with:  ./scripts/run_desktop_detector.sh"
