#!/usr/bin/env bash
# One command to retrain the detector from scratch.
#
# It does the three steps for you:
#   1. Slice any NEW videos in data/raw/videos into photos. Videos that were
#      already sliced are skipped, so this is safe to run every time.
#   2. Rebuild the dataset from ALL your source folders
#      (data/raw/photos + data/hand_labeled). This picks up anything new you
#      added, together with everything old.
#   3. Train the detector and save it to models/yolo_detector.pt
#
# USAGE
#   ./scripts/retrain.sh                # normal retrain (uses the Apple GPU)
#   ./scripts/retrain.sh --epochs 60    # pass extra training options through
#   ./scripts/retrain.sh --scale 0.9    # e.g. more size variety for distance
#   DEVICE=cpu ./scripts/retrain.sh     # train on the CPU instead of the GPU
#   FRAME_STEP=10 ./scripts/retrain.sh  # save more frames per new video
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

# How many frames to keep per new video (every Nth frame). Set FRAME_STEP=10 etc.
FRAME_STEP="${FRAME_STEP:-15}"

echo "==> Step 1/3: slicing any new videos in data/raw/videos into photos..."
"$PY" data/extract_video_frames.py --all --frame-step "$FRAME_STEP"

echo ""
echo "==> Step 2/3: building the dataset from all your source folders..."
"$PY" data/auto_label_frames.py --overwrite

echo ""
echo "==> Step 3/3: training the detector on device '$DEVICE'..."
"$PY" ml/train_yolo.py --device "$DEVICE" "$@"

echo ""
echo "==> All done. Try it live with:  ./scripts/run_desktop_detector.sh"
