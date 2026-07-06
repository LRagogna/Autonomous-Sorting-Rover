#!/usr/bin/env bash
# Split raw object clips into JPG frames for training.
#
# Put videos in:
#   data/raw/clips/<object>/<video_file>
#
# This writes frames to:
#   data/raw/photos/<object>/<clip_name>__frame_<frame_number>.jpg
#
# USAGE
#   ./scripts/split_frames.sh
#   FRAME_STEP=10 ./scripts/split_frames.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

FRAME_STEP="${FRAME_STEP:-15}"

echo "==> Splitting clips in data/raw/clips into JPG frames..."
"$PY" data/extract_video_frames.py --all --frame-step "$FRAME_STEP" --image-format jpg

echo ""
echo "==> Frames are in data/raw/photos/<object>/"
