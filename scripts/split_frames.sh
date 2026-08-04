#!/usr/bin/env bash
# Split raw clips into JPG frames for training (extraction only, no labeling).
#
# Put videos in:
#   data/raw_videos/<class>/<video_file>
#
# This writes frames to:
#   data/frames/<class>/<class>__<clip>__frame_<number>.jpg
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

echo "==> Splitting clips in data/raw_videos into JPG frames..."
"$PY" ml/extract_frames.py --all --frame-step "$FRAME_STEP"

echo ""
echo "==> Frames are in data/frames/<class>/"
