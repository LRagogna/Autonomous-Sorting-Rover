#!/usr/bin/env bash
# Extract frames from new clips, then auto-label them into the YOLO dataset.
#
# Reads:
#   data/raw_videos/<class>/
#
# Writes:
#   data/frames/<class>/
#   data/yolo_dataset/images/{train,val}/
#   data/yolo_dataset/labels/{train,val}/
#   data/yolo_dataset/dataset.yaml
#
# The GUI's "Process Dataset" tab runs the same ml/process_dataset.py.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

FRAME_STEP="${FRAME_STEP:-15}"

echo "==> Extracting frames and auto-labeling new clips..."
"$PY" ml/process_dataset.py --all --frame-step "$FRAME_STEP" "$@"

# Sweep any duplicate files iCloud created while the dataset was rewritten.
echo ""
echo "==> Cleaning up any iCloud duplicates..."
./scripts/clean_icloud_dupes.sh || true

echo ""
echo "==> Processed data is in data/yolo_dataset/"
