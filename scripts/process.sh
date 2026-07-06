#!/usr/bin/env bash
# Split new clips, then process new raw photos into YOLO images, labels, and
# per-object review folders.
#
# Reads:
#   data/raw/clips/<object>/
#   data/raw/photos/<object>/
#
# Writes:
#   data/raw/photos/<object>/
#   data/labels/<object>/images/{train,val}/
#   data/labels/<object>/labels/{train,val}/
#   data/labels/<object>/review/
#   data/labels/dataset.yaml
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

FRAME_STEP="${FRAME_STEP:-15}"

echo "==> Splitting new clips in data/raw/clips into JPG frames..."
"$PY" data/extract_video_frames.py --all --frame-step "$FRAME_STEP" --image-format jpg

# Remove iCloud "conflict copy" duplicate frames (e.g. "..._000000 2.jpg") BEFORE
# labeling, so duplicates never get folded into the training dataset.
echo ""
echo "==> Removing any iCloud duplicate frames..."
./scripts/clean_icloud_dupes.sh || true

echo ""
echo "==> Processing new frames into YOLO labels and review images..."
"$PY" data/auto_label_frames.py --incremental "$@"

# Sweep any duplicate files/folders iCloud created while the dataset was rewritten.
echo ""
echo "==> Cleaning up any iCloud duplicates created during processing..."
./scripts/clean_icloud_dupes.sh || true

echo ""
echo "==> Processed data is in data/labels/"
