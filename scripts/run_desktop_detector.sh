#!/usr/bin/env bash
# Launch the live webcam YOLO detector on your computer.
# Any options you add are passed straight through, for example:
#   ./scripts/run_desktop_detector.sh --conf 0.5 --camera-index 1
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -x ".venv/bin/python" ]]; then
  ".venv/bin/python" src/desktop_yolo_detector.py "$@"
else
  python3 src/desktop_yolo_detector.py "$@"
fi
