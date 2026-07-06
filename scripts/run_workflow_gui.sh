#!/usr/bin/env bash
# Start the local browser GUI for uploading clips, processing data, reviewing
# frames, and training the YOLO detector.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

"$PY" tools/workflow_gui.py "$@"
