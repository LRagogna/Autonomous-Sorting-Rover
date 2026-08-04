#!/usr/bin/env bash
# Start the local browser control center: upload clips, process data, review and
# edit labels, train versioned models, test the detector, and deploy.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

"$PY" gui/app.py "$@"
