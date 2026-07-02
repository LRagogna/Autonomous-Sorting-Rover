#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 src/pi_realtime_classifier.py --backend edgetpu "$@"
