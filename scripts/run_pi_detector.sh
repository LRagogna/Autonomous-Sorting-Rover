#!/usr/bin/env bash
set -euo pipefail

# Run the live Pi-camera YOLO detector and stream it to a browser.
#
# Run this ON THE RASPBERRY PI. It reads the Pi camera, runs the trained model,
# draws boxes around recognized objects, and serves an MJPEG video stream. Open
# the URL it prints (http://<pi-ip>:8000/) in a browser on the same network.
#
# One-time setup on the Pi (see src/pi_yolo_detector.py for details):
#   sudo apt install -y python3-picamera2
#   pip install ultralytics
#   scp deploy/active_model.pt  pi@<pi-host>:~/AutonomousRover/models/   # from your Mac
#
# Extra options pass straight through, e.g.:
#   ./scripts/run_pi_detector.sh --conf 0.4 --port 9000 --window

cd "$(dirname "$0")/.."
exec python3 src/pi_yolo_detector.py "$@"
