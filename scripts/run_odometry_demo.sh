#!/usr/bin/env bash
# Odometry demo for the autonomous rover (development computer).
#
# Drive the rover with the keyboard and watch its estimated path trace out live
# on the odometry mini-map in the "rover view" window. Odometry is dead-reckoned
# from /cmd_vel (no encoders), so the map moves ONLY when you drive.
#
# No YOLO model needed — this demo runs just the camera, odometry integrator,
# the overlay window, and keyboard teleop.
#
#   ./scripts/run_odometry_demo.sh
#
# Controls (in this terminal): w/s forward-back, a/d turn, space stop, q quit.
# Quitting (q here, or q/Esc in the window) shuts the whole demo down.
set -euo pipefail

REPO="$HOME/Desktop/AutonomousRover"
LOG="${TMPDIR:-/tmp}/rover_odometry_demo.log"

# rover_env.sh references $ZSH_VERSION unguarded, so relax 'nounset' while sourcing.
set +u
source "$REPO/ros2_ws/rover_env.sh"
set -u

echo
echo "Starting odometry demo (camera + odometry + overlay window)..."
echo "Background node logs -> $LOG"
: > "$LOG"

pids=()
cleanup() {
  echo; echo "Shutting down odometry demo..."
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

# Background nodes (output to a log so this terminal stays clean for the keyboard).
ros2 run rover_control camera_node --ros-args -p source:=usb >>"$LOG" 2>&1 &
pids+=($!)
ros2 run rover_control odometry_node >>"$LOG" 2>&1 &
pids+=($!)
ros2 run rover_control viz_node >>"$LOG" 2>&1 &
pids+=($!)

sleep 3
echo "Drive with the keyboard: w/s = forward/back, a/d = turn, space = stop, q = quit."
echo "Watch the green path grow on the odometry mini-map (top-right of the window)."
echo

# Keyboard teleop runs in the foreground so it receives your keystrokes.
ros2 run rover_control keyboard_node
