#!/usr/bin/env bash
# Interactive desktop tester for the autonomous rover (development computer).
#
# It ASKS which object the rover should detect before starting, then launches the
# full perception pipeline (webcam -> YOLO -> steer -> simulated motors) plus the
# live OpenCV overlay window and odometry, configured to detect and chase ONLY
# the object you pick. This enforces the rule that the rover roves toward the
# specified object, never "any object".
#
#   ./scripts/run_desktop_tester.sh                # prompts, then runs
#   ./scripts/run_desktop_tester.sh conf:=0.5      # extra launch args pass through
#
# Quit: press q/Esc in the "rover view" window, or Ctrl-C in this terminal.
set -euo pipefail

REPO="$HOME/Desktop/AutonomousRover"
CLASSES_FILE="$REPO/deploy/classes.txt"
MODEL="$REPO/models/active_model.pt"

# --- load the list of detectable classes -------------------------------------
classes=()
while IFS= read -r line || [ -n "$line" ]; do
  line="$(echo "$line" | tr -d '[:space:]')"
  [ -n "$line" ] && classes+=("$line")
done < "$CLASSES_FILE"

if [ "${#classes[@]}" -eq 0 ]; then
  echo "No classes found in $CLASSES_FILE" >&2
  exit 1
fi

# --- ask which object to detect ----------------------------------------------
echo "=================================================="
echo "  Autonomous Rover - desktop detection test"
echo "=================================================="
echo "Which object should the rover detect and chase?"
echo
i=1
for c in "${classes[@]}"; do
  printf "   %d) %s\n" "$i" "$c"
  i=$((i + 1))
done
echo

target=""
while [ -z "$target" ]; do
  printf "Enter number or name: "
  read -r choice || exit 1
  if echo "$choice" | grep -qE '^[0-9]+$'; then
    if [ "$choice" -ge 1 ] && [ "$choice" -le "${#classes[@]}" ]; then
      target="${classes[$((choice - 1))]}"
    fi
  else
    for c in "${classes[@]}"; do
      [ "$c" = "$choice" ] && target="$c"
    done
  fi
  [ -z "$target" ] && echo "  '$choice' is not one of the options - try again."
done

echo
echo ">> Rover will detect and chase ONLY: '$target'"
echo ">> Starting pipeline..."
echo

# --- activate the ROS 2 env + launch -----------------------------------------
# rover_env.sh picks the right (bash/zsh) overlay files for the current shell.
# It references $ZSH_VERSION unguarded, so relax 'nounset' while sourcing it.
set +u
source "$REPO/ros2_ws/rover_env.sh"
set -u

exec ros2 launch rover_control perception_pipeline.launch.py \
  source:=usb \
  target_class:="$target" \
  model:="$MODEL" \
  "$@"
