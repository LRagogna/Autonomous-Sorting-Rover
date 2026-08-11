#!/usr/bin/env bash
# Source this in every new terminal ON THE RASPBERRY PI before using the rover
# ROS 2 workspace:
#     source ~/AutonomousRover/ros2_pi/rover_env.sh
#
# It activates the native (RoboStack) ROS 2 Humble env and overlays this
# workspace's build. No Docker, VM, Gazebo, or simulator involved.
#
# This is the Pi twin of ros2_ws/rover_env.sh; only the workspace path differs
# (ros2_pi instead of ros2_ws) and the repo lives at ~/AutonomousRover on the Pi.

export MAMBA_ROOT_PREFIX="$HOME/micromamba"

# Detect the interactive shell so the hook and overlay use matching syntax.
if [ -n "$ZSH_VERSION" ]; then
    _SH=zsh; _EXT=zsh
else
    _SH=bash; _EXT=bash
fi

eval "$(micromamba shell hook --shell "$_SH")"
micromamba activate ros_env

# Overlay this workspace (only if it has been built).
_WS_DIR="$HOME/AutonomousRover/ros2_pi"
if [ -f "$_WS_DIR/install/setup.$_EXT" ]; then
    source "$_WS_DIR/install/setup.$_EXT"
    echo "rover Pi ROS 2 env ready: ROS_DISTRO=$ROS_DISTRO, workspace overlay loaded."
else
    echo "rover Pi ROS 2 env ready: ROS_DISTRO=$ROS_DISTRO (workspace not built yet -> run 'colcon build')."
fi
