# Windows (PowerShell) equivalent of rover_env.sh.
# Run this in every new PowerShell terminal before using the workspace:
#     . $HOME\Desktop\AutonomousRover\ros2_ws\rover_env.ps1
#
# Assumes ROS 2 Humble was installed with RoboStack into a micromamba/conda env
# named "ros_env" (see README "Running on Windows"). No Docker/VM/Gazebo/sim.

$env:MAMBA_ROOT_PREFIX = "$HOME\micromamba"

# Activate the ROS 2 env. Use whichever tool you installed RoboStack with.
if (Get-Command micromamba -ErrorAction SilentlyContinue) {
    micromamba activate ros_env
} else {
    conda activate ros_env
}

# Overlay this workspace (only if it has been built).
$wsDir = "$HOME\Desktop\AutonomousRover\ros2_ws"
$overlay = Join-Path $wsDir "install\setup.ps1"
if (Test-Path $overlay) {
    . $overlay
    Write-Host "rover ROS 2 env ready: ROS_DISTRO=$env:ROS_DISTRO, workspace overlay loaded."
} else {
    Write-Host "rover ROS 2 env ready: ROS_DISTRO=$env:ROS_DISTRO (workspace not built yet -> run 'colcon build')."
}
