# Autonomous Rover — Raspberry Pi ROS 2 workspace (`ros2_pi`)

The **on-robot** ROS 2 Humble workspace. Same nodes as the development workspace
([`../ros2_ws`](../ros2_ws)), but the defaults and one extra node are for the
real rover:

* camera defaults to the **Pi CSI camera** (picamera2)
* motors default to the **real Arduino serial bridge** (`arduino_bridge_node`)
* the OpenCV overlay window (`viz_node`) is **off** by default — the Pi is headless

```
camera_node ─▶ perception_node ─▶ action_node ─┬─▶ arduino_bridge_node ─▶ Arduino (serial)
 (Pi camera)      (YOLO)           (/cmd_vel)   └─▶ odometry_node ─▶ /odom
```

> The development twin, `ros2_ws`, is where you test on the laptop with a webcam
> and the live overlay window. This `ros2_pi` copy is what the Pi pulls and runs.

---

## 1. Getting it onto the Pi

The Pi pulls this folder from git via sparse checkout. On the Pi, in the repo
(`~/AutonomousRover`):

```bash
bash scripts/setup_pi_sparse_checkout.sh   # allowlists ros2_pi/ (+ runtime files)
git pull                                    # ros2_pi/ now appears in the working tree
```

`setup_pi_sparse_checkout.sh` pulls `ros2_pi/` (this workspace) and **not**
`ros2_ws/` (the desktop one). Re-run it after pulling if you set up the Pi before
this folder existed.

## 2. One-time environment (RoboStack ROS 2 Humble, native — no Docker/VM)

```bash
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
micromamba create -y -n ros_env -c robostack-staging -c conda-forge \
  python=3.11 ros-humble-ros-base ros-humble-geometry-msgs ros-humble-nav-msgs \
  ros-humble-sensor-msgs colcon-common-extensions
# Pi runtime pip deps (camera/detector/serial):
pip install ultralytics opencv-python pyserial
# picamera2 is an apt package on Raspberry Pi OS Bookworm:
sudo apt install -y python3-picamera2
```

## 3. Build

```bash
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$(micromamba shell hook --shell bash)"
micromamba activate ros_env
cd ~/AutonomousRover/ros2_pi
colcon build          # plain build — NOT --symlink-install (broken with this env)
```

## 4. Every new terminal

```bash
source ~/AutonomousRover/ros2_pi/rover_env.sh
```

## 5. Run

```bash
# Full autonomous rover: Pi camera -> YOLO -> steer -> real Arduino motors
ros2 launch rover_control perception_pipeline.launch.py
```

Useful overrides:

```bash
# Bench test with no Arduino (just prints motor values):
ros2 launch rover_control perception_pipeline.launch.py motor_backend:=fake

# Different serial port / chase one class:
ros2 launch rover_control perception_pipeline.launch.py port:=/dev/ttyUSB0 target_class:=bit
```

Watch the odometry estimate from another terminal:

```bash
ros2 topic echo /odom
```

## Arduino serial contract

`arduino_bridge_node` writes one line per command:

```
LEFT,RIGHT\n        e.g.  54,-51
```

integers in `[-max_motor, max_motor]` (default ±255). The sketch in
`src/serial_drive_turns.ino` parses two comma-separated integers per line. If the
serial port can't be opened, the node logs a warning and falls back to printing,
so the software still runs for testing.

## Nodes (identical to `ros2_ws`, plus the bridge)

| Node | Role |
|------|------|
| `camera_node` | Pi camera → `/camera/image/compressed` |
| `perception_node` | YOLO → `/perception/detection` |
| `action_node` | steer toward target → `/cmd_vel` |
| `odometry_node` | dead-reckon `/cmd_vel` → `/odom` |
| `arduino_bridge_node` | `/cmd_vel` → Arduino motor values over serial |
| `fake_motor_node` | `/cmd_vel` → printed motor values (bench testing) |
| `keyboard_node` | manual WASD teleop → `/cmd_vel` |
| `viz_node` | OpenCV overlay window (needs a display; off by default) |
