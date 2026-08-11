# Autonomous Rover — ROS 2 control workspace (development computer)

Native ROS 2 **Humble** on Apple Silicon (macOS), installed via **RoboStack**
(conda-forge packages). **No Gazebo, no VM, no Docker, no 3D simulator.**

This is the **development/test** workspace — you run it on the laptop with a USB
webcam (or a video file) and watch the rover "think" in a live OpenCV window.
The on-robot twin is [`../ros2_pi`](../ros2_pi): the same nodes, but with Pi
camera defaults and a real Arduino motor bridge instead of the viz window.

Everything is wired around one integration point, `/cmd_vel`
(`geometry_msgs/Twist`), so any command source drops into the same motor path.

**Manual teleop:**

```
 keyboard_node  --(Twist)-->  /cmd_vel  -->  fake_motor_node
   reads keys                               prints simulated LEFT / RIGHT motors
```

**Autonomous perception pipeline** (one launch file starts all of it):

```
 camera_node ─▶ perception_node ─▶ action_node ─┬─▶ fake_motor_node
  (webcam)        (YOLO)          (/cmd_vel)     ├─▶ odometry_node ─▶ /odom
                                                 └─▶ viz_node  (OpenCV overlay window)
```

`viz_node` shows the live feed with the detected object box, the LEFT/RIGHT motor
values, a turn/forward arrow, and an odometry mini-map — so you can *see* exactly
what the rover would do. `odometry_node` dead-reckons pose from `/cmd_vel` (no
encoders yet) onto `/odom`.

Because `/cmd_vel` is the only integration point, `fake_motor_node` is replaced
on the Pi by `arduino_bridge_node` (real serial) with no other changes.

### Autonomous pipeline — one command

```bash
source ~/Desktop/AutonomousRover/ros2_ws/rover_env.sh
# Laptop webcam -> YOLO -> steer -> simulated motors, with the live overlay window:
ros2 launch rover_control perception_pipeline.launch.py source:=usb
```

Common overrides: `source:=video video_path:=clip.mp4` (offline replay),
`target_class:=bit` (chase one class), `search:=true` (rotate to look around),
`viz:=false` / `odom:=false` (turn those nodes off).

---

## 1. One-time installation (already done on this machine)

ROS 2 is installed in a dedicated micromamba environment called `ros_env`, kept
separate from your Anaconda install.

```bash
# micromamba (fast native conda solver) — via Homebrew
brew install micromamba
export MAMBA_ROOT_PREFIX="$HOME/micromamba"

# Create the ROS 2 Humble environment (native arm64, no Docker/VM)
micromamba create -y -n ros_env \
  -c robostack-staging -c conda-forge \
  python=3.12 \
  ros-humble-ros-base ros-humble-geometry-msgs \
  colcon-common-extensions compilers cmake pkg-config make ninja
```

## 2. Build the workspace

```bash
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$(micromamba shell hook --shell zsh)"
micromamba activate ros_env

cd ~/Desktop/AutonomousRover/ros2_ws
colcon build            # NOTE: plain build — do NOT use --symlink-install
```

> `--symlink-install` is broken with the newer setuptools in this env (it uses a
> removed `--editable` flag). Plain `colcon build` works.

## 3. Environment setup (every new terminal)

A helper does the activation + workspace overlay for you:

```bash
source ~/Desktop/AutonomousRover/ros2_ws/rover_env.sh
```

This activates `ros_env` and overlays the built workspace. It is **zsh-aware**:
on macOS your shell is zsh, so it sources the `.zsh` setup files. (Sourcing the
`.bash` files under zsh silently breaks because they rely on `$BASH_SOURCE`.)

## 4. Run — two terminals

**Terminal A — fake motor driver (subscriber):**
```bash
source ~/Desktop/AutonomousRover/ros2_ws/rover_env.sh
ros2 run rover_control fake_motor_node
```

**Terminal B — keyboard teleop (publisher):**
```bash
source ~/Desktop/AutonomousRover/ros2_ws/rover_env.sh
ros2 run rover_control keyboard_node
```

Then, in Terminal B (no Enter needed):

| Key         | Action                      |
|-------------|-----------------------------|
| `w` / `s`   | forward speed  + / -        |
| `a` / `d`   | turn left / right  + / -    |
| `space`/`x` | stop                        |
| `q`         | quit                        |

Terminal A prints simulated motor values for every command, e.g.:

```
cmd_vel v=+0.15 w=+0.20  ->  LEFT motor =   +42.2   RIGHT motor =   +63.3
```

## 5. Tuning the rover geometry (optional)

`fake_motor_node` exposes ROS parameters, so no code change is needed for a
different chassis:

```bash
ros2 run rover_control fake_motor_node --ros-args \
  -p wheel_base:=0.25 -p max_linear:=0.6 -p max_motor:=1023.0
```

---

## Running on Windows

The nodes are cross-platform (`keyboard_node` uses `msvcrt` on Windows,
`termios` on macOS/Linux; `fake_motor_node` and the math are pure Python). Only
the install + activation differs. The macOS setup above was tested; the Windows
steps below are the standard RoboStack flow but have **not** been run on this
machine — treat them as a starting point.

1. Install RoboStack ROS 2 Humble (native, no Docker/VM). In a fresh terminal
   with `micromamba`/`mamba` available:
   ```powershell
   micromamba create -n ros_env -c robostack-staging -c conda-forge `
     python=3.11 ros-humble-ros-base ros-humble-geometry-msgs `
     colcon-common-extensions
   ```
2. Build (PowerShell), from the workspace root:
   ```powershell
   micromamba activate ros_env
   cd $HOME\Desktop\AutonomousRover\ros2_ws
   colcon build
   ```
   Our package is pure Python, so no Visual Studio C++ compiler is required.
3. Every new terminal — activate + overlay with the PowerShell helper:
   ```powershell
   . $HOME\Desktop\AutonomousRover\ros2_ws\rover_env.ps1
   ```
4. Run the two nodes in two PowerShell windows exactly as on macOS:
   ```powershell
   ros2 run rover_control fake_motor_node   # window A
   ros2 run rover_control keyboard_node     # window B
   ```

Notes for Windows:
- Use **PowerShell**, and source `install\setup.ps1` (done by `rover_env.ps1`) —
  not the `.bash`/`.zsh` files.
- `keyboard_node` needs a real console window (works in PowerShell/cmd, not
  inside some IDE output panes that don't forward raw key input).

## Package layout

```
ros2_ws/
├── rover_env.sh                    # activate env + overlay workspace (zsh/bash aware)
└── src/rover_control/
    ├── package.xml                 # ament_python; deps: rclpy, geometry/sensor/std/nav_msgs
    ├── setup.py                     # console_scripts for every node below
    ├── setup.cfg
    ├── launch/
    │   └── perception_pipeline.launch.py  # starts camera+perception+action+motor+odom+viz
    └── rover_control/
        ├── differential_drive.py   # pure Twist -> (left,right) math (ROS/HW-free)
        ├── keyboard_node.py        # reads keys, publishes Twist on /cmd_vel
        ├── fake_motor_node.py      # subscribes /cmd_vel, prints motor values
        ├── camera_node.py          # webcam/Pi cam/video -> /camera/image/compressed
        ├── perception_node.py      # YOLO -> /perception/detection
        ├── action_node.py          # detection -> /cmd_vel (steer toward target)
        ├── odometry_node.py        # integrate /cmd_vel -> /odom (dead reckoning)
        └── viz_node.py             # OpenCV overlay: feed + boxes + motor values + odom map
```

> The on-robot copy is `../ros2_pi`, which adds `arduino_bridge_node.py` (real
> serial motors) and defaults to the Pi camera. Keep node changes in sync between
> the two workspaces.

---

## Swapping in a real Arduino later (no changes elsewhere)

The differential-drive math lives in `differential_drive.py`, independent of ROS
and hardware. To drive a real Arduino:

1. Add a dependency such as `pyserial` to the environment.
2. Create `arduino_bridge_node.py` that subscribes to `/cmd_vel`, calls
   `twist_to_motors(...)` exactly as the fake node does, and — instead of
   printing — writes the values over serial:
   ```python
   self.serial.write(f"{int(left)},{int(right)}\n".encode())
   ```
   (It can even subclass `FakeMotorNode` and override only `_drive_motors()`.)
3. Add its `console_scripts` entry in `setup.py`, rebuild, and run it in place of
   `fake_motor_node`.

`keyboard_node`, the `/cmd_vel` topic, and the Twist message all stay identical.
