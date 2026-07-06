# Autonomous Rover — ROS 2 control workspace

Native ROS 2 **Humble** on Apple Silicon (macOS), installed via **RoboStack**
(conda-forge packages). **No Gazebo, no VM, no Docker, no 3D simulator.**

Two nodes wired over the standard `/cmd_vel` topic:

```
 keyboard_node  --(geometry_msgs/Twist)-->  /cmd_vel  -->  fake_motor_node
   reads keys                                              prints simulated
   publishes Twist                                         LEFT / RIGHT motors
```

`/cmd_vel` is the only integration point, so `fake_motor_node` can later be
replaced by an Arduino serial bridge without touching `keyboard_node` or any
other part of the system.

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
    ├── package.xml                 # ament_python package, deps: rclpy, geometry_msgs
    ├── setup.py                     # console_scripts: keyboard_node, fake_motor_node
    ├── setup.cfg
    └── rover_control/
        ├── differential_drive.py   # pure Twist -> (left,right) math (ROS/HW-free)
        ├── keyboard_node.py        # reads keys, publishes Twist on /cmd_vel
        └── fake_motor_node.py      # subscribes /cmd_vel, prints motor values
```

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
