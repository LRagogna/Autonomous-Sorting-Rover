"""arduino_bridge_node: drive a real Arduino over serial from /cmd_vel.

This is the Pi's REAL motor driver — the hardware counterpart of
fake_motor_node. It subscribes to the exact same /cmd_vel Twist stream, runs the
exact same differential_drive math, and instead of *printing* the left/right
values it writes them to an Arduino over a serial port:

        LEFT,RIGHT\\n           e.g.  "54,-51\\n"

so the Arduino sketch (see src/serial_drive_turns.ino) can parse two integers
per line and drive the motor controller.

    subscribes : /cmd_vel   (geometry_msgs/Twist)

Because it reuses FakeMotorNode's decoding and only overrides _drive_motors(),
teleop, the perception pipeline, and every other publisher on /cmd_vel work with
zero changes — you just run this node instead of fake_motor_node on the Pi.

If the serial port cannot be opened (no Arduino plugged in) the node logs a
warning and falls back to printing, so the software stack still runs for testing.

PARAMETERS (in addition to fake_motor_node's wheel_base / max_* params)
    port (str)     : serial device            (default '/dev/ttyACM0')
    baud (int)     : serial baud rate          (default 115200)
    send_rate (float): max serial writes/sec; 0 = every message (default 20.0)
"""

from __future__ import annotations

import time

import rclpy

from rover_control.fake_motor_node import FakeMotorNode


class ArduinoBridgeNode(FakeMotorNode):
    def __init__(self):
        super().__init__()
        # Rename so ros2 node list / logs show what this really is.
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('send_rate', 20.0)

        self.port = self.get_parameter('port').value
        self.baud = int(self.get_parameter('baud').value)
        rate = float(self.get_parameter('send_rate').value)
        self._min_period = (1.0 / rate) if rate > 0 else 0.0
        self._last_send = 0.0

        self._serial = self._open_serial()

    def _open_serial(self):
        try:
            import serial
        except ImportError:
            self.get_logger().error(
                "arduino_bridge_node needs pyserial. Install it on the Pi:\n"
                "    pip install pyserial   (it is in requirements-pi.txt)\n"
                "Falling back to printing motor values.")
            return None
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1.0)
            time.sleep(2.0)  # let the Arduino finish its auto-reset on connect
            self.get_logger().info(
                'arduino_bridge_node: serial open on %s @ %d baud' % (self.port, self.baud))
            return ser
        except Exception as exc:  # serial.SerialException and friends
            self.get_logger().warn(
                "arduino_bridge_node: could not open %s (%s). Falling back to "
                "printing motor values so the stack still runs." % (self.port, exc))
            return None

    def _drive_motors(self, left, right, msg):
        now = time.time()
        if self._min_period and (now - self._last_send) < self._min_period:
            return
        self._last_send = now

        if self._serial is None:
            # No hardware: behave like fake_motor_node.
            super()._drive_motors(left, right, msg)
            return
        line = '%d,%d\n' % (int(round(left)), int(round(right)))
        try:
            self._serial.write(line.encode())
        except Exception as exc:
            self.get_logger().warn('arduino_bridge_node: serial write failed (%s)' % exc)

    def destroy_node(self):
        # Command a full stop, then close the port.
        try:
            if self._serial is not None:
                try:
                    self._serial.write(b'0,0\n')
                    self._serial.close()
                except Exception:
                    pass
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArduinoBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
