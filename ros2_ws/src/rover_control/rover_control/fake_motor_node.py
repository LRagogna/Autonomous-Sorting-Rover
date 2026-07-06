"""fake_motor_node: subscribe to /cmd_vel and print simulated motor values.

This node stands in for the real motor driver. It subscribes to
geometry_msgs/Twist on /cmd_vel, converts each command into left/right motor
values via the shared differential_drive module, and prints them instead of
driving hardware.

Swapping in an Arduino serial bridge later:
    * /cmd_vel and the Twist message stay exactly the same, so keyboard_node
      (and any other publisher) needs no changes.
    * differential_drive.twist_to_motors() is reused as-is.
    * Only _drive_motors() changes: replace the print with a serial write, e.g.
          self.serial.write(f"{int(left)},{int(right)}\\n".encode())
      A drop-in arduino_bridge_node can subclass this node and override that one
      method.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from rover_control.differential_drive import DriveConfig, twist_to_motors


class FakeMotorNode(Node):
    def __init__(self):
        super().__init__('fake_motor_node')

        # Expose the drive parameters as ROS parameters so the same node works
        # for a differently-sized rover without code changes.
        self.declare_parameter('wheel_base', 0.30)
        self.declare_parameter('max_linear', 0.50)
        self.declare_parameter('max_angular', 1.50)
        self.declare_parameter('max_motor', 255.0)

        self.cfg = DriveConfig(
            wheel_base=self.get_parameter('wheel_base').value,
            max_linear=self.get_parameter('max_linear').value,
            max_angular=self.get_parameter('max_angular').value,
            max_motor=self.get_parameter('max_motor').value,
        )

        self.subscription = self.create_subscription(
            Twist, '/cmd_vel', self._on_cmd_vel, 10)

        self.get_logger().info(
            'fake_motor_node started; listening on /cmd_vel '
            '(wheel_base=%.2f m, max_motor=%.0f)'
            % (self.cfg.wheel_base, self.cfg.max_motor))

    def _on_cmd_vel(self, msg: Twist):
        left, right = twist_to_motors(msg.linear.x, msg.angular.z, self.cfg)
        self._drive_motors(left, right, msg)

    def _drive_motors(self, left, right, msg):
        """Simulated actuation. Replace this body with a serial write for Arduino."""
        self.get_logger().info(
            'cmd_vel v=%+.2f w=%+.2f  ->  LEFT motor = %+7.1f   RIGHT motor = %+7.1f'
            % (msg.linear.x, msg.angular.z, left, right))


def main(args=None):
    rclpy.init(args=args)
    node = FakeMotorNode()
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
