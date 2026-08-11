"""odometry_node: dead-reckon the rover's pose from /cmd_vel.

There are no wheel encoders yet, so this node estimates where the rover would be
by integrating the velocity commands it is told to execute. It listens to the
same Twist stream the motors act on and integrates it over time:

    theta += omega * dt
    x     += v * cos(theta) * dt
    y     += v * sin(theta) * dt

    subscribes : /cmd_vel   (geometry_msgs/Twist)
    publishes  : /odom      (nav_msgs/Odometry)

The result is *open-loop* odometry: it tells you where the rover should be if the
wheels perfectly obeyed every command. It will drift from reality (no encoder
feedback), which is exactly why it is useful for testing on the computer — you
can watch the intended trajectory without any hardware. Swap in encoder-based
odometry later and every downstream consumer of /odom keeps working unchanged.

PARAMETERS
    rate (float)      : integration + publish rate [Hz]      (default 20.0)
    frame_id (str)    : odometry frame                       (default 'odom')
    child_frame (str) : rover body frame                     (default 'base_link')
"""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')

        self.declare_parameter('rate', 20.0)
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('child_frame', 'base_link')

        rate = float(self.get_parameter('rate').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.child_frame = self.get_parameter('child_frame').value

        # Estimated pose (planar) and the latest commanded velocity.
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._v = 0.0        # latest linear.x  [m/s]
        self._w = 0.0        # latest angular.z [rad/s]
        self._last_t = time.time()

        self.sub = self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)
        self.pub = self.create_publisher(Odometry, '/odom', 10)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._tick)

        self.get_logger().info(
            'odometry_node started; integrating /cmd_vel -> /odom @ %.0f Hz' % rate)

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._v = float(msg.linear.x)
        self._w = float(msg.angular.z)

    def _tick(self) -> None:
        now = time.time()
        dt = now - self._last_t
        self._last_t = now
        if dt <= 0.0:
            return

        # Integrate the (held) command. Midpoint on theta keeps curved paths honest.
        self.theta += self._w * dt
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        self.x += self._v * math.cos(self.theta) * dt
        self.y += self._v * math.sin(self.theta) * dt

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        # Yaw -> quaternion (planar, so only z/w are non-zero).
        msg.pose.pose.orientation.z = math.sin(self.theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.theta / 2.0)
        msg.twist.twist.linear.x = self._v
        msg.twist.twist.angular.z = self._w
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
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
