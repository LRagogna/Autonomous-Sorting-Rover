"""action_node: turn detections into wheel commands on /cmd_vel.

This is the "decide and drive" node. It listens to what the perception_node
sees and publishes a geometry_msgs/Twist velocity command, steering the rover
toward the target object.

    subscribes : /perception/detection   (std_msgs/String, JSON)
    publishes  : /cmd_vel                 (geometry_msgs/Twist)

/cmd_vel is the SAME topic keyboard_node publishes and fake_motor_node (or a
future Arduino bridge) consumes, so this node drops straight into the existing
motor path — no downstream changes.

BEHAVIOR (a simple proportional controller)
    * No target seen (or confidence too low): stop, or slowly rotate to search
      if 'search' is enabled.
    * Target seen but off to one side: turn in place toward it
      (angular_z proportional to the object's horizontal offset cx).
    * Target seen and roughly centered: drive forward.
    * Target seen and large/close enough (area >= stop_area): stop — arrived.

A watchdog stops the rover if no detection message arrives for 'lost_timeout'
seconds, so a crashed perception node can never leave the wheels running.

PARAMETERS
    target_class (str)     : object to approach; '' = approach any object
    forward_speed (float)  : linear.x when driving toward it   [m/s] (0.15)
    turn_gain (float)      : how hard to steer per unit cx offset      (1.2)
    max_angular (float)    : angular.z clamp                   [rad/s] (1.5)
    center_tolerance (float): |cx| below this counts as centered       (0.15)
    stop_area (float)      : bbox area fraction that means "arrived"    (0.25)
    min_confidence (float) : ignore detections below this              (0.40)
    search (bool)          : rotate slowly when nothing is seen        (False)
    search_speed (float)   : angular.z while searching        [rad/s]  (0.4)
    lost_timeout (float)   : stop if no detection for this long [s]    (1.0)
    publish_rate (float)   : /cmd_vel publish rate            [Hz]     (10.0)
"""

from __future__ import annotations

import json
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class ActionNode(Node):
    def __init__(self):
        super().__init__('action_node')

        self.declare_parameter('target_class', '')
        self.declare_parameter('forward_speed', 0.15)
        self.declare_parameter('turn_gain', 1.2)
        self.declare_parameter('max_angular', 1.5)
        self.declare_parameter('center_tolerance', 0.15)
        self.declare_parameter('stop_area', 0.25)
        self.declare_parameter('min_confidence', 0.40)
        self.declare_parameter('search', False)
        self.declare_parameter('search_speed', 0.4)
        self.declare_parameter('lost_timeout', 1.0)
        self.declare_parameter('publish_rate', 10.0)

        g = self.get_parameter
        self.target_class = g('target_class').value or ''
        self.forward_speed = float(g('forward_speed').value)
        self.turn_gain = float(g('turn_gain').value)
        self.max_angular = float(g('max_angular').value)
        self.center_tolerance = float(g('center_tolerance').value)
        self.stop_area = float(g('stop_area').value)
        self.min_confidence = float(g('min_confidence').value)
        self.search = bool(g('search').value)
        self.search_speed = float(g('search_speed').value)
        self.lost_timeout = float(g('lost_timeout').value)
        rate = float(g('publish_rate').value)

        # The current command the timer keeps publishing until a new detection
        # updates it. Held so /cmd_vel is a steady stream, like keyboard_node.
        self._linear = 0.0
        self._angular = 0.0
        self._last_detection = 0.0

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub = self.create_subscription(
            String, '/perception/detection', self._on_detection, 10)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._publish)

        target = self.target_class or 'any object'
        self.get_logger().info(
            'action_node started; approaching %s, forward=%.2f m/s -> /cmd_vel'
            % (target, self.forward_speed))

    # -- decision ----------------------------------------------------------- #

    def _on_detection(self, msg: String) -> None:
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return

        label = d.get('label', 'none')
        confidence = float(d.get('confidence', 0.0))
        cx = float(d.get('cx', 0.0))
        area = float(d.get('area', 0.0))

        target_ok = (
            label != 'none'
            and confidence >= self.min_confidence
            and (not self.target_class or label == self.target_class)
        )

        if not target_ok:
            # Nothing to chase right now. The watchdog/_publish will search or
            # stop; here we just record that this frame had no valid target.
            self._set_no_target()
            return

        self._last_detection = time.time()

        if area >= self.stop_area:
            # Close enough — arrived. Stop.
            self._linear, self._angular = 0.0, 0.0
            self.get_logger().info(
                'reached %s (area=%.2f) — stopping' % (label, area),
                throttle_duration_sec=2.0)
            return

        # Steer toward the object: turn proportional to its horizontal offset.
        self._angular = _clamp(-self.turn_gain * cx, self.max_angular)
        # Only drive forward once it is roughly centered; otherwise turn in place
        # so we face it before advancing.
        self._linear = self.forward_speed if abs(cx) <= self.center_tolerance else 0.0

    def _set_no_target(self) -> None:
        if self.search:
            self._linear = 0.0
            self._angular = self.search_speed
        else:
            self._linear = 0.0
            self._angular = 0.0

    # -- steady command stream + watchdog ---------------------------------- #

    def _publish(self) -> None:
        # Watchdog: if perception has gone quiet, don't keep driving on a stale
        # command — search or stop.
        if (time.time() - self._last_detection) > self.lost_timeout:
            self._set_no_target()

        msg = Twist()
        msg.linear.x = float(self._linear)
        msg.angular.z = float(self._angular)
        self.pub.publish(msg)

    def destroy_node(self):
        # Command a full stop on the way out.
        try:
            self.pub.publish(Twist())
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ActionNode()
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
