"""viz_node: one OpenCV window showing what the rover sees AND what it will do.

This is the "watch it think" node for testing on the computer. It overlays, on
the live camera feed, everything the autonomous loop is deciding right now:

    * the detected object's box + label + confidence
    * the LEFT / RIGHT motor values the wheels are being commanded (from /cmd_vel,
      run through the SAME differential_drive math fake_motor_node uses)
    * a big arrow showing turn direction + forward/back/stop
    * a small odometry mini-map (top-right) tracing the estimated path from /odom

    subscribes : /camera/image/compressed   (sensor_msgs/CompressedImage)
                 /perception/detection        (std_msgs/String, JSON)
                 /cmd_vel                      (geometry_msgs/Twist)
                 /odom                         (nav_msgs/Odometry)

It publishes nothing — it is a monitor. Because it opens a GUI window it needs a
display, so run it on the development computer, not a headless Pi.

NOTE ON THE DETECTION BOX: /perception/detection carries the object's center
(cx, cy) and area as fractions of the frame, not a pixel box. viz_node
reconstructs an approximate square box from the area for display; it is a visual
aid, not the exact detector box.

PARAMETERS
    window_name (str) : OpenCV window title              (default 'rover view')
    map_size (int)    : odom mini-map side in pixels     (default 180)
    map_scale (float) : pixels per meter on the mini-map (default 120.0)
"""

from __future__ import annotations

import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from rover_control.differential_drive import DriveConfig, twist_to_motors


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class VizNode(Node):
    def __init__(self):
        super().__init__('viz_node')

        self.declare_parameter('window_name', 'rover view')
        self.declare_parameter('map_size', 180)
        self.declare_parameter('map_scale', 120.0)
        self.window = self.get_parameter('window_name').value
        self.map_size = int(self.get_parameter('map_size').value)
        self.map_scale = float(self.get_parameter('map_scale').value)

        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "viz_node needs opencv + numpy. Install them in this ROS "
                "environment:\n    pip install opencv-python numpy"
            ) from exc
        self._cv2, self._np = cv2, np

        self.cfg = DriveConfig()          # default geometry for the motor readout
        self._frame = None                # latest decoded BGR frame
        self._det = {"label": "none", "confidence": 0.0, "cx": 0.0, "cy": 0.0,
                     "area": 0.0, "count": 0}
        self._v = 0.0
        self._w = 0.0
        self._path = []                   # list of (x, y) from /odom for the mini-map

        self.create_subscription(
            CompressedImage, '/camera/image/compressed', self._on_image, SENSOR_QOS)
        self.create_subscription(String, '/perception/detection', self._on_det, 10)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        # Render on a timer in the main (spinning) thread so imshow is happy.
        self.timer = self.create_timer(1.0 / 30.0, self._render)
        self.get_logger().info("viz_node started; showing window '%s'" % self.window)

    # -- inputs ------------------------------------------------------------- #

    def _on_image(self, msg: CompressedImage) -> None:
        buf = self._np.frombuffer(msg.data, dtype=self._np.uint8)
        frame = self._cv2.imdecode(buf, self._cv2.IMREAD_COLOR)
        if frame is not None:
            self._frame = frame

    def _on_det(self, msg: String) -> None:
        try:
            self._det = json.loads(msg.data)
        except (ValueError, TypeError):
            pass

    def _on_cmd(self, msg: Twist) -> None:
        self._v = float(msg.linear.x)
        self._w = float(msg.angular.z)

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self._path.append((p.x, p.y))
        if len(self._path) > 2000:
            self._path = self._path[-2000:]

    # -- drawing ------------------------------------------------------------ #

    def _render(self) -> None:
        cv2 = self._cv2
        if self._frame is None:
            return
        frame = self._frame.copy()
        h, w = frame.shape[:2]

        self._draw_detection(frame, w, h)
        self._draw_wheels(frame, w, h)
        self._draw_minimap(frame, w)

        cv2.imshow(self.window, frame)
        # A crash-safe quit path when running viz standalone.
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            self.get_logger().info('viz_node: quit key pressed, shutting down.')
            rclpy.shutdown()

    def _draw_detection(self, frame, w, h) -> None:
        cv2 = self._cv2
        d = self._det
        label = d.get('label', 'none')
        if label == 'none' or d.get('area', 0.0) <= 0.0:
            cv2.putText(frame, 'no target', (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 220), 2)
            return
        cx, cy, area = d.get('cx', 0.0), d.get('cy', 0.0), d.get('area', 0.0)
        # cx,cy in [-1,1] -> pixels; approximate a square box from the area fraction.
        px = int((cx + 1.0) / 2.0 * w)
        py = int((cy + 1.0) / 2.0 * h)
        side = math.sqrt(max(area, 0.0))
        bw = int(side * w / 2.0)
        bh = int(side * h / 2.0)
        cv2.rectangle(frame, (px - bw, py - bh), (px + bw, py + bh), (0, 220, 0), 2)
        cv2.circle(frame, (px, py), 4, (0, 220, 0), -1)
        cv2.line(frame, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)  # image center
        text = '%s %.0f%%' % (label, 100.0 * d.get('confidence', 0.0))
        cv2.putText(frame, text, (px - bw, max(py - bh - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

    def _draw_wheels(self, frame, w, h) -> None:
        cv2 = self._cv2
        left, right = twist_to_motors(self._v, self._w, self.cfg)

        # Action word from the commanded Twist.
        if abs(self._v) < 1e-3 and abs(self._w) < 1e-3:
            action, color = 'STOP', (60, 60, 220)
        elif abs(self._w) >= 0.05 and abs(self._v) < 1e-3:
            action = 'TURN LEFT' if self._w > 0 else 'TURN RIGHT'
            color = (0, 200, 220)
        elif self._v > 0:
            action = 'FORWARD' + ('  <-' if self._w > 0.05 else ('  ->' if self._w < -0.05 else ''))
            color = (0, 220, 0)
        else:
            action, color = 'REVERSE', (0, 140, 220)

        bar_h = 70
        y0 = h - bar_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(frame, action, (12, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame,
                    'LEFT %+6.1f   RIGHT %+6.1f   (v=%+.2f w=%+.2f)'
                    % (left, right, self._v, self._w),
                    (12, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

        # Wheel strength bars on the right edge.
        self._wheel_bar(frame, w - 60, y0 + 10, bar_h - 20, left)
        self._wheel_bar(frame, w - 30, y0 + 10, bar_h - 20, right)

    def _wheel_bar(self, frame, x, y, height, value) -> None:
        cv2 = self._cv2
        frac = max(-1.0, min(1.0, value / max(self.cfg.max_motor, 1.0)))
        mid = y + height // 2
        cv2.rectangle(frame, (x, y), (x + 18, y + height), (120, 120, 120), 1)
        top = mid - int(frac * (height // 2))
        c = (0, 220, 0) if frac >= 0 else (0, 140, 220)
        cv2.rectangle(frame, (x, min(mid, top)), (x + 18, max(mid, top)), c, -1)

    def _draw_minimap(self, frame, w) -> None:
        cv2 = self._cv2
        np = self._np
        s = self.map_size
        pad = 10
        x0, y0 = w - s - pad, pad
        panel = np.full((s, s, 3), 30, dtype=np.uint8)
        cx, cy = s // 2, s // 2
        cv2.line(panel, (cx, 0), (cx, s), (60, 60, 60), 1)
        cv2.line(panel, (0, cy), (s, cy), (60, 60, 60), 1)

        if self._path:
            pts = []
            for (px, py) in self._path:
                mx = int(cx + px * self.map_scale)
                my = int(cy - py * self.map_scale)   # y up on the map
                if 0 <= mx < s and 0 <= my < s:
                    pts.append((mx, my))
            if len(pts) >= 2:
                cv2.polylines(panel, [np.array(pts, dtype=np.int32)], False, (0, 220, 0), 2)
            if pts:
                cv2.circle(panel, pts[-1], 4, (0, 0, 220), -1)   # current position
            x, y = self._path[-1]
            cv2.putText(panel, 'x=%.2f y=%.2f' % (x, y), (6, s - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(panel, 'odom', (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        frame[y0:y0 + s, x0:x0 + s] = panel

    def destroy_node(self):
        try:
            self._cv2.destroyAllWindows()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VizNode()
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
