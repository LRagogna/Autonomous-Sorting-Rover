"""perception_node: run YOLO on camera frames and publish what it sees.

This is the "brain's eyes" node. It subscribes to the camera feed, runs the
trained YOLO detector on each frame, and publishes the single most confident
detection so the action_node can decide how to drive.

    subscribes : /camera/image/compressed   (sensor_msgs/CompressedImage)
    publishes  : /perception/detection       (std_msgs/String, JSON)
                 /perception/object           (std_msgs/String, just the label)

The JSON on /perception/detection is self-describing so no custom ROS message
package is needed:

    {"label": "bit", "confidence": 0.87,
     "cx": -0.12, "cy": 0.03,    # object center offset from image center, -1..1
     "area": 0.08,               # bounding-box area as a fraction of the frame
     "count": 2,                 # how many objects were seen this frame
     "stamp": 1785799849.12}

When nothing is detected, label is "none" and count is 0. cx < 0 means the
object is to the LEFT of center, cx > 0 to the right — that is what lets the
action_node steer toward it.

MODEL
    Loads the trained weights (default models/active_model.pt). The class names
    come from the model itself, so this node needs no separate class list.

PARAMETERS
    model (str)        : weights path or name (default 'active_model.pt')
    conf (float)       : confidence cutoff 0-1               (default 0.35)
    imgsz (int)        : YOLO inference size                 (default 640)
    target_class (str) : if set, only report this class      (default '' = any)
    publish_rate (float): max detections/sec published; 0 = every frame (default 0)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def resolve_model(name_or_path: str) -> Path:
    """Find the weights file from a path or bare name, checking common locations."""
    p = Path(name_or_path).expanduser()
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        p,
        Path.cwd() / p,
        Path.home() / 'AutonomousRover' / 'models' / p.name,
        Path(__file__).resolve().parents[4] / 'models' / p.name,  # repo/models when run from source
    ]
    env = os.environ.get('ROVER_MODEL')
    if env:
        candidates.insert(0, Path(env).expanduser())
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find model '{name_or_path}'. Set the 'model' parameter to a "
        f"full path, put it at ~/AutonomousRover/models/{p.name}, or set "
        f"ROVER_MODEL. Tried: {[str(c) for c in candidates]}")


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.declare_parameter('model', 'active_model.pt')
        self.declare_parameter('conf', 0.35)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('target_class', '')
        self.declare_parameter('publish_rate', 0.0)

        self.conf = float(self.get_parameter('conf').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.target_class = self.get_parameter('target_class').value or ''
        rate = float(self.get_parameter('publish_rate').value)
        self._min_period = (1.0 / rate) if rate > 0 else 0.0
        self._last_pub = 0.0

        self._load_backend()

        model_path = resolve_model(self.get_parameter('model').value)
        self.get_logger().info(f'perception_node loading model: {model_path}')
        self.model = self._YOLO(str(model_path))

        self.det_pub = self.create_publisher(String, '/perception/detection', 10)
        self.obj_pub = self.create_publisher(String, '/perception/object', 10)
        self.sub = self.create_subscription(
            CompressedImage, '/camera/image/compressed', self._on_image, SENSOR_QOS)

        target = self.target_class or 'any'
        self.get_logger().info(
            'perception_node started; conf=%.2f target=%s -> /perception/detection'
            % (self.conf, target))

    def _load_backend(self) -> None:
        try:
            import cv2
            import numpy as np
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "perception_node needs ultralytics + opencv. Install them in this "
                "ROS environment:\n    pip install ultralytics opencv-python"
            ) from exc
        self._cv2, self._np, self._YOLO = cv2, np, YOLO

    def _on_image(self, msg: CompressedImage) -> None:
        now = time.time()
        if self._min_period and (now - self._last_pub) < self._min_period:
            return

        frame = self._cv2.imdecode(
            self._np.frombuffer(msg.data, dtype=self._np.uint8), self._cv2.IMREAD_COLOR)
        if frame is None:
            return
        height, width = frame.shape[:2]

        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
        result = results[0]
        names = result.names

        best = None          # (confidence, label, cx, cy, area)
        count = 0
        for box in result.boxes:
            label = names[int(box.cls[0])]
            if self.target_class and label != self.target_class:
                continue
            count += 1
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            cx = ((x1 + x2) / 2.0) / width * 2.0 - 1.0     # -1 (left) .. +1 (right)
            cy = ((y1 + y2) / 2.0) / height * 2.0 - 1.0
            area = ((x2 - x1) * (y2 - y1)) / (width * height)
            if best is None or confidence > best[0]:
                best = (confidence, label, cx, cy, area)

        if best is None:
            payload = {"label": "none", "confidence": 0.0, "cx": 0.0, "cy": 0.0,
                       "area": 0.0, "count": 0, "stamp": now}
        else:
            confidence, label, cx, cy, area = best
            payload = {"label": label, "confidence": round(confidence, 4),
                       "cx": round(cx, 4), "cy": round(cy, 4),
                       "area": round(area, 5), "count": count, "stamp": now}

        self.det_pub.publish(String(data=json.dumps(payload)))
        self.obj_pub.publish(String(data=payload["label"]))
        self._last_pub = now


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
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
