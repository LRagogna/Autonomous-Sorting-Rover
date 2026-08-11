"""camera_node: publish camera frames as sensor_msgs/CompressedImage.

This is the "eyes" node of the rover's perception pipeline. It grabs frames
from a camera and publishes them as JPEG-compressed images on

    /camera/image/compressed   (sensor_msgs/CompressedImage)

so the perception_node can decode them and run object detection. Compressed
images are used (instead of raw sensor_msgs/Image) so the node needs no
cv_bridge and the topic stays light enough to view over the network.

CAMERA SOURCES (parameter 'source')

    picamera2 : the Raspberry Pi CSI camera (default; the rover's real camera)
    usb       : a USB webcam via OpenCV (handy for development on a laptop)
    video     : loop a video file via OpenCV (fully offline development/testing)

Only 'picamera2' needs the Pi. 'usb' and 'video' let you develop the whole
pipeline on your Mac with no rover attached.

PARAMETERS
    source (str)      : 'picamera2' | 'usb' | 'video'   (default 'picamera2')
    video_path (str)  : path to a video file when source='video'
    usb_index (int)   : OpenCV camera index when source='usb'   (default 0)
    width, height     : capture size                            (default 640x480)
    fps (float)       : publish rate                            (default 15.0)
    jpeg_quality (int): 1-100                                   (default 80)
    rotate (int)      : 0/90/180/270 if the camera is mounted rotated
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage


# Best-effort, keep-last-1: for a live video feed we want the freshest frame,
# not a reliable backlog of stale ones.
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('source', 'picamera2')
        self.declare_parameter('video_path', '')
        self.declare_parameter('usb_index', 0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('rotate', 0)

        self.source = self.get_parameter('source').value
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = float(self.get_parameter('fps').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.rotate = int(self.get_parameter('rotate').value)

        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "camera_node needs OpenCV. Install it in this ROS environment:\n"
                "    pip install opencv-python"
            ) from exc

        self._picam = None
        self._cap = None
        self._open_source()

        self.publisher = self.create_publisher(
            CompressedImage, '/camera/image/compressed', SENSOR_QOS)
        self.timer = self.create_timer(1.0 / max(self.fps, 1.0), self._tick)
        self._frame_id = 0

        self.get_logger().info(
            "camera_node started; source=%s %dx%d @ %.1f fps -> "
            "/camera/image/compressed" % (self.source, self.width, self.height, self.fps))

    # -- source setup ------------------------------------------------------- #

    def _open_source(self) -> None:
        if self.source == 'picamera2':
            self._open_picamera2()
        elif self.source == 'usb':
            self._open_usb(int(self.get_parameter('usb_index').value))
        elif self.source == 'video':
            self._open_video(self.get_parameter('video_path').value)
        else:
            raise RuntimeError(f"Unknown source '{self.source}' (use picamera2|usb|video).")

    def _open_picamera2(self) -> None:
        try:
            from picamera2 import Picamera2
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Picamera2 not available. Run on the Pi (sudo apt install -y "
                "python3-picamera2), or use source:=usb / source:=video for "
                "development."
            ) from exc
        self._picam = Picamera2()
        config = self._picam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"})
        self._picam.configure(config)
        self._picam.start()

    def _open_usb(self, index: int) -> None:
        import cv2
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open USB camera index {index}.")

    def _open_video(self, path: str) -> None:
        import cv2
        if not path:
            raise RuntimeError("source='video' requires the 'video_path' parameter.")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        self._video_path = path

    # -- frame grab + publish ---------------------------------------------- #

    def _read_frame(self):
        """Return one BGR frame (OpenCV order) or None."""
        import cv2
        if self._picam is not None:
            # Picamera2 'RGB888' returns bytes already in B,G,R order for OpenCV.
            frame = self._picam.capture_array()
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        ok, frame = self._cap.read()
        if not ok:
            # Loop video files so development feeds run continuously.
            if self.source == 'video':
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                return None
        return frame

    def _tick(self) -> None:
        import cv2
        frame = self._read_frame()
        if frame is None:
            self.get_logger().warn('camera_node: no frame from source', throttle_duration_sec=5.0)
            return

        if self.rotate:
            codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                     270: cv2.ROTATE_90_COUNTERCLOCKWISE}
            frame = cv2.rotate(frame, codes[self.rotate])

        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        self.publisher.publish(msg)
        self._frame_id += 1

    def destroy_node(self):
        try:
            if self._picam is not None:
                self._picam.stop()
                self._picam.close()
            if self._cap is not None:
                self._cap.release()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
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
