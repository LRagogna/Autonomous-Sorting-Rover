"""Live YOLO detector for the Raspberry Pi camera, viewable in a web browser.

This is the rover's "eyes" running on the Pi itself: it reads frames from the
Pi camera, runs the trained YOLO model on each one, draws a box + label around
every object it recognizes, and streams the annotated video so you can watch it
from your laptop or phone.

WHY A WEB STREAM

The rover is usually headless (no monitor). Instead of opening a window on the
Pi, this serves an MJPEG video stream over the local network using only the
Python standard library. Open the printed URL in any browser on the same
Wi-Fi/LAN and you'll see the live camera with the detections drawn on top. If
the Pi *does* have an HDMI monitor, pass --window to show a local window too.

BEFORE YOU RUN IT (one-time setup on the Pi)

1. Picamera2 (ships with Raspberry Pi OS Bookworm; install if missing):

       sudo apt install -y python3-picamera2

2. The inference engine. The rover's runtime deps (requirements-pi.txt) do NOT
   include ultralytics, so install it on the Pi:

       pip install ultralytics

   (This pulls in torch + opencv. On a Pi 4/5 a YOLOv8n runs at a few FPS. For
   much faster inference, export the model to NCNN on your dev machine:
       yolo export model=models/active_model.pt format=ncnn
   then copy the resulting *_ncnn_model/ folder to the Pi and point --weights
   at it. This script loads .pt and NCNN/ONNX exports the same way.)

3. The trained model. models/ is kept off the Pi, so copy the active model over
   from your dev machine, e.g.:

       scp deploy/active_model.pt  pi@<pi-host>:~/AutonomousRover/models/

USAGE

    # On the Pi:
    python3 src/pi_yolo_detector.py
    # then open the printed http://<pi-ip>:8000/ URL in a browser.

    python3 src/pi_yolo_detector.py --conf 0.4          # only surer boxes
    python3 src/pi_yolo_detector.py --port 9000         # different port
    python3 src/pi_yolo_detector.py --window            # also show a local window
    python3 src/pi_yolo_detector.py --camera usb        # a USB webcam, not the CSI cam
    python3 src/pi_yolo_detector.py --weights models/yolo_detector_final.pt
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Prefer the deploy bundle's model, fall back to the active model in models/.
_DEPLOY_MODEL = PROJECT_ROOT / "deploy" / "active_model.pt"
_ACTIVE_MODEL = PROJECT_ROOT / "models" / "active_model.pt"
DEFAULT_WEIGHTS = _DEPLOY_MODEL if _DEPLOY_MODEL.exists() else _ACTIVE_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live Pi-camera YOLO detector streamed to a web browser."
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="Trained YOLO weights: a .pt file or an exported NCNN/ONNX model.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence cutoff (0-1). Lower shows more (but shakier) boxes.")
    parser.add_argument("--width", type=int, default=640, help="Camera frame width (default 640).")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height (default 480).")
    parser.add_argument("--fps", type=int, default=30, help="Camera capture frame rate (default 30).")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size (default 640).")
    parser.add_argument("--host", default="0.0.0.0", help="Address to serve on (default all interfaces).")
    parser.add_argument("--port", type=int, default=8000, help="Web stream port (default 8000).")
    parser.add_argument("--camera", choices=["picamera2", "usb"], default="picamera2",
                        help="Camera backend: the CSI cam (picamera2) or a USB webcam (usb).")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="USB webcam index when --camera usb (0 is usually the first).")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                        help="Rotate the image if the camera is mounted sideways/upside-down.")
    parser.add_argument("--window", action="store_true",
                        help="Also show a local OpenCV window (needs an HDMI monitor / desktop).")
    parser.add_argument("--jpeg-quality", type=int, default=80,
                        help="Streamed JPEG quality 1-100 (default 80).")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Model + camera
# --------------------------------------------------------------------------- #

def load_model(weights_path: str):
    """Load the trained YOLO model, with friendly errors if something is missing."""
    if not Path(weights_path).exists():
        sys.exit(
            f"Could not find the model: {weights_path}\n"
            "Copy the active model onto the Pi, e.g.:\n"
            "    scp deploy/active_model.pt  pi@<pi-host>:~/AutonomousRover/models/\n"
            "or point --weights at a model that exists on this Pi."
        )
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "The 'ultralytics' package is not installed on this Pi.\n"
            "Install it with:\n"
            "    pip install ultralytics"
        )
    return YOLO(weights_path)


class PiCamera:
    """Frame source for the CSI camera (Picamera2) with a USB-webcam fallback."""

    def __init__(self, args: argparse.Namespace):
        self.kind = args.camera
        self.width, self.height = args.width, args.height
        self._cv2 = None
        self._picam = None
        if self.kind == "usb":
            self._open_usb(args.camera_index)
        else:
            self._open_picamera2(args.fps)

    def _open_picamera2(self, fps: int) -> None:
        try:
            from picamera2 import Picamera2
        except ModuleNotFoundError:
            sys.exit(
                "Picamera2 is not available. Run this on the Raspberry Pi, and if needed:\n"
                "    sudo apt install -y python3-picamera2\n"
                "Or use a USB webcam with:  --camera usb"
            )
        self._picam = Picamera2()
        config = self._picam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self._picam.configure(config)
        self._picam.start()
        time.sleep(0.5)  # let auto-exposure/white-balance settle

    def _open_usb(self, index: int) -> None:
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            sys.exit(f"Could not open USB camera index {index}. Try a different --camera-index.")

    def read(self):
        """Return one BGR frame (OpenCV order), or None if the camera stopped."""
        if self._picam is not None:
            import cv2
            # Picamera2 'RGB888' hands back bytes in B,G,R order already (a known
            # libcamera quirk), so this array is effectively BGR for OpenCV/YOLO.
            frame = self._picam.capture_array()
            if frame.shape[2] == 4:  # drop alpha if present
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._picam is not None:
            self._picam.stop()
            self._picam.close()
        if self._cv2 is not None:
            self._cap.release()


# --------------------------------------------------------------------------- #
# Detection loop (runs in a background thread, publishes the latest JPEG)
# --------------------------------------------------------------------------- #

class Detector:
    """Grabs frames, runs YOLO, and keeps the newest annotated JPEG for the stream."""

    def __init__(self, model, camera: PiCamera, args: argparse.Namespace):
        self.model = model
        self.camera = camera
        self.args = args
        self.latest_jpeg: bytes | None = None
        self.fps = 0.0
        self.running = True
        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)

    def _rotate(self, frame, cv2):
        codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        return cv2.rotate(frame, codes[self.args.rotate]) if self.args.rotate else frame

    def run(self) -> None:
        import cv2
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(self.args.jpeg_quality)]
        smoothed_fps = 0.0
        while self.running:
            frame = self.camera.read()
            if frame is None:
                print("Camera stopped sending frames.", file=sys.stderr)
                break
            frame = self._rotate(frame, cv2)

            start = time.time()
            results = self.model.predict(
                frame, conf=self.args.conf, imgsz=self.args.imgsz, verbose=False
            )
            # result.plot() returns the frame with boxes + labels already drawn.
            annotated = results[0].plot()

            dt = time.time() - start
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            smoothed_fps = inst_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * inst_fps
            cv2.putText(annotated, f"{smoothed_fps:4.1f} FPS", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            ok, buf = cv2.imencode(".jpg", annotated, encode_params)
            if ok:
                with self._new:
                    self.latest_jpeg = buf.tobytes()
                    self.fps = smoothed_fps
                    self._new.notify_all()

            if self.args.window:
                cv2.imshow("Rover Pi YOLO detector (press q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.running = False

        self.running = False
        with self._new:
            self._new.notify_all()

    def wait_for_jpeg(self, timeout: float = 5.0) -> bytes | None:
        """Block until a newer frame is ready, then return it."""
        with self._new:
            self._new.wait(timeout)
            return self.latest_jpeg


# --------------------------------------------------------------------------- #
# Web stream (stdlib only)
# --------------------------------------------------------------------------- #

_PAGE = b"""<!doctype html><html><head><title>Rover Pi Detector</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;text-align:center}
h1{font-size:1rem;padding:.6rem;margin:0;background:#000}
img{max-width:100%;height:auto}</style></head>
<body><h1>Rover Pi YOLO detector &mdash; live</h1>
<img src="/stream.mjpg" alt="live detector stream"></body></html>"""


def make_handler(detector: Detector):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):  # silence per-request console spam
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(_PAGE)))
                self.end_headers()
                self.wfile.write(_PAGE)
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
                )
                self.end_headers()
                try:
                    while detector.running:
                        jpeg = detector.wait_for_jpeg()
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg)))
                        self.end_headers()
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # viewer closed the tab; that's fine
            else:
                self.send_error(404)

    return Handler


def local_ip() -> str:
    """Best-effort LAN IP so the printed URL is clickable from another device."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> int:
    args = parse_args()
    model = load_model(args.weights)
    camera = PiCamera(args)
    detector = Detector(model, camera, args)

    worker = threading.Thread(target=detector.run, daemon=True)
    worker.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(detector))
    url = f"http://{local_ip()}:{args.port}/"
    print(f"Pi detector running. Model: {args.weights}")
    print(f"Open this in a browser on the same network:\n    {url}")
    print("Press Ctrl-C to stop.")

    try:
        while detector.running:
            server.handle_request()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.running = False
        server.server_close()
        worker.join(timeout=2)
        camera.close()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
