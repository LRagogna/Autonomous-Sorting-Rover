"""Show the webcam and draw a green box around any object YOLO recognizes.

HOW TO USE THIS FILE

1. Train the detector once (this makes models/yolo_detector.pt):

       python data/auto_label_frames.py --overwrite
       python ml/train_yolo.py

2. Run the live webcam detector:

       python src/desktop_yolo_detector.py

   or use the helper script:

       ./scripts/run_desktop_detector.sh

3. Hold a wrench or a bit in front of your webcam. When the model is confident,
   a green box with the object's name appears around it. Press q to quit.

WHAT THIS FILE DOES

This is the "simulator" for the rover's eyes, running on your laptop instead of
the Raspberry Pi:

- OpenCV opens your computer's webcam and reads one frame (picture) at a time.
- Each frame is sent to the trained YOLO model.
- YOLO returns a list of boxes it found, each with a name and a confidence
  score (how sure it is, from 0 to 1).
- We draw a green box and label for every detection above the confidence cutoff.
- The loop repeats fast, so the boxes seem to follow the object in real time.

USEFUL OPTIONS

    python src/desktop_yolo_detector.py --camera-index 1   # use a second camera
    python src/desktop_yolo_detector.py --conf 0.5         # only very sure boxes
    python src/desktop_yolo_detector.py --weights path.pt  # a different model
    python src/desktop_yolo_detector.py --headless         # no window; print only

MAC CAMERA PERMISSION NOTE

If macOS says camera access was denied, allow your terminal app under:

    System Settings > Privacy & Security > Camera
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


# This file lives in src/, so parents[1] is the project folder above src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The trained detector that ml/train_yolo.py wrote.
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "yolo_detector.pt"

# Green, in OpenCV's Blue-Green-Red color order.
GREEN = (0, 255, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live webcam YOLO detector that draws green boxes."
    )
    parser.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS),
        help="Path to the trained YOLO weights file (.pt).",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0,
        help="Which webcam to use. 0 is usually the built-in camera.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence cutoff (0-1). Lower shows more (but shakier) boxes.",
    )
    parser.add_argument(
        "--smooth-frames", type=int, default=6,
        help="Keep showing the last box for this many frames after it is lost. "
             "This stops the box from flickering on and off. 0 turns it off.",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Do not open a window. Just print detections to the terminal.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after this many frames (0 = run until you press q).",
    )
    return parser.parse_args()


def load_model(weights_path: str):
    """Load the trained YOLO model, with friendly errors if something is missing."""
    if not Path(weights_path).exists():
        sys.exit(
            f"Could not find the trained model: {weights_path}\n"
            "Train it first with:\n"
            "    python data/auto_label_frames.py --overwrite\n"
            "    python ml/train_yolo.py"
        )
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "The 'ultralytics' package is not installed.\n"
            "Install it with:\n"
            "    pip install -r requirements.txt"
        )
    return YOLO(weights_path)


def extract_detections(result) -> list[dict]:
    """Pull the boxes out of a YOLO result into a simple list we can reuse.

    Returning plain data (not drawing yet) lets us remember the last boxes for a
    few frames so they do not flicker on and off.
    """
    names = result.names  # {class_id: "wrench", ...}
    detections: list[dict] = []
    for box in result.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        name = names[int(box.cls[0])]
        detections.append(
            {"box": (x1, y1, x2, y2), "name": name, "conf": float(box.conf[0])}
        )
    return detections


def draw_detections(frame, detections: list[dict]) -> None:
    """Draw a green box + label for each detection in the list."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['name']} {det['conf']:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)

        # Draw a filled strip behind the text so it stays readable on any color.
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), GREEN, -1
        )
        cv2.putText(
            frame, label, (x1 + 2, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    model = load_model(args.weights)

    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        sys.exit(
            f"Could not open camera index {args.camera_index}.\n"
            "Try a different --camera-index, or check camera permissions."
        )

    print("Webcam detector running. Press q in the window to quit.")
    last_printed = ""
    frame_count = 0

    # Box smoothing: remember the most recent detections and keep drawing them
    # for a few frames after they disappear, so the box does not blink.
    remembered: list[dict] = []
    frames_since_seen = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera stopped sending frames.")
                break

            # verbose=False keeps the terminal quiet; we do our own printing.
            results = model.predict(frame, conf=args.conf, verbose=False)
            detections = extract_detections(results[0])

            if detections:
                remembered = detections
                frames_since_seen = 0
            else:
                frames_since_seen += 1

            # Draw the fresh boxes, or the remembered ones if we lost them only a
            # moment ago.
            if detections or frames_since_seen <= args.smooth_frames:
                draw_detections(frame, remembered)
                found = [det["name"] for det in remembered]
            else:
                found = []

            if args.headless:
                # Only print when the set of visible objects changes.
                summary = ", ".join(sorted(set(found))) if found else "(nothing)"
                if summary != last_printed:
                    print(f"Detected: {summary}")
                    last_printed = summary
            else:
                cv2.imshow("Rover YOLO detector (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        # Small pause so macOS fully releases the window before the script ends.
        time.sleep(0.1)


if __name__ == "__main__":
    main()
