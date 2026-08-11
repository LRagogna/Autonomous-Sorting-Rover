"""Mac webcam -> YOLO -> drive instructions written to files (no ROS needed).

This is a laptop "simulator" of the rover's autonomous loop. It uses your Mac
webcam as the camera, runs the trained detector to decide what object is in
view and where, applies the SAME steering policy as the ROS action_node, and
writes the resulting drive instruction to files instead of driving motors.

It reuses the real differential-drive kinematics from the ROS package, so the
left/right motor values here match what the rover would actually command.

WHAT IT WRITES (default: log/drive/)

    drive_command.json   overwritten every frame with the latest command:
        {"action": "FORWARD", "label": "bit", "confidence": 0.87,
         "cx": -0.05, "area": 0.09,
         "linear_x": 0.15, "angular_z": 0.06,     # the Twist (v, omega)
         "left": 54.0, "right": 51.5,             # motor values (-255..255)
         "timestamp": 1785807099.65}

    drive_log.csv        one appended row per frame (a full history you can plot)

Other programs (or you) can watch drive_command.json to see, in real time, what
the rover would do as you hold objects in front of the webcam.

USAGE

    python src/webcam_drive.py                    # webcam -> files (+ console)
    python src/webcam_drive.py --window           # also show the video with boxes
    python src/webcam_drive.py --target-class bit # only chase 'bit'
    python src/webcam_drive.py --out /tmp/rover    # write the files elsewhere
    python src/webcam_drive.py --video clip.mp4    # use a file instead of webcam

Press q to quit (in --window mode), or Ctrl-C in the terminal.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "active_model.pt"
DEFAULT_OUT = PROJECT_ROOT / "log" / "drive"

# Reuse the rover's real differential-drive kinematics from the ROS package so
# the motor values written here match the hardware command.
_DD_DIR = PROJECT_ROOT / "ros2_ws" / "src" / "rover_control" / "rover_control"
sys.path.insert(0, str(_DD_DIR))
try:
    from differential_drive import DriveConfig, twist_to_motors
    _HAVE_DD = True
except Exception:  # pragma: no cover - fall back to a passthrough if unavailable
    _HAVE_DD = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Webcam -> YOLO -> drive instructions in files.")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO weights (.pt).")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output folder for the instruction files.")
    p.add_argument("--camera-index", type=int, default=0, help="Webcam index (0 = built-in).")
    p.add_argument("--video", default="", help="Use a video file instead of the webcam.")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence cutoff.")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference size.")
    p.add_argument("--target-class", default="", help="Only chase this class ('' = any object).")
    # Steering policy (matches ROS action_node defaults).
    p.add_argument("--forward-speed", type=float, default=0.15, help="linear.x toward target [m/s].")
    p.add_argument("--turn-gain", type=float, default=1.2, help="steer strength per unit cx.")
    p.add_argument("--max-angular", type=float, default=1.5, help="angular.z clamp [rad/s].")
    p.add_argument("--center-tolerance", type=float, default=0.15, help="|cx| that counts as centered.")
    p.add_argument("--stop-area", type=float, default=0.25, help="bbox area fraction meaning 'arrived'.")
    p.add_argument("--min-confidence", type=float, default=0.40, help="ignore detections below this.")
    p.add_argument("--search", action="store_true", help="rotate to search when nothing is seen.")
    p.add_argument("--search-speed", type=float, default=0.4, help="angular.z while searching [rad/s].")
    p.add_argument("--window", action="store_true", help="show the video with boxes + decision.")
    return p.parse_args()


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def decide(best, args) -> tuple[str, float, float]:
    """Same policy as the ROS action_node. Returns (action, linear_x, angular_z)."""
    if best is None:
        if args.search:
            return "SEARCHING", 0.0, args.search_speed
        return "STOP (no target)", 0.0, 0.0

    _conf, _label, cx, _cy, area = best
    if area >= args.stop_area:
        return "ARRIVED (stop)", 0.0, 0.0

    angular = _clamp(-args.turn_gain * cx, args.max_angular)
    if abs(cx) <= args.center_tolerance:
        return "FORWARD", args.forward_speed, angular
    return ("TURN LEFT" if cx < 0 else "TURN RIGHT"), 0.0, angular


def pick_best(result, target_class):
    """Return the highest-confidence detection as (conf, label, cx, cy, area), or None."""
    names = result.names
    best = None
    for box in result.boxes:
        label = names[int(box.cls[0])]
        if target_class and label != target_class:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        h, w = result.orig_shape
        cx = ((x1 + x2) / 2.0) / w * 2.0 - 1.0
        cy = ((y1 + y2) / 2.0) / h * 2.0 - 1.0
        area = ((x2 - x1) * (y2 - y1)) / (w * h)
        if best is None or conf > best[0]:
            best = (conf, label, cx, cy, area)
    return best


def main() -> int:
    args = parse_args()

    if not Path(args.weights).exists():
        sys.exit(f"Model not found: {args.weights}\nTrain/activate one first.")
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install -r requirements.txt")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "drive_command.json"
    log_path = out_dir / "drive_log.csv"

    cfg = DriveConfig() if _HAVE_DD else None

    model = YOLO(args.weights)
    source = args.video if args.video else args.camera_index
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open {'video ' + args.video if args.video else f'webcam {args.camera_index}'}.")

    # CSV header (only if the file is new/empty).
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    log_file = log_path.open("a", newline="")
    writer = csv.writer(log_file)
    if write_header:
        writer.writerow(["timestamp", "action", "label", "confidence", "cx", "area",
                         "linear_x", "angular_z", "left", "right"])

    print(f"Webcam drive sim running. Writing instructions to:\n  {latest_path}\n  {log_path}")
    print("Hold an object in front of the camera. Ctrl-C (or q in --window) to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video:  # loop the file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("Camera stopped.")
                break

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            best = pick_best(result, args.target_class)
            # Gate on confidence like the action_node does.
            if best is not None and best[0] < args.min_confidence:
                best = None

            action, linear_x, angular_z = decide(best, args)
            if cfg is not None:
                left, right = twist_to_motors(linear_x, angular_z, cfg)
            else:
                left, right = linear_x, angular_z  # no kinematics available

            ts = time.time()
            label = best[1] if best else "none"
            conf = round(best[0], 4) if best else 0.0
            cx = round(best[2], 4) if best else 0.0
            area = round(best[4], 5) if best else 0.0

            command = {
                "action": action, "label": label, "confidence": conf,
                "cx": cx, "area": area,
                "linear_x": round(linear_x, 4), "angular_z": round(angular_z, 4),
                "left": round(left, 1), "right": round(right, 1),
                "timestamp": ts,
            }
            # Atomic-ish overwrite of the latest command.
            tmp = latest_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(command, indent=2))
            tmp.replace(latest_path)
            writer.writerow([ts, action, label, conf, cx, area,
                             command["linear_x"], command["angular_z"],
                             command["left"], command["right"]])
            log_file.flush()

            sys.stdout.write(
                f"\r  {action:<16} | {label:<11} cx={cx:+.2f} area={area:.2f}"
                f" -> L={command['left']:+6.1f} R={command['right']:+6.1f}   ")
            sys.stdout.flush()

            if args.window:
                annotated = result.plot()
                cv2.putText(annotated, f"{action}  L={command['left']:+.0f} R={command['right']:+.0f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Rover webcam drive sim (press q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        # Leave a final STOP command in the file so nothing downstream keeps driving.
        latest_path.write_text(json.dumps({
            "action": "STOP (shutdown)", "label": "none", "confidence": 0.0,
            "cx": 0.0, "area": 0.0, "linear_x": 0.0, "angular_z": 0.0,
            "left": 0.0, "right": 0.0, "timestamp": time.time()}, indent=2))
        print("\nStopped. Wrote a final STOP command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
