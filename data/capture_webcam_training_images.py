"""Capture labeled webcam images for classifier training.

Examples:

    python data/capture_webcam_training_images.py wrench
    python data/capture_webcam_training_images.py bit --auto-save

The script writes frames to:

    data/raw/photos/<label>/<session_name>/frame_000001.jpg

Press:

    s      save one frame
    a      toggle auto-save
    q/esc  quit
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "photos"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_CROP_SCALE = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture labeled webcam images into data/raw/photos."
    )
    parser.add_argument(
        "label",
        help="Object label to capture, for example wrench or bit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Raw photo root directory. Defaults to data/raw/photos.",
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="Session folder name. Defaults to webcam_<timestamp>.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV webcam index. Defaults to 0.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Camera frame width. Defaults to {DEFAULT_WIDTH}.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Camera frame height. Defaults to {DEFAULT_HEIGHT}.",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=DEFAULT_CROP_SCALE,
        help=(
            "Preview center crop guide as a fraction of the shorter frame side. "
            f"Defaults to {DEFAULT_CROP_SCALE}."
        ),
    )
    parser.add_argument(
        "--auto-save",
        action="store_true",
        help="Start with auto-save enabled.",
    )
    parser.add_argument(
        "--auto-every",
        type=int,
        default=10,
        help="Save every N frames when auto-save is enabled. Defaults to 10.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Stop after saving this many images.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=92,
        help="JPEG quality from 1 to 100. Defaults to 92.",
    )
    return parser.parse_args()


def require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "OpenCV is not installed. Run: pip install -r requirements.txt"
        ) from error

    return cv2


def sanitize_name(value: str) -> str:
    safe = []
    for character in value.strip().lower():
        if character.isalnum() or character in {"-", "_"}:
            safe.append(character)
        elif character.isspace():
            safe.append("_")
    name = "".join(safe).strip("_")
    if not name:
        raise ValueError("Label/session name must contain at least one letter or number.")
    return name


def make_session_name() -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"webcam_{timestamp}"


def draw_crop_guide(cv2, frame, crop_scale: float) -> None:
    height, width = frame.shape[:2]
    side = int(min(height, width) * crop_scale)
    side = max(1, min(side, height, width))
    center_x = width // 2
    center_y = height // 2
    x1 = max(0, center_x - side // 2)
    y1 = max(0, center_y - side // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 190, 0), 2)


def save_frame(cv2, frame, output_path: Path, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not write image: {output_path}")


def main() -> int:
    args = parse_args()

    try:
        if args.camera_index < 0:
            raise ValueError("--camera-index must be 0 or greater.")
        if args.width <= 0 or args.height <= 0:
            raise ValueError("--width and --height must be greater than 0.")
        if not 0 < args.crop_scale <= 1:
            raise ValueError("--crop-scale must be greater than 0 and at most 1.")
        if args.auto_every < 1:
            raise ValueError("--auto-every must be at least 1.")
        if args.max_images is not None and args.max_images < 1:
            raise ValueError("--max-images must be at least 1.")
        if not 1 <= args.quality <= 100:
            raise ValueError("--quality must be between 1 and 100.")

        cv2 = require_cv2()
        label = sanitize_name(args.label)
        session_name = sanitize_name(args.session_name or make_session_name())
        session_dir = args.output_dir / label / session_name

        camera = cv2.VideoCapture(args.camera_index)
        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {args.camera_index}. "
                "On macOS, allow Terminal or your Python app in "
                "System Settings > Privacy & Security > Camera."
            )

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        auto_save = args.auto_save
        saved_count = 0
        frame_count = 0

        print(f"Saving {label!r} images to: {session_dir}")
        print("Press s to save, a to toggle auto-save, q/esc to quit.")

        try:
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise RuntimeError("Could not read a frame from the webcam.")

                frame_count += 1
                should_save = auto_save and frame_count % args.auto_every == 0

                preview = frame.copy()
                draw_crop_guide(cv2, preview, args.crop_scale)
                status = (
                    f"{label} saved={saved_count} auto={'on' if auto_save else 'off'}"
                )
                cv2.rectangle(preview, (0, 0), (preview.shape[1], 36), (0, 90, 0), -1)
                cv2.putText(
                    preview,
                    status,
                    (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Capture Rover Training Images", preview)

                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    break
                if key == ord("a"):
                    auto_save = not auto_save
                    print(f"Auto-save {'enabled' if auto_save else 'disabled'}.")
                if key == ord("s"):
                    should_save = True

                if should_save:
                    saved_count += 1
                    output_path = session_dir / f"frame_{saved_count:06d}.jpg"
                    save_frame(cv2, frame, output_path, args.quality)
                    print(f"Saved {output_path}")

                if args.max_images is not None and saved_count >= args.max_images:
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()

    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Saved {saved_count} images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
