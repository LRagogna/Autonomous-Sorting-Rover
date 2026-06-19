"""Build image datasets by extracting frames from object-specific videos.

Expected input layout:
    data/videos/<object_type>/<video_file>

Example:
    data/videos/washer/pan_01.mp4

Expected output layout:
    data/photos/<object_type>/<generated_frame_images>

The folder name under data/videos is treated as the object label. This keeps
video collection simple: record a few videos of an object, put them in that
object's folder, then run this script to create raw training images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
DEFAULT_PHOTO_DIR = PROJECT_ROOT / "data" / "photos"


def parse_args() -> argparse.Namespace:
    """Read command-line options for one video extraction run."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from data/videos/<object_type>/<video> into "
            "data/photos/<object_type>/ for ML processing."
        )
    )
    parser.add_argument(
        "video",
        help=(
            "Video path inside data/videos/, such as washer/pan_01.mp4, "
            "or a path to a video file."
        ),
    )
    parser.add_argument(
        "--object-type",
        default=None,
        help=(
            "Override the output object label. By default, this is inferred "
            "from the video's folder name."
        ),
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Save every Nth frame. Defaults to 1.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to consider. Defaults to 0.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of images to save.",
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "jpg", "jpeg"),
        default="png",
        help="Output image format. Defaults to png.",
    )
    return parser.parse_args()


def resolve_video_path(video_arg: str) -> Path:
    """Resolve user input into an actual video path.

    Common use:
        washer/pan_01.mp4 -> data/videos/washer/pan_01.mp4

    A path starting with data/ is treated as project-relative, and an absolute
    path is used exactly as given.
    """
    video_path = Path(video_arg).expanduser()

    if video_path.is_absolute():
        return video_path

    if video_path.parts[0] == "data":
        return (PROJECT_ROOT / video_path).resolve()

    return (DEFAULT_VIDEO_DIR / video_path).resolve()


def infer_object_type(video_path: Path) -> str:
    """Infer the object label from the first folder under data/videos."""
    try:
        relative_path = video_path.resolve().relative_to(DEFAULT_VIDEO_DIR.resolve())
    except ValueError as error:
        raise ValueError(
            "Could not infer object_type from a video outside data/videos/. "
            "Pass --object-type explicitly."
        ) from error

    if len(relative_path.parts) < 2:
        raise ValueError(
            "Video must be inside an object folder like data/videos/washer/video.mp4, "
            "or pass --object-type explicitly."
        )

    return relative_path.parts[0]


def validate_object_type(object_type: str) -> str:
    """Keep object labels safe to use as output folder names."""
    output_path = Path(object_type)

    if output_path.is_absolute() or ".." in output_path.parts:
        raise ValueError("object_type must be a folder name, not an absolute or parent path.")

    if not object_type.strip():
        raise ValueError("object_type cannot be empty.")

    return object_type.strip().replace(" ", "_")


def extract_frames(
    video_path: Path,
    object_type: str,
    frame_step: int,
    start_frame: int,
    max_frames: int | None,
    image_format: str,
) -> int:
    """Read a video with OpenCV and save selected frames as image files."""
    if frame_step < 1:
        raise ValueError("--frame-step must be 1 or greater.")

    if start_frame < 0:
        raise ValueError("--start-frame must be 0 or greater.")

    if max_frames is not None and max_frames < 1:
        raise ValueError("--max-frames must be 1 or greater when provided.")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Import OpenCV only when extraction starts. This lets --help and simple
    # argument checks work even before dependencies are installed.
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "OpenCV is not installed. Run: pip install -r requirements.txt"
        ) from error

    object_type = validate_object_type(object_type)
    output_dir = DEFAULT_PHOTO_DIR / object_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # VideoCapture reads the video sequentially, one frame at a time.
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    saved_count = 0
    frame_index = 0
    video_stem = video_path.stem
    extension = "jpg" if image_format == "jpeg" else image_format

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # frame_step controls dataset density. For example, frame_step=10
            # saves frame 0, 10, 20, ... instead of every single frame.
            should_save = frame_index >= start_frame and (
                frame_index - start_frame
            ) % frame_step == 0

            if should_save:
                image_name = f"{object_type}_{video_stem}_frame_{frame_index:06d}.{extension}"
                image_path = output_dir / image_name

                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError(f"Could not write image: {image_path}")

                saved_count += 1

                if max_frames is not None and saved_count >= max_frames:
                    break

            frame_index += 1
    finally:
        # Always release the video handle, including when writing a frame fails.
        capture.release()

    return saved_count


def main() -> int:
    """Run the command-line workflow and return a shell-friendly exit code."""
    args = parse_args()
    video_path = resolve_video_path(args.video)
    object_type = args.object_type

    try:
        if object_type is None:
            object_type = infer_object_type(video_path)

        saved_count = extract_frames(
            video_path=video_path,
            object_type=object_type,
            frame_step=args.frame_step,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            image_format=args.image_format,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Saved {saved_count} image(s) to {DEFAULT_PHOTO_DIR / validate_object_type(object_type)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
