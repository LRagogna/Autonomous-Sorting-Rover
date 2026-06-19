"""Turn short object videos into photo folders for machine learning.

HOW TO USE THIS FILE

1. Put your videos in this folder shape:

       data/raw/videos/<object_type>/<video_file>

   Example:

       data/raw/videos/washer/pan_01.mp4

2. Extract photos from one video:

       python data/extract_video_frames.py washer pan_01.mp4 --frame-step 15

   This creates:

       data/raw/photos/washer/pan_01.mp4/frame_000000.png
       data/raw/photos/washer/pan_01.mp4/frame_000015.png
       data/raw/photos/washer/pan_01.mp4/frame_000030.png

3. Extract photos from every video:

       python data/extract_video_frames.py --all --frame-step 15

   The --all option looks through every object folder in data/raw/videos.
   If a video already has photos, the script skips it so it does not redo work.

WHAT THIS FILE DOES

- A video is just a lot of pictures shown quickly.
- This script opens a video and saves some of those pictures as image files.
- Those image files can later be used to train an ML model.
- The object type comes from the folder name, like washer, screw, or bolt.

EXPECTED INPUT LAYOUT

    data/raw/videos/washer/pan_01.mp4

EXPECTED OUTPUT LAYOUT

    data/raw/photos/<object_type>/<video_file>/<generated_frame_images>

Example:

    data/raw/photos/washer/pan_01.mp4/frame_000000.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# This file lives in src/, so parents[1] means "the project folder above src".
# Keeping paths based on PROJECT_ROOT lets the script work no matter where the
# terminal is opened from, as long as Python runs this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw videos are the original clips recorded by a person.
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "videos"

# Raw photos are the image frames created from those videos.
DEFAULT_PHOTO_DIR = PROJECT_ROOT / "data" / "raw" / "photos"

# When batch mode checks whether a video was already processed, these are the
# file endings that count as "photos already exist".
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    """Read the user's command-line options.

    argparse turns terminal text like:

        python data/extract_video_frames.py washer pan_01.mp4 --frame-step 15

    into a Python object with fields like args.object_type, args.video_name,
    and args.frame_step.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from data/raw/videos/<object_type>/<video> into "
            "data/raw/photos/<object_type>/<video>/ for ML processing."
        )
    )
    parser.add_argument(
        "object_type",
        nargs="?",
        help="Object folder under data/raw/videos/, such as washer.",
    )
    parser.add_argument(
        "video_name",
        nargs="?",
        help="Video filename inside the object folder, such as pan_01.mp4.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every video under data/raw/videos/ and skip videos with existing photos.",
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


def resolve_video_path(object_type: str, video_name: str) -> Path:
    """Build the video path from the object folder and video filename.

    Common use:
        washer pan_01.mp4 -> data/raw/videos/washer/pan_01.mp4
    """
    # Validate the object folder before using it in a filesystem path. This
    # prevents accidental paths like "../somewhere_else".
    object_type = validate_object_type(object_type)

    # video_name should be only a filename, not a folder path. For example,
    # "pan_01.mp4" is OK, but "washer/pan_01.mp4" is not needed anymore.
    video_path = Path(video_name)

    if video_path.is_absolute() or ".." in video_path.parts or len(video_path.parts) != 1:
        raise ValueError("video_name must be a filename inside the object folder.")

    # Final path example:
    # data/raw/videos/washer/pan_01.mp4
    return (DEFAULT_VIDEO_DIR / object_type / video_path).resolve()


def validate_object_type(object_type: str) -> str:
    """Keep object labels safe to use as folder names.

    This project uses folder names as labels. For example, every video inside
    data/raw/videos/washer is treated as a washer video.
    """
    output_path = Path(object_type)

    if output_path.is_absolute() or ".." in output_path.parts:
        raise ValueError("object_type must be a folder name, not an absolute or parent path.")

    if not object_type.strip():
        raise ValueError("object_type cannot be empty.")

    # Spaces in labels become underscores so "metal washer" becomes
    # "metal_washer". That keeps folder names simple.
    return object_type.strip().replace(" ", "_")


def extract_frames(
    video_path: Path,
    object_type: str,
    frame_step: int,
    start_frame: int,
    max_frames: int | None,
    image_format: str,
) -> int:
    """Read one video with OpenCV and save selected frames as image files.

    Think of this like flipping through a flipbook:

    - OpenCV opens the video.
    - The while loop asks for the next picture.
    - frame_step decides whether to keep or skip that picture.
    - cv2.imwrite saves kept pictures as PNG or JPG files.
    """
    # These checks make bad input fail early with a friendly message instead of
    # getting confusing OpenCV errors later.
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

    # Every video gets its own photo folder. This keeps different pans, angles,
    # and lighting conditions separated:
    #
    # data/raw/photos/washer/pan_01.mp4/
    # data/raw/photos/washer/pan_02.mp4/
    output_dir = DEFAULT_PHOTO_DIR / object_type / video_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # VideoCapture reads the video sequentially, one frame at a time.
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    saved_count = 0
    frame_index = 0

    # Treat "jpeg" and "jpg" as the same file type on disk.
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
                # The frame number goes in the filename so it is easy to trace
                # an image back to its position in the video.
                image_name = f"frame_{frame_index:06d}.{extension}"
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


def output_dir_for_video(video_path: Path, object_type: str) -> Path:
    """Return the folder where this video's extracted frames should live.

    Example:
        video_path = data/raw/videos/washer/pan_01.mp4
        object_type = washer
        returns data/raw/photos/washer/pan_01.mp4
    """
    return DEFAULT_PHOTO_DIR / validate_object_type(object_type) / video_path.name


def has_existing_photos(output_dir: Path) -> bool:
    """Check whether a video already has extracted image files.

    Batch mode uses this to avoid doing the same work twice. If the output
    folder already contains a .png, .jpg, or .jpeg file, the script assumes that
    video has already been processed.
    """
    if not output_dir.exists():
        return False

    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in output_dir.iterdir()
    )


def iter_dataset_videos() -> list[tuple[str, Path]]:
    """Find videos stored as data/raw/videos/<object_type>/<video_name>.

    Returns a list like:

        [("washer", Path("data/raw/videos/washer/pan_01.mp4"))]

    The object type is included with each path so batch mode knows which output
    folder to use.
    """
    if not DEFAULT_VIDEO_DIR.exists():
        return []

    videos: list[tuple[str, Path]] = []

    # First loop: each folder directly inside data/raw/videos is an object type.
    for object_dir in sorted(DEFAULT_VIDEO_DIR.iterdir()):
        # Skip files and hidden folders like .DS_Store.
        if not object_dir.is_dir() or object_dir.name.startswith("."):
            continue

        object_type = validate_object_type(object_dir.name)

        # Second loop: each file inside that object folder is a video to process.
        for video_path in sorted(object_dir.iterdir()):
            # Skip hidden files. macOS often creates .DS_Store, which is not a
            # real video and should not be sent to OpenCV.
            if video_path.is_file() and not video_path.name.startswith("."):
                videos.append((object_type, video_path.resolve()))

    return videos


def extract_all_frames(args: argparse.Namespace) -> tuple[int, int, int]:
    """Process every dataset video that does not already have photos.

    Returns three numbers:

    - how many videos were processed
    - how many videos were skipped
    - how many total images were saved
    """
    processed_count = 0
    skipped_count = 0
    saved_total = 0

    for object_type, video_path in iter_dataset_videos():
        output_dir = output_dir_for_video(video_path, object_type)

        # If there are already photos for this video, leave them alone. This is
        # useful when adding new videos over time: --all only works on the new
        # ones.
        if has_existing_photos(output_dir):
            print(f"Skipping {video_path}: photos already exist in {output_dir}")
            skipped_count += 1
            continue

        saved_count = extract_frames(
            video_path=video_path,
            object_type=object_type,
            frame_step=args.frame_step,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            image_format=args.image_format,
        )
        print(f"Saved {saved_count} image(s) to {output_dir}")
        processed_count += 1
        saved_total += saved_count

    return processed_count, skipped_count, saved_total


def main() -> int:
    """Run the command-line workflow and return a shell-friendly exit code.

    Returning 0 means success. Returning 1 means something went wrong. Shells,
    scripts, and CI tools understand those numbers.
    """
    args = parse_args()

    try:
        # Batch mode: process everything under data/raw/videos.
        if args.all:
            processed_count, skipped_count, saved_total = extract_all_frames(args)
            print(
                "Done: "
                f"processed {processed_count} video(s), "
                f"skipped {skipped_count}, "
                f"saved {saved_total} image(s)."
            )
            return 0

        # Single-video mode: object_type and video_name are required.
        if args.object_type is None or args.video_name is None:
            raise ValueError("Provide object_type and video_name, or use --all.")

        object_type = validate_object_type(args.object_type)
        video_path = resolve_video_path(object_type, args.video_name)

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

    output_dir = output_dir_for_video(video_path, object_type)
    print(f"Saved {saved_count} image(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
