"""Turn recorded clips into still frames for training.

Reads videos from:

    data/raw_videos/<class>/<video>

and writes evenly-spaced still frames to:

    data/frames/<class>/<class>__<video>__frame_NNN.jpg

The class name and the source video are baked into every frame filename so the
frames stay globally unique and the training/validation split can happen by
whole video later (frames from the same clip look nearly identical).

USAGE

    python ml/extract_frames.py --all --frame-step 15
    python ml/extract_frames.py --frame-step 10 --select bit/IMG_1297.MOV wrench/IMG_1929.MOV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml import dataset_utils as du  # noqa: E402


def has_existing_frames(class_name: str, video_path: Path) -> bool:
    """Return True when a clip already has extracted frames on disk."""
    class_dir = du.FRAMES_DIR / class_name
    if not class_dir.exists():
        return False
    prefix = f"{class_name}__{du.clip_stem_for_video(video_path)}__frame_"
    return any(
        p.is_file() and p.suffix.lower() in du.IMAGE_EXTENSIONS and p.name.startswith(prefix)
        for p in class_dir.iterdir()
    )


def iter_all_videos() -> list[tuple[str, Path]]:
    """List every (class, video) under data/raw_videos/."""
    videos: list[tuple[str, Path]] = []
    if not du.RAW_VIDEOS_DIR.exists():
        return videos
    for class_dir in sorted(p for p in du.RAW_VIDEOS_DIR.iterdir() if p.is_dir()):
        # Skip the reserved background/ folder — its pictures are negatives, not a
        # class, and are folded in separately (never turned into object frames).
        if class_dir.name.startswith(".") or du.is_background_class(class_dir.name):
            continue
        for video in sorted(p for p in class_dir.iterdir() if p.is_file()):
            if video.name.startswith(".") or video.suffix.lower() not in du.VIDEO_EXTENSIONS:
                continue
            videos.append((class_dir.name, video.resolve()))
    return videos


def resolve_selection(select: list[str] | None, use_all: bool) -> list[tuple[str, Path]]:
    """Turn the CLI selection into a list of (class, video_path)."""
    if use_all or not select:
        return iter_all_videos()
    chosen: list[tuple[str, Path]] = []
    for key in select:
        if "/" not in key:
            raise ValueError(f"--select values must look like class/video: {key!r}")
        class_name, video_name = key.split("/", 1)
        class_name = du.validate_class_name(class_name)
        video_path = du.RAW_VIDEOS_DIR / class_name / du.safe_filename(video_name)
        if not video_path.exists():
            print(f"  skip {key}: video not found")
            continue
        chosen.append((class_name, video_path.resolve()))
    return chosen


def extract_video(video_path: Path, class_name: str, frame_step: int,
                  start_frame: int = 0, max_frames: int | None = None) -> int:
    """Extract frames from one video and record it as processed."""
    if frame_step < 1:
        raise ValueError("--frame-step must be 1 or greater.")
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError("OpenCV is not installed. Run: pip install -r requirements.txt") from error

    class_dir = du.FRAMES_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index >= start_frame and (frame_index - start_frame) % frame_step == 0:
                stem = du.frame_stem(class_name, video_path, frame_index)
                out_path = class_dir / f"{stem}.jpg"
                if not cv2.imwrite(str(out_path), frame):
                    raise RuntimeError(f"Could not write image: {out_path}")
                saved += 1
                if max_frames is not None and saved >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()

    # Record the clip so the GUI can show a "processed" badge and skip re-work.
    clips = du.load_processed_clips()
    clips[du.clip_key(class_name, video_path.name)] = {
        "class": class_name,
        "video": video_path.name,
        "path": du.repo_relative(video_path),
        "frames": saved,
        "frame_step": frame_step,
        "duration": round(du.video_duration_seconds(video_path), 2),
        "status": "extracted",
    }
    du.save_processed_clips(clips)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract still frames from recorded clips.")
    parser.add_argument("--all", action="store_true", help="Process every clip under data/raw_videos/.")
    parser.add_argument("--select", nargs="*", default=None,
                        help="Specific clips as class/video keys, e.g. bit/IMG_1297.MOV.")
    parser.add_argument("--frame-step", type=int, default=15, help="Save every Nth frame (default 15).")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to consider.")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap on frames saved per clip.")
    parser.add_argument("--reprocess", action="store_true",
                        help="Re-extract even clips that already have frames.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    du.ensure_core_dirs()
    try:
        videos = resolve_selection(args.select, args.all)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if not videos:
        print("No clips to extract. Upload videos to data/raw_videos/<class>/ first.")
        return 0

    processed, skipped, total = 0, 0, 0
    for class_name, video_path in videos:
        if not args.reprocess and has_existing_frames(class_name, video_path):
            print(f"Skipping {class_name}/{video_path.name}: frames already extracted")
            skipped += 1
            continue
        print(f"Extracting {class_name}/{video_path.name} (every {args.frame_step} frames)...")
        saved = extract_video(video_path, class_name, args.frame_step,
                              args.start_frame, args.max_frames)
        print(f"  saved {saved} frame(s) to data/frames/{class_name}/")
        processed += 1
        total += saved

    print(f"\nExtracted {total} frame(s) from {processed} clip(s); skipped {skipped}.")
    print(f"FRAMES_EXTRACTED={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
