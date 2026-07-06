"""Draw a box around the object in each photo and build a YOLO dataset.

HOW TO USE THIS FILE

1. The normal workflow is to run:

       ./scripts/process.sh

   That command first extracts new photos from videos, then calls this file.
   The extracted photos live here:

       data/raw/photos/<object_type>/<clip_name>__frame_000000.jpg

2. Make sure every object type has a line in the class list:

       data/labels/object_classes.txt

   The file looks like this (one number and one name per line):

       0 bit
       1 wrench

3. This file builds or updates the YOLO training dataset:

       ./scripts/process.sh

   This looks at every photo, finds the metal object sitting on the mat,
   and writes a matching label file that says where the object is.

WHAT THIS FILE DOES

A YOLO detector learns from two things for every picture:

- the picture itself, and
- a tiny text file that says "the object is inside this box".

Our objects (a wrench, a bit) are shiny/dark metal on a plain dark mat, so a
computer can guess the box on its own using OpenCV. For each photo this script:

- separates the object from the background (OpenCV GrabCut, with a simpler
  brightness/darkness fallback),
- measures the smallest box that still holds the whole object,
- turns that box into the four YOLO numbers (center-x, center-y, width, height,
  all as fractions from 0 to 1), and
- copies the picture and its label into the training dataset.

WHERE THE RESULTS GO

    data/labels/
      object_classes.txt      <- master class list
      dataset.yaml            <- tells YOLO where the object folders are
      <object>/
        images/train/         <- most pictures, used for learning
        images/val/           <- a few pictures, used for checking
        labels/train/         <- one .txt box file per training picture
        labels/val/           <- one .txt box file per check picture
        review/               <- boxed copies for this specific object

TIP: open the review folder for each object after running. If the green boxes
hug the object, you are ready to train. If a class looks wrong, delete those
photos or re-run with a different --pad value.

HOW TO ADD A NEW OBJECT LATER

1. Record short videos into data/raw/clips/<new_object>/
2. Add a new line to data/labels/object_classes.txt, e.g. "2 washer"
3. ./scripts/process.sh
4. Review data/labels/<new_object>/review/
5. ./scripts/train.sh
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


# This file lives in data/, so parents[1] is the project folder above data/.
# Building paths from PROJECT_ROOT lets the script run from any terminal folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Where the extracted photos already live (input).
PHOTO_DIR = PROJECT_ROOT / "data" / "raw" / "photos"

# Where the original captured videos live (input). We use this only to map a
# frame prefix like IMG_1297 back to the source video name for the tracker.
CLIPS_DIR = PROJECT_ROOT / "data" / "raw" / "clips"

# Raw frames rejected in the review UI are recorded here so they do not come
# back the next time the dataset is rebuilt.
DELETED_FRAMES_FILE = PHOTO_DIR / ".deleted_frames.json"

# Clips are added here after their frames have been written into data/labels.
PROCESSED_CLIPS_FILE = PHOTO_DIR / ".processed_clips.json"

# The plain-text class list, one "id name" per line (input).
CLASS_LIST_FILE = PROJECT_ROOT / "data" / "labels" / "object_classes.txt"

# Where the finished YOLO dataset is written (output).
DATASET_DIR = PROJECT_ROOT / "data" / "labels"

# File endings that count as pictures.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# Every clip is processed and every auto-boxed frame starts as "Pass" in the
# review gallery. Frames that the auto-labeler gets wrong can be moved to the
# "Fail" tab in the GUI, which pulls them out of training. If you ever want to
# skip an entire clip up front, add its folder name here (the same name shown in
# data/raw/photos/<object>/).
IGNORE_CLIPS: set[str] = set()


def load_classes() -> dict[str, int]:
    """Read data/labels/object_classes.txt into a {name: id} dictionary.

    Each line looks like "0 bit". Blank lines are ignored. We return a mapping
    from the object folder name to its YOLO class number.
    """
    if not CLASS_LIST_FILE.exists():
        sys.exit(
            f"Could not find the class list: {CLASS_LIST_FILE}\n"
            "Create it with one line per object, for example:\n"
            "    0 bit\n    1 wrench"
        )

    name_to_id: dict[str, int] = {}
    for line in CLASS_LIST_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        class_id = int(parts[0])
        class_name = parts[1]
        name_to_id[class_name] = class_id
    return name_to_id


def relative_to_project(path: Path) -> str:
    """Return a stable repo-relative path for workflow manifests."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json_file(path: Path, default):
    """Read a small JSON workflow file, returning default if it is missing."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def write_json_file(path: Path, data) -> None:
    """Write a small JSON workflow file in a readable, deterministic format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_deleted_frames() -> set[str]:
    """Read raw frames the user rejected in the review GUI."""
    data = read_json_file(DELETED_FRAMES_FILE, {"deleted_frames": []})
    if isinstance(data, dict):
        entries = data.get("deleted_frames", [])
    elif isinstance(data, list):
        entries = data
    else:
        entries = []
    return {str(entry) for entry in entries}


def load_processed_clips() -> dict[str, dict]:
    """Load clips already added to the YOLO dataset."""
    data = read_json_file(PROCESSED_CLIPS_FILE, {"clips": {}})
    if isinstance(data, dict) and isinstance(data.get("clips"), dict):
        return data["clips"]
    return {}


def save_processed_clips(clips: dict[str, dict]) -> None:
    """Save clips already added to the YOLO dataset."""
    write_json_file(PROCESSED_CLIPS_FILE, {"clips": clips})


def clip_stem_for_video(video_path: Path) -> str:
    """Match the frame prefix created by extract_video_frames.py."""
    return video_path.stem.strip().replace(" ", "_").replace(".", "_")


def source_video_for_clip(object_type: str, clip_name: str) -> Path | None:
    """Find the original video file for a processed clip name."""
    object_dir = CLIPS_DIR / object_type
    if not object_dir.exists():
        return None
    for video_path in sorted(p for p in object_dir.iterdir() if p.is_file()):
        if clip_stem_for_video(video_path) == clip_name:
            return video_path
    return None


def clip_manifest_key(object_type: str, clip_name: str) -> str:
    """Return the processed tracker key for one clip."""
    video_path = source_video_for_clip(object_type, clip_name)
    if video_path is not None:
        return f"{object_type}/{video_path.name}"
    return f"{object_type}/{clip_name}"


def mark_clip_processed(
    clips: dict[str, dict],
    object_type: str,
    clip_name: str,
    seen_frames: int,
    written_frames: int,
    skipped_frames: int,
) -> None:
    """Record that a clip has been incorporated into data/labels."""
    video_path = source_video_for_clip(object_type, clip_name)
    key = clip_manifest_key(object_type, clip_name)
    clips[key] = {
        "clip_stem": clip_name,
        "object": object_type,
        "path": relative_to_project(video_path) if video_path else "",
        "processed_frames": written_frames,
        "seen_frames": seen_frames,
        "skipped_frames": skipped_frames,
        "status": "dataset",
        "video": video_path.name if video_path else clip_name,
    }
    save_processed_clips(clips)


# We shrink every photo to this width before looking for the object, then scale
# the box back up. Full phone photos are 1920 pixels wide; working small keeps
# the whole job fast and does not hurt the box quality.
WORKING_WIDTH = 640

# If the found box covers more than this share of the whole picture, we treat it
# as a failure (usually a dark, low-contrast photo where shadow blended into the
# mat) and skip the frame instead of writing a bad label. Real wrench/bit boxes
# are small, so a large box means the object did not separate from the mat.
MAX_BOX_AREA_FRACTION = 0.4


def find_object_box(image: np.ndarray, pad: float) -> tuple[int, int, int, int] | None:
    """Guess the box around the single object sitting on the mat.

    The mat is a smooth gray cloth; the object is a compact, high-contrast lump
    on top of it. A wrench is much brighter than the mat and a bit is darker, so
    we use two classic OpenCV tricks together:

    - "top-hat" keeps small BRIGHT spots that stand out from the background,
    - "black-hat" keeps small DARK spots that stand out from the background.

    Taking the stronger of the two at each pixel highlights the object whether it
    is shiny or dark, while ignoring the mat's smooth shading and vignette.

    Returns the box as (x, y, width, height) in the ORIGINAL picture's pixels,
    or None if nothing object-like was found.
    """
    full_height, full_width = image.shape[:2]

    # Work on a smaller copy for speed. Remember the shrink factor so we can turn
    # the small-image box back into full-size pixels at the end.
    if full_width > WORKING_WIDTH:
        scale = WORKING_WIDTH / full_width
        small = cv2.resize(
            image, (WORKING_WIDTH, int(full_height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
        small = image

    small_height, small_width = small.shape[:2]

    mask = _object_mask(small)
    if mask is None or int(mask.sum()) == 0:
        return None

    # Find the object blob. The object sits near the middle of these clips, so we
    # prefer the biggest blob whose CENTER lands in the central band of the
    # picture. That ignores stray bright things near the edges (like furniture
    # peeking in at the top). Ignore tiny specks of noise.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    def is_centered(contour: np.ndarray) -> bool:
        bx, by, bw, bh = cv2.boundingRect(contour)
        blob_center_x = bx + bw / 2
        blob_center_y = by + bh / 2
        return (
            0.2 * small_width <= blob_center_x <= 0.8 * small_width
            and 0.2 * small_height <= blob_center_y <= 0.8 * small_height
        )

    centered = [c for c in contours if is_centered(c)]
    candidates = centered if centered else contours
    best = max(candidates, key=cv2.contourArea)
    if cv2.contourArea(best) < (0.0008 * small_width * small_height):
        return None

    x, y, w, h = cv2.boundingRect(best)

    # Scale the small-image box back to the original picture size.
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    # Grow the box a little so we do not clip the edges of the object, then keep
    # it inside the picture.
    pad_x = int(w * pad)
    pad_y = int(h * pad)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(full_width, x + w + pad_x)
    y1 = min(full_height, y + h + pad_y)
    box_w, box_h = x1 - x0, y1 - y0

    # Sanity check: a box that fills almost the whole frame means the object did
    # not separate cleanly from the mat. Skip it rather than teach a bad box.
    if box_w * box_h > MAX_BOX_AREA_FRACTION * full_width * full_height:
        return None
    return x0, y0, box_w, box_h


# The object is always roughly centered in these clips, so we ignore this share
# of the picture around every edge. That throws away background clutter (a table
# or furniture peeking in at the top) and the darker corners of the mat.
BORDER_MARGIN = 0.07


def _object_mask(image: np.ndarray) -> np.ndarray | None:
    """Build a black-and-white mask where white pixels are the object."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # The background-removing kernel must be bigger than the object is "thick" so
    # the object shows up as a spot on top of the smooth mat.
    background_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bright_spots = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, background_kernel)
    dark_spots = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, background_kernel)
    standout = cv2.max(bright_spots, dark_spots)

    # Stretch the contrast, then let Otsu pick the bright/dark cutoff for us.
    standout = cv2.normalize(standout, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(standout, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = _clean_mask(mask)

    # Blank out the edge margin so only the central area can become the object.
    mx = int(width * BORDER_MARGIN)
    my = int(height * BORDER_MARGIN)
    mask[:my, :] = 0
    mask[height - my:, :] = 0
    mask[:, :mx] = 0
    mask[:, width - mx:] = 0
    return mask


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Remove small noise and fill small holes so one solid blob remains."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask


def box_to_yolo(box: tuple[int, int, int, int], width: int, height: int) -> str:
    """Turn a pixel box into the YOLO text line numbers (without the class id).

    YOLO wants the box center and size as fractions of the picture, from 0 to 1.
    """
    x, y, w, h = box
    cx = (x + w / 2) / width
    cy = (y + h / 2) / height
    nw = w / width
    nh = h / height
    return f"{cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def clip_name_for_frame(frame_path: Path, object_dir: Path) -> str:
    """Return the source clip name for either new flat or legacy nested photos."""
    if frame_path.parent == object_dir:
        if "__frame_" in frame_path.stem:
            return frame_path.stem.split("__frame_", 1)[0]
        return "manual"
    return frame_path.parent.name


def collect_clip_frames() -> dict[tuple[str, str], list[Path]]:
    """Group every photo by (object_type, source video folder).

    We split train/val by whole video clips, not by single frames. Frames from
    the same short video look almost identical, so keeping a clip together stops
    the check set from being an easy copy of the training set.
    """
    clips: dict[tuple[str, str], list[Path]] = {}
    deleted_frames = load_deleted_frames()
    for object_dir in sorted(p for p in PHOTO_DIR.iterdir() if p.is_dir()):
        direct_frames = sorted(
            p
            for p in object_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for frame_path in direct_frames:
            if relative_to_project(frame_path) in deleted_frames:
                continue
            clip_name = clip_name_for_frame(frame_path, object_dir)
            if clip_name in IGNORE_CLIPS:
                continue
            clips.setdefault((object_dir.name, clip_name), []).append(frame_path)

        # Legacy support: older split runs wrote frames into
        # data/raw/photos/<object>/<clip_name>/.
        for clip_dir in sorted(p for p in object_dir.iterdir() if p.is_dir()):
            if clip_dir.name in IGNORE_CLIPS:
                continue
            frames = sorted(
                p
                for p in clip_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
                and relative_to_project(p) not in deleted_frames
            )
            if frames:
                clips.setdefault((object_dir.name, clip_dir.name), []).extend(frames)
    return clips


def object_output_dir(object_type: str) -> Path:
    """Return the generated label/review folder for one object type."""
    return DATASET_DIR / object_type


def output_stem_for_frame(object_type: str, clip_name: str, frame_path: Path) -> str:
    """Return a unique processed filename stem for a raw frame."""
    if frame_path.parent == PHOTO_DIR / object_type:
        stem = f"{object_type}__{frame_path.stem}"
    else:
        stem = f"{object_type}__{clip_name}__{frame_path.stem}"
    return stem.replace(".", "_")


def ensure_object_output_dirs(class_names: list[str]) -> None:
    """Create the per-object YOLO image, label, and review folders."""
    for name in class_names:
        root = object_output_dir(name)
        for sub in ("images/train", "images/val", "labels/train", "labels/val", "review"):
            (root / sub).mkdir(parents=True, exist_ok=True)


def reset_output_dirs(class_names: list[str]) -> None:
    """Delete generated label outputs while keeping the class list in place."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for path in DATASET_DIR.iterdir():
        if path.name in {"object_classes.txt", ".gitkeep"}:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    ensure_object_output_dirs(class_names)


def write_dataset_yaml(class_names_in_order: list[str]) -> None:
    """Write dataset.yaml so YOLO knows the folders and class names."""
    names_block = "\n".join(
        f"  {index}: {name}" for index, name in enumerate(class_names_in_order)
    )
    train_block = "\n".join(
        f"  - {name}/images/train" for name in class_names_in_order
    )
    val_block = "\n".join(
        f"  - {name}/images/val" for name in class_names_in_order
    )
    text = (
        "# This file tells YOLO where the pictures are and what the classes are.\n"
        "# 'path' is an absolute folder so training works no matter which folder\n"
        "# the training command is run from. It is rewritten every time you\n"
        "# process data, so it stays correct after moving the repo.\n"
        f"path: {DATASET_DIR.resolve().as_posix()}\n"
        "train:\n"
        f"{train_block}\n"
        "val:\n"
        f"{val_block}\n"
        f"nc: {len(class_names_in_order)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    (DATASET_DIR / "dataset.yaml").write_text(text)


def save_review_image(
    image: np.ndarray, box: tuple[int, int, int, int], label: str, out_path: Path
) -> None:
    """Save a copy of the picture with the guessed green box drawn on it."""
    preview = image.copy()
    x, y, w, h = box
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(
        preview, label, (x, max(0, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), preview)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-draw boxes and build the YOLO detection dataset."
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Share of video clips used for checking instead of learning (0-1).",
    )
    parser.add_argument(
        "--pad", type=float, default=0.06,
        help="Extra padding around each guessed box, as a fraction of its size.",
    )
    parser.add_argument(
        "--limit-per-object", type=int, default=0,
        help="Optional cap on how many photos to use per object type (0 = all).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete and rebuild the dataset folder before writing.",
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Only process clips that are not already in the processed tracker.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    name_to_id = load_classes()
    # Class names ordered by their id, so dataset.yaml matches the label numbers.
    class_names_in_order = [
        name for name, _ in sorted(name_to_id.items(), key=lambda kv: kv[1])
    ]

    if not PHOTO_DIR.exists():
        sys.exit(
            f"No photos folder found at {PHOTO_DIR}\n"
            "Run ./scripts/split_frames.sh first."
        )

    if (DATASET_DIR / "dataset.yaml").exists() and not args.overwrite and not args.incremental:
        sys.exit(
            f"{DATASET_DIR / 'dataset.yaml'} already exists. "
            "Re-run with --overwrite to rebuild generated labels."
        )

    if args.overwrite:
        reset_output_dirs(class_names_in_order)
        processed_clips: dict[str, dict] = {}
    else:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        ensure_object_output_dirs(class_names_in_order)
        processed_clips = load_processed_clips()

    clips = collect_clip_frames()
    if not clips:
        sys.exit(f"No photos found under {PHOTO_DIR}. Extract some frames first.")

    # Count clips per object so we can send about --val-fraction of each object's
    # clips to the check set. This keeps both classes represented in val.
    per_object_seen: dict[str, int] = {}
    per_object_used: dict[str, int] = {}
    val_counter: dict[str, int] = {}
    written = {"train": 0, "val": 0}
    skipped_no_box = 0
    skipped_tracked = 0
    clips_processed = 0

    for (object_type, clip_name), frames in clips.items():
        if object_type not in name_to_id:
            print(f"Skipping '{object_type}': no matching line in object_classes.txt")
            continue
        if args.incremental and clip_manifest_key(object_type, clip_name) in processed_clips:
            skipped_tracked += 1
            continue
        class_id = name_to_id[object_type]

        # Decide train vs val for this whole clip. Every Nth clip goes to val.
        val_counter[object_type] = val_counter.get(object_type, 0) + 1
        step = max(2, round(1 / args.val_fraction)) if args.val_fraction > 0 else 0
        is_val = step and (val_counter[object_type] % step == 0)
        split = "val" if is_val else "train"
        clip_seen = 0
        clip_written = 0
        clip_skipped = 0

        for frame_path in frames:
            per_object_seen[object_type] = per_object_seen.get(object_type, 0) + 1
            clip_seen += 1
            if args.limit_per_object and (
                per_object_used.get(object_type, 0) >= args.limit_per_object
            ):
                break

            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            box = find_object_box(image, args.pad)
            if box is None:
                skipped_no_box += 1
                clip_skipped += 1
                continue

            height, width = image.shape[:2]
            # A unique, flat name so pictures from different clips never collide.
            stem = output_stem_for_frame(object_type, clip_name, frame_path)

            object_dir = object_output_dir(object_type)
            image_out = object_dir / "images" / split / f"{stem}.jpg"
            label_out = object_dir / "labels" / split / f"{stem}.txt"
            review_out = object_dir / "review" / f"{stem}.jpg"

            cv2.imwrite(str(image_out), image)
            label_out.write_text(f"{class_id} {box_to_yolo(box, width, height)}\n")
            save_review_image(image, box, object_type, review_out)

            per_object_used[object_type] = per_object_used.get(object_type, 0) + 1
            written[split] += 1
            clip_written += 1

        mark_clip_processed(
            processed_clips,
            object_type,
            clip_name,
            clip_seen,
            clip_written,
            clip_skipped,
        )
        clips_processed += 1

    write_dataset_yaml(class_names_in_order)

    print("\nDone building the YOLO dataset.")
    print(f"  classes:        {', '.join(class_names_in_order)}")
    prefix = "new " if args.incremental else ""
    print(f"  {prefix}training images: {written['train']}")
    print(f"  {prefix}check images:    {written['val']}")
    print(f"  clips processed: {clips_processed}")
    if skipped_tracked:
        print(f"  clips skipped (already tracked): {skipped_tracked}")
    if args.incremental and clips_processed == 0:
        print("  no new clips needed processing")
    if skipped_no_box:
        print(f"  skipped (no object found): {skipped_no_box}")
    print(f"\nDataset folder: {DATASET_DIR}")
    print("Spot-check the boxes in each object's review folder:")
    for name in class_names_in_order:
        print(f"  {object_output_dir(name) / 'review'}")
    print("Next step:  ./scripts/train.sh")

    if clips_processed > 0 and written["val"] == 0:
        print(
            "\nWARNING: no check images were created. Add more video clips per "
            "object, or lower --val-fraction is not the issue here."
        )


if __name__ == "__main__":
    main()
