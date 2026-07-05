"""Draw a box around the object in each photo and build a YOLO dataset.

HOW TO USE THIS FILE

1. Make sure you already extracted photos from your videos:

       python data/extract_video_frames.py --all --frame-step 15

   That fills these folders with pictures:

       data/raw/photos/<object_type>/<video_file>/frame_000000.png

2. Make sure every object type has a line in the class list:

       data/labels/object_classes.txt

   The file looks like this (one number and one name per line):

       0 bit
       1 wrench

3. Build the YOLO training dataset:

       python data/auto_label_frames.py --overwrite

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

    data/processed/detection/
      dataset.yaml            <- tells YOLO the class names and folders
      images/train/           <- most pictures, used for learning
      images/val/             <- a few pictures, used for checking
      labels/train/           <- one .txt box file per training picture
      labels/val/             <- one .txt box file per check picture
      review/                 <- the same pictures with the box drawn on top,
                                 so a human can quickly confirm the boxes

TIP: open a handful of images in review/ after running. If the green boxes hug
the object, you are ready to train. If a class looks wrong, delete those photos
or re-run with a different --pad value.

HOW TO ADD A NEW OBJECT LATER

1. Record short videos into data/raw/videos/<new_object>/
2. python data/extract_video_frames.py --all --frame-step 15
3. Add a new line to data/labels/object_classes.txt, e.g. "2 washer"
4. python data/auto_label_frames.py --overwrite
5. python ml/train_yolo.py
"""

from __future__ import annotations

import argparse
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

# The plain-text class list, one "id name" per line (input).
CLASS_LIST_FILE = PROJECT_ROOT / "data" / "labels" / "object_classes.txt"

# Optional hand-labeled pictures (input). These are pictures you boxed yourself
# with a tool like LabelImg, for hard scenes the auto-labeler cannot handle
# (object held in your hand, busy or colored backgrounds). Any image here that
# has a matching YOLO .txt file next to it is folded into the dataset. See
# data/hand_labeled/README.md.
HAND_LABELED_DIR = PROJECT_ROOT / "data" / "hand_labeled"

# Where the finished YOLO dataset is written (output).
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "detection"

# File endings that count as pictures.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# Some source clips are too hard for automatic labeling, so we leave the whole
# clip out of the dataset. Two kinds of clips fail:
#   1. Low-light shots on a dark, wrinkled blanket, where shadows look like the
#      object (the IMG_1929-1933 bit clips).
#   2. Busy webcam shots where a face and room fill the frame and the object is
#      small and off to the side, so the auto-labeler boxes the face instead of
#      the object (the webcam wrench clips).
# Auto-labeling works best on a plain, uncluttered background with the object
# near the middle. To use a clip again, delete its name from this list and
# re-run. To exclude a new bad clip, add its folder name (the same name shown in
# data/raw/photos/<object>/).
IGNORE_CLIPS = {
    "IMG_1929.MOV",
    "IMG_1930.MOV",
    "IMG_1931.MOV",
    "IMG_1932.MOV",
    "IMG_1933.MOV",
    "webcam_20260703_215357",
    "webcam_20260704_112722",
}


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


def collect_clip_frames() -> dict[tuple[str, str], list[Path]]:
    """Group every photo by (object_type, source video folder).

    We split train/val by whole video clips, not by single frames. Frames from
    the same short video look almost identical, so keeping a clip together stops
    the check set from being an easy copy of the training set.
    """
    clips: dict[tuple[str, str], list[Path]] = {}
    for object_dir in sorted(p for p in PHOTO_DIR.iterdir() if p.is_dir()):
        for clip_dir in sorted(p for p in object_dir.iterdir() if p.is_dir()):
            if clip_dir.name in IGNORE_CLIPS:
                continue
            frames = sorted(
                p
                for p in clip_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if frames:
                clips[(object_dir.name, clip_dir.name)] = frames
    return clips


def reset_output_dirs() -> None:
    """Delete any previous dataset and recreate the empty folder structure."""
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "review"):
        (DATASET_DIR / sub).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(class_names_in_order: list[str]) -> None:
    """Write dataset.yaml so YOLO knows the folders and class names."""
    names_block = "\n".join(
        f"  {index}: {name}" for index, name in enumerate(class_names_in_order)
    )
    text = (
        "# This file tells YOLO where the pictures are and what the classes are.\n"
        "# 'path' is the full path to this dataset folder so YOLO always finds it.\n"
        f"path: {DATASET_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
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


def _read_classes_txt(folder: Path) -> dict[int, str] | None:
    """Read a LabelImg classes.txt (one name per line) into {local_id: name}."""
    classes_file = folder / "classes.txt"
    if not classes_file.exists():
        return None
    local: dict[int, str] = {}
    for index, line in enumerate(classes_file.read_text().splitlines()):
        name = line.strip()
        if name:
            local[index] = name
    return local


def ingest_hand_labeled(name_to_id: dict[str, int], val_fraction: float) -> dict[str, int]:
    """Fold hand-labeled pictures (image + YOLO .txt) into the dataset.

    LabelImg saves the class numbers in the order shown in its class list. To be
    safe, if a picture's folder has a classes.txt we remap those local numbers to
    our official numbers by NAME, so the ids always line up with
    data/labels/object_classes.txt no matter what order LabelImg used.

    Every Nth labeled picture (per --val-fraction) goes to the check set.
    """
    written = {"train": 0, "val": 0, "unlabeled": 0}
    if not HAND_LABELED_DIR.exists():
        return written

    step = max(2, round(1 / val_fraction)) if val_fraction > 0 else 0
    val_counter = 0

    images = sorted(
        p for p in HAND_LABELED_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    for image_path in images:
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists() or not label_path.read_text().strip():
            # Picture is not labeled yet, so we cannot use it. Skip quietly.
            written["unlabeled"] += 1
            continue

        local_classes = _read_classes_txt(image_path.parent)

        # Rewrite each line so the class number matches our official numbering.
        fixed_lines: list[str] = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            local_id = int(parts[0])
            if local_classes is not None and local_id in local_classes:
                name = local_classes[local_id]
                if name not in name_to_id:
                    print(f"  hand-labeled: unknown class '{name}' in {label_path.name}, skipping line")
                    continue
                global_id = name_to_id[name]
            else:
                global_id = local_id  # assume already correct
            fixed_lines.append(f"{global_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

        if not fixed_lines:
            continue

        val_counter += 1
        split = "val" if (step and val_counter % step == 0) else "train"

        # Flatten the path into a unique name, tagged "hand__" so these are easy
        # to tell apart from the auto-labeled mat pictures.
        rel = image_path.relative_to(HAND_LABELED_DIR)
        stem = "hand__" + "__".join(rel.with_suffix("").parts).replace(".", "_")

        suffix = image_path.suffix.lower()
        shutil.copyfile(image_path, DATASET_DIR / "images" / split / f"{stem}{suffix}")
        (DATASET_DIR / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(fixed_lines) + "\n"
        )
        written[split] += 1

    return written


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
            "Run data/extract_video_frames.py first."
        )

    if DATASET_DIR.exists() and not args.overwrite:
        sys.exit(
            f"{DATASET_DIR} already exists. Re-run with --overwrite to rebuild it."
        )

    reset_output_dirs()

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

    for (object_type, clip_name), frames in clips.items():
        if object_type not in name_to_id:
            print(f"Skipping '{object_type}': no matching line in object_classes.txt")
            continue
        class_id = name_to_id[object_type]

        # Decide train vs val for this whole clip. Every Nth clip goes to val.
        val_counter[object_type] = val_counter.get(object_type, 0) + 1
        step = max(2, round(1 / args.val_fraction)) if args.val_fraction > 0 else 0
        is_val = step and (val_counter[object_type] % step == 0)
        split = "val" if is_val else "train"

        for frame_path in frames:
            per_object_seen[object_type] = per_object_seen.get(object_type, 0) + 1
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
                continue

            height, width = image.shape[:2]
            # A unique, flat name so pictures from different clips never collide.
            stem = f"{object_type}__{clip_name}__{frame_path.stem}".replace(".", "_")

            image_out = DATASET_DIR / "images" / split / f"{stem}.jpg"
            label_out = DATASET_DIR / "labels" / split / f"{stem}.txt"
            review_out = DATASET_DIR / "review" / f"{stem}.jpg"

            cv2.imwrite(str(image_out), image)
            label_out.write_text(f"{class_id} {box_to_yolo(box, width, height)}\n")
            save_review_image(image, box, object_type, review_out)

            per_object_used[object_type] = per_object_used.get(object_type, 0) + 1
            written[split] += 1

    # Fold in any pictures you boxed yourself with LabelImg (hands, backgrounds).
    hand = ingest_hand_labeled(name_to_id, args.val_fraction)
    written["train"] += hand["train"]
    written["val"] += hand["val"]

    write_dataset_yaml(class_names_in_order)

    print("\nDone building the YOLO dataset.")
    print(f"  classes:        {', '.join(class_names_in_order)}")
    print(f"  training images: {written['train']}")
    print(f"  check images:    {written['val']}")
    if hand["train"] or hand["val"]:
        print(f"  (of those, hand-labeled: {hand['train']} train, {hand['val']} check)")
    if hand["unlabeled"]:
        print(f"  hand_labeled pictures with no boxes yet (skipped): {hand['unlabeled']}")
    if skipped_no_box:
        print(f"  skipped (no object found): {skipped_no_box}")
    print(f"\nDataset folder: {DATASET_DIR}")
    print(f"Spot-check the boxes here: {DATASET_DIR / 'review'}")
    print("Next step:  python ml/train_yolo.py")

    if written["val"] == 0:
        print(
            "\nWARNING: no check images were created. Add more video clips per "
            "object, or lower --val-fraction is not the issue here."
        )


if __name__ == "__main__":
    main()
