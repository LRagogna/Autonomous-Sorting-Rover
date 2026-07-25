"""Auto-draw a box around the object in each frame and build the YOLO dataset.

Reads extracted frames from:

    data/frames/<class>/<class>__<video>__frame_NNN.jpg

and writes the standard YOLO dataset:

    data/yolo_dataset/images/{train,val}/<stem>.jpg
    data/yolo_dataset/labels/{train,val}/<stem>.txt
    data/yolo_dataset/dataset.yaml

Our objects (a wrench, a bit) are shiny/dark metal on a plain mat, so OpenCV can
guess a box on its own (GrabCut-style top-hat/black-hat masking). Boxes are NOT
baked into review images anymore: the Review tab draws them live from the label
files onto the raw frame, which is what lets the user drag, redraw, and correct
them. Frames auto-labeled here start life "unreviewed".

The train/val split is by whole video clip (every Nth clip goes to val) because
frames from one short clip look nearly identical.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml import dataset_utils as du  # noqa: E402
from ml import label_utils as lu  # noqa: E402


# ---------------------------------------------------------------------------
# Object box detection (ported unchanged from the original auto-labeler).
# ---------------------------------------------------------------------------
WORKING_WIDTH = 640
MAX_BOX_AREA_FRACTION = 0.4
BORDER_MARGIN = 0.07


def find_object_box(image: np.ndarray, pad: float) -> tuple[int, int, int, int] | None:
    """Guess the box around the single object sitting on the mat.

    Uses "top-hat" (bright spots) and "black-hat" (dark spots) morphology so the
    object stands out whether it is shiny or dark, then takes the biggest blob
    whose center lands in the middle of the frame. Returns (x, y, w, h) in the
    original picture's pixels, or None when nothing object-like is found.
    """
    full_height, full_width = image.shape[:2]
    if full_width > WORKING_WIDTH:
        scale = WORKING_WIDTH / full_width
        small = cv2.resize(image, (WORKING_WIDTH, int(full_height * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = image

    small_height, small_width = small.shape[:2]
    mask = _object_mask(small)
    if mask is None or int(mask.sum()) == 0:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    def is_centered(contour: np.ndarray) -> bool:
        bx, by, bw, bh = cv2.boundingRect(contour)
        cx, cy = bx + bw / 2, by + bh / 2
        return (0.2 * small_width <= cx <= 0.8 * small_width
                and 0.2 * small_height <= cy <= 0.8 * small_height)

    centered = [c for c in contours if is_centered(c)]
    best = max(centered if centered else contours, key=cv2.contourArea)
    if cv2.contourArea(best) < (0.0008 * small_width * small_height):
        return None

    x, y, w, h = cv2.boundingRect(best)
    x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)

    pad_x, pad_y = int(w * pad), int(h * pad)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(full_width, x + w + pad_x)
    y1 = min(full_height, y + h + pad_y)
    box_w, box_h = x1 - x0, y1 - y0
    if box_w * box_h > MAX_BOX_AREA_FRACTION * full_width * full_height:
        return None
    return x0, y0, box_w, box_h


def _object_mask(image: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    background_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bright_spots = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, background_kernel)
    dark_spots = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, background_kernel)
    standout = cv2.max(bright_spots, dark_spots)
    standout = cv2.normalize(standout, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(standout, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = _clean_mask(mask)
    mx, my = int(width * BORDER_MARGIN), int(height * BORDER_MARGIN)
    mask[:my, :] = 0
    mask[height - my:, :] = 0
    mask[:, :mx] = 0
    mask[:, width - mx:] = 0
    return mask


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask


# ---------------------------------------------------------------------------
# Group frames by (class, source video) so we can split train/val by clip.
# ---------------------------------------------------------------------------
def collect_clip_frames() -> dict[tuple[str, str], list[Path]]:
    """Return {(class, video_stem): [frame paths]} from data/frames/."""
    clips: dict[tuple[str, str], list[Path]] = {}
    if not du.FRAMES_DIR.exists():
        return clips
    for class_dir in sorted(p for p in du.FRAMES_DIR.iterdir() if p.is_dir()):
        if class_dir.name.startswith("."):
            continue
        for frame in sorted(p for p in class_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in du.IMAGE_EXTENSIONS):
            video_stem = du.source_video_from_stem(frame.stem)
            clips.setdefault((class_dir.name, video_stem), []).append(frame)
    return clips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-draw boxes and build the YOLO dataset.")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Share of clips used for checking instead of learning (0-1).")
    parser.add_argument("--pad", type=float, default=0.06,
                        help="Extra padding around each guessed box (fraction of its size).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Clear and rebuild the whole dataset before writing.")
    parser.add_argument("--incremental", action="store_true",
                        help="Only label clips that are not already in the dataset.")
    return parser.parse_args()


def run(overwrite: bool = False, incremental: bool = True,
        val_fraction: float = 0.2, pad: float = 0.06) -> dict:
    """Label frames and build the dataset. Returns a summary dict.

    This is the reusable core used by both ``main()`` (CLI) and the Process
    Dataset job orchestrator (``ml/process_dataset.py``).
    """
    du.ensure_core_dirs()

    name_to_id = du.class_name_to_id()
    names_in_order = du.class_names()
    if not name_to_id:
        print("No classes defined yet. Add clips (which create classes) first.", file=sys.stderr)
        return {"ok": False, "reason": "no-classes"}

    if overwrite:
        du.reset_dataset()

    clips = collect_clip_frames()
    if not clips:
        print("No frames found under data/frames/. Extract frames first.", file=sys.stderr)
        return {"ok": False, "reason": "no-frames"}

    processed = du.load_processed_clips()

    # Seed the per-class val counter from clips already in the dataset so new
    # clips keep continuing the "every Nth clip -> val" pattern.
    val_counter: dict[str, int] = {}
    if incremental and not overwrite:
        for entry in processed.values():
            if isinstance(entry, dict) and entry.get("labeled"):
                cls = entry.get("class", "")
                val_counter[cls] = val_counter.get(cls, 0) + 1

    written = {"train": 0, "val": 0}
    labels_created = 0
    skipped_no_box = 0
    skipped_tracked = 0
    clips_labeled = 0

    for (class_name, video_stem), frames in sorted(clips.items()):
        if class_name not in name_to_id:
            print(f"Skipping '{class_name}': no line in object_classes.txt")
            continue
        key = du.clip_key_for_stem(class_name, video_stem)
        if incremental and processed.get(key, {}).get("labeled"):
            skipped_tracked += 1
            continue
        class_id = name_to_id[class_name]

        val_counter[class_name] = val_counter.get(class_name, 0) + 1
        step = max(2, round(1 / val_fraction)) if val_fraction > 0 else 0
        split = "val" if (step and val_counter[class_name] % step == 0) else "train"

        clip_written = 0
        for frame_path in frames:
            stem = frame_path.stem
            # Respect earlier review decisions: never resurrect a rejected frame
            # or duplicate one already in the dataset (keeps re-processing safe).
            if (du.REJECTED_IMAGES_DIR / f"{stem}.jpg").exists() or du.find_dataset_image(stem)[0] is not None:
                continue
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            box = find_object_box(image, pad)
            if box is None:
                skipped_no_box += 1
                continue
            height, width = image.shape[:2]
            cv2.imwrite(str(du.dataset_image_path(stem, split)), image)
            yolo = lu.pixel_box_to_yolo(box, width, height)
            du.dataset_label_path(stem, split).write_text(
                lu.format_label_lines([{"cls": class_id, **yolo}])
            )
            written[split] += 1
            labels_created += 1
            clip_written += 1

        entry = processed.get(key, {})
        entry.update({
            "class": class_name,
            "labeled": True,
            "labeled_frames": clip_written,
            "split": split,
            "status": "labeled",
        })
        entry.setdefault("video", du.find_source_video(class_name, video_stem).name
                         if du.find_source_video(class_name, video_stem) else video_stem)
        processed[key] = entry
        clips_labeled += 1

    du.save_processed_clips(processed)
    du.write_dataset_yaml(names_in_order)

    print("\nDone building the YOLO dataset.")
    print(f"  classes:          {', '.join(names_in_order)}")
    print(f"  training images:  {written['train']}")
    print(f"  check images:     {written['val']}")
    print(f"  clips labeled:    {clips_labeled}")
    if skipped_tracked:
        print(f"  clips skipped (already labeled): {skipped_tracked}")
    if skipped_no_box:
        print(f"  frames skipped (no object found): {skipped_no_box}")
    # Machine-readable summary line the GUI parses for the processing summary.
    print(f"LABELS_CREATED={labels_created} NO_BOX={skipped_no_box} "
          f"TRAIN={written['train']} VAL={written['val']}")
    return {
        "ok": True,
        "labels_created": labels_created,
        "no_box": skipped_no_box,
        "train": written["train"],
        "val": written["val"],
        "clips_labeled": clips_labeled,
    }


def main() -> int:
    args = parse_args()
    result = run(
        overwrite=args.overwrite,
        incremental=args.incremental,
        val_fraction=args.val_fraction,
        pad=args.pad,
    )
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
