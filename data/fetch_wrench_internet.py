"""Download real wrench pictures from the internet, already boxed, for training.

WHY THIS FILE EXISTS

Our own wrench photos are all close-up shots on the dark mat, so the detector is
good up close but weak when the wrench is far away (small) or on a new background.
Google's free "Open Images" dataset has thousands of real wrench pictures taken by
many people, at many distances and backgrounds, and every wrench already has a
hand-drawn box. This script downloads a batch of them and saves them in the exact
folder our dataset builder already reads.

HOW TO USE THIS FILE

    python data/fetch_wrench_internet.py --max-samples 400

Then rebuild the dataset and retrain (the new pictures fold in automatically):

    python data/auto_label_frames.py --overwrite
    python ml/train_yolo.py

WHAT IT DOES

- Uses the "fiftyone" library to pull the "Wrench" class out of Open Images.
- Saves each picture plus a matching YOLO box file into:

      data/hand_labeled/wrench_openimages/

- Writes a classes.txt there so our builder lines the class number up with
  data/labels/object_classes.txt.

Only the wrench class is downloaded. The bit is a unique, unusual object that the
internet does not have good pictures of, so we keep training the bit only on our
own photos.

NOTE: The first run installs nothing but does download image files, so it needs
internet and some disk space. Start small (a few hundred) and grow later.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Save into the hand-labeled area so data/auto_label_frames.py folds these in.
OUTPUT_DIR = PROJECT_ROOT / "data" / "hand_labeled" / "wrench_openimages"

# The Open Images name for this object, and the class name our project uses.
OPEN_IMAGES_CLASS = "Wrench"
OUR_CLASS_NAME = "wrench"


def get_detections(sample):
    """Return the list of boxes on a fiftyone sample, whatever the field is named."""
    for field_name in sample.field_names:
        value = sample[field_name]
        if hasattr(value, "detections"):
            return value.detections
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download boxed wrench pictures from Open Images."
    )
    parser.add_argument(
        "--max-samples", type=int, default=400,
        help="How many wrench pictures to download.",
    )
    parser.add_argument(
        "--split", default="train",
        choices=["train", "validation", "test"],
        help="Which Open Images split to pull from.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete previously downloaded wrench pictures first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import fiftyone.zoo as foz
    except ImportError:
        sys.exit(
            "The 'fiftyone' package is not installed.\n"
            "Install it with:\n"
            "    pip install fiftyone"
        )

    if OUTPUT_DIR.exists() and args.overwrite:
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading up to {args.max_samples} '{OPEN_IMAGES_CLASS}' pictures...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=args.split,
        label_types=["detections"],
        classes=[OPEN_IMAGES_CLASS],
        max_samples=args.max_samples,
        only_matching=True,   # keep only the wrench boxes, drop other labels
        shuffle=True,
        seed=42,
    )

    # Tell our builder the class name, so its number matches object_classes.txt.
    (OUTPUT_DIR / "classes.txt").write_text(f"{OUR_CLASS_NAME}\n")

    saved = 0
    boxes = 0
    for sample in dataset:
        detections = [
            d for d in get_detections(sample)
            if d.label == OPEN_IMAGES_CLASS
        ]
        if not detections:
            continue

        source = Path(sample.filepath)
        stem = source.stem
        image_out = OUTPUT_DIR / f"{stem}{source.suffix.lower()}"
        label_out = OUTPUT_DIR / f"{stem}.txt"

        lines = []
        for det in detections:
            # Open Images boxes are [top-left-x, top-left-y, width, height],
            # already as fractions of the picture. YOLO wants the center point.
            x, y, w, h = det.bounding_box
            cx = x + w / 2
            cy = y + h / 2
            if w <= 0 or h <= 0:
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not lines:
            continue

        shutil.copyfile(source, image_out)
        label_out.write_text("\n".join(lines) + "\n")
        saved += 1
        boxes += len(lines)

    print(f"\nSaved {saved} wrench pictures ({boxes} boxes) to:")
    print(f"  {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  python data/auto_label_frames.py --overwrite")
    print("  python ml/train_yolo.py")

    if saved == 0:
        print(
            "\nWARNING: nothing was saved. The download may have failed, or the "
            "class name may have changed. Check your internet connection."
        )


if __name__ == "__main__":
    main()
