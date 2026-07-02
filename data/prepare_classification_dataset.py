"""Prepare a cleaner train/validation image dataset from raw extracted frames.

The raw frame extractor keeps images grouped like this:

    data/raw/photos/<label>/<source_video>/frame_000000.png

This script turns those frames into a processed classifier dataset:

    data/processed/classification/train/<label>/<image>.jpg
    data/processed/classification/val/<label>/<image>.jpg

It keeps whole source videos together in either train or validation so near
duplicate frames from one video do not leak across the split.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PHOTO_DIR = PROJECT_ROOT / "data" / "raw" / "photos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "classification"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class RawImage:
    path: Path
    label: str
    group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a processed classification dataset from raw extracted frames."
    )
    parser.add_argument(
        "--raw-photo-dir",
        type=Path,
        default=DEFAULT_RAW_PHOTO_DIR,
        help="Input directory shaped as data/raw/photos/<label>/<source_video>/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for processed train/val folders.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Ratio of source-video groups to reserve for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable group splitting and class balancing.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Processed square image size in pixels.",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=0.85,
        help="Centered square crop size as a fraction of the shorter image side.",
    )
    parser.add_argument(
        "--min-frame-gap",
        type=int,
        default=2,
        help="Keep every Nth frame within each source video before balancing.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Keep all selected images instead of downsampling each class equally.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing processed classification dataset.",
    )
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_raw_images(raw_photo_dir: Path) -> list[RawImage]:
    images: list[RawImage] = []
    if not raw_photo_dir.exists():
        return images

    for label_dir in sorted(raw_photo_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("."):
            continue

        for source_dir in sorted(label_dir.iterdir()):
            if not source_dir.is_dir() or source_dir.name.startswith("."):
                continue

            group = f"{label_dir.name}/{source_dir.name}"
            for image_path in sorted(source_dir.iterdir()):
                if is_image(image_path):
                    images.append(
                        RawImage(
                            path=image_path.resolve(),
                            label=label_dir.name,
                            group=group,
                        )
                    )

    return images


def split_by_group(
    images: list[RawImage],
    val_ratio: float,
    seed: int,
) -> tuple[list[RawImage], list[RawImage]]:
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio must be greater than 0 and less than 1.")

    groups_by_label: dict[str, dict[str, list[RawImage]]] = defaultdict(dict)
    for image in images:
        groups_by_label[image.label].setdefault(image.group, []).append(image)

    train: list[RawImage] = []
    val: list[RawImage] = []
    rng = random.Random(seed)

    for label in sorted(groups_by_label):
        group_names = sorted(groups_by_label[label])
        rng.shuffle(group_names)

        if len(group_names) < 2:
            raise ValueError(
                f"Label {label!r} needs at least two source-video folders "
                "to create train and validation data."
            )

        val_count = max(1, round(len(group_names) * val_ratio))
        if val_count >= len(group_names):
            val_count = len(group_names) - 1

        val_groups = set(group_names[:val_count])
        for group_name in group_names:
            destination = val if group_name in val_groups else train
            destination.extend(groups_by_label[label][group_name])

    return train, val


def keep_spaced_frames(images: list[RawImage], min_frame_gap: int) -> list[RawImage]:
    if min_frame_gap <= 1:
        return images

    kept: list[RawImage] = []
    by_group: dict[str, list[RawImage]] = defaultdict(list)
    for image in images:
        by_group[image.group].append(image)

    for group_images in by_group.values():
        kept.extend(sorted(group_images, key=lambda image: image.path.name)[::min_frame_gap])

    return kept


def balance_by_label(images: list[RawImage], seed: int) -> list[RawImage]:
    by_label: dict[str, list[RawImage]] = defaultdict(list)
    for image in images:
        by_label[image.label].append(image)

    if not by_label:
        return []

    target_count = min(len(label_images) for label_images in by_label.values())
    rng = random.Random(seed)
    balanced: list[RawImage] = []

    for label in sorted(by_label):
        label_images = sorted(by_label[label], key=lambda image: str(image.path))
        rng.shuffle(label_images)
        balanced.extend(label_images[:target_count])

    return sorted(balanced, key=lambda image: (image.label, image.group, image.path.name))


def center_square_crop(image, crop_scale: float):
    height, width = image.shape[:2]
    side = int(min(height, width) * crop_scale)
    side = max(1, min(side, height, width))

    center_x = width // 2
    center_y = height // 2
    x1 = max(0, center_x - side // 2)
    y1 = max(0, center_y - side // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)

    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)
    return image[y1:y2, x1:x2]


def write_processed_images(
    images: list[RawImage],
    output_dir: Path,
    split_name: str,
    image_size: int,
    crop_scale: float,
) -> int:
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "OpenCV is not installed. Run: pip install -r requirements.txt"
        ) from error

    written = 0
    for index, raw_image in enumerate(images):
        image = cv2.imread(str(raw_image.path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {raw_image.path}", file=sys.stderr)
            continue

        cropped = center_square_crop(image, crop_scale)
        resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)

        safe_group = raw_image.group.replace("/", "__").replace(".", "_")
        output_name = f"{safe_group}__{raw_image.path.stem}__{index:05d}.jpg"
        output_path = output_dir / split_name / raw_image.label / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"Could not write processed image: {output_path}")

        written += 1

    return written


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def count_by_label(images: list[RawImage]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for image in images:
        counts[image.label] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    args = parse_args()

    try:
        if not 0 < args.crop_scale <= 1:
            raise ValueError("--crop-scale must be greater than 0 and at most 1.")
        if args.image_size < 16:
            raise ValueError("--image-size must be at least 16.")
        if args.min_frame_gap < 1:
            raise ValueError("--min-frame-gap must be at least 1.")

        raw_images = collect_raw_images(args.raw_photo_dir)
        if not raw_images:
            raise ValueError(f"No raw images found under {args.raw_photo_dir}.")

        train, val = split_by_group(raw_images, args.val_ratio, args.seed)
        train = keep_spaced_frames(train, args.min_frame_gap)
        val = keep_spaced_frames(val, args.min_frame_gap)

        if not args.no_balance:
            train = balance_by_label(train, args.seed)
            val = balance_by_label(val, args.seed + 1)

        ensure_output_dir(args.output_dir, args.overwrite)

        train_written = write_processed_images(
            train, args.output_dir, "train", args.image_size, args.crop_scale
        )
        val_written = write_processed_images(
            val, args.output_dir, "val", args.image_size, args.crop_scale
        )

    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Processed dataset: {args.output_dir}")
    print(f"Train images: {train_written} {count_by_label(train)}")
    print(f"Validation images: {val_written} {count_by_label(val)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
