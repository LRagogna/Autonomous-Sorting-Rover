"""Backtest the live Pi object filter against objects and background crops."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pi_realtime_classifier as live  # noqa: E402


DEFAULT_POSITIVE_DIR = PROJECT_ROOT / "data" / "processed" / "classification" / "val"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class BacktestCounts:
    total: int = 0
    accepted: int = 0
    correct: int = 0
    wrong: int = 0
    rejected: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest live object acceptance against validation and background crops."
    )
    parser.add_argument(
        "--positive-dir",
        type=Path,
        default=DEFAULT_POSITIVE_DIR,
        help="Processed validation folder shaped as val/<label>/<image>.",
    )
    parser.add_argument(
        "--positive-limit",
        type=int,
        default=0,
        help="Limit positive images. 0 means all.",
    )
    parser.add_argument(
        "--background-limit",
        type=int,
        default=0,
        help="Limit background crops. 0 means all generated crops.",
    )
    parser.add_argument(
        "--corner-scale",
        type=float,
        default=0.32,
        help="Fraction of each image side used for corner background crops.",
    )
    parser.add_argument(
        "--min-vote-fraction",
        type=float,
        default=live.DEFAULT_MIN_VOTE_FRACTION,
        help="Minimum crop-vote agreement required for acceptance.",
    )
    parser.add_argument(
        "--min-objectness",
        type=float,
        default=live.DEFAULT_MIN_OBJECTNESS,
        help="Minimum proposal objectness required for acceptance.",
    )
    parser.add_argument(
        "--min-object-area-ratio",
        type=float,
        default=live.DEFAULT_MIN_OBJECT_AREA_RATIO,
        help="Minimum proposed object area ratio.",
    )
    parser.add_argument(
        "--max-object-area-ratio",
        type=float,
        default=live.DEFAULT_MAX_OBJECT_AREA_RATIO,
        help="Maximum proposed object area ratio.",
    )
    parser.add_argument(
        "--min-foreground-ratio",
        type=float,
        default=live.DEFAULT_MIN_FOREGROUND_RATIO,
        help="Minimum foreground ratio required for acceptance.",
    )
    parser.add_argument(
        "--max-foreground-ratio",
        type=float,
        default=live.DEFAULT_MAX_FOREGROUND_RATIO,
        help="Maximum foreground ratio allowed for acceptance.",
    )
    parser.add_argument(
        "--min-edge-density",
        type=float,
        default=live.DEFAULT_MIN_EDGE_DENSITY,
        help="Minimum edge density required for acceptance.",
    )
    parser.add_argument(
        "--max-edge-density",
        type=float,
        default=live.DEFAULT_MAX_EDGE_DENSITY,
        help="Maximum edge density allowed for acceptance.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_positive_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if is_image_file(path))


def expected_label_for_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if not relative.parts:
        raise ValueError(f"Could not infer label for: {path}")
    return relative.parts[0]


def build_live_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        backend="opencv",
        model_path=live.DEFAULT_MODEL_PATH,
        metadata_path=live.DEFAULT_METADATA_PATH,
        wrench_override_model_path=live.DEFAULT_WRENCH_OVERRIDE_MODEL_PATH,
        wrench_override_metadata_path=live.DEFAULT_WRENCH_OVERRIDE_METADATA_PATH,
        disable_wrench_override=True,
        enable_wrench_override=False,
        detection_mode="objects",
        max_objects=1,
        min_object_area_ratio=args.min_object_area_ratio,
        max_object_area_ratio=args.max_object_area_ratio,
        box_padding=live.DEFAULT_BOX_PADDING,
        proposal_width=live.DEFAULT_PROPOSAL_WIDTH,
        crop_scale=live.DEFAULT_CROP_SCALE,
        crop_mode="vote",
        min_vote_fraction=args.min_vote_fraction,
        min_confidence=live.DEFAULT_MIN_CONFIDENCE,
        min_objectness=args.min_objectness,
        min_foreground_ratio=args.min_foreground_ratio,
        max_foreground_ratio=args.max_foreground_ratio,
        min_edge_density=args.min_edge_density,
        max_edge_density=args.max_edge_density,
        edge_margin_ratio=live.DEFAULT_EDGE_MARGIN_RATIO,
        allow_edge_boxes=False,
        fallback_center_box=False,
        show_rejected=False,
        reject_label=list(live.DEFAULT_REJECT_LABELS),
    )


def accepted_predictions(predictions: list[live.FramePrediction]) -> list[live.FramePrediction]:
    return [prediction for prediction in predictions if prediction.accepted]


def corner_crops(image, corner_scale: float):
    height, width = image.shape[:2]
    crop_width = max(16, int(width * corner_scale))
    crop_height = max(16, int(height * corner_scale))
    return [
        image[0:crop_height, 0:crop_width],
        image[0:crop_height, width - crop_width : width],
        image[height - crop_height : height, 0:crop_width],
        image[height - crop_height : height, width - crop_width : width],
    ]


def print_positive_summary(counts: BacktestCounts) -> None:
    accepted_rate = counts.accepted / counts.total if counts.total else 0.0
    precision = counts.correct / counts.accepted if counts.accepted else 0.0
    recall = counts.correct / counts.total if counts.total else 0.0
    print("Positive objects")
    print(f"  total: {counts.total}")
    print(f"  accepted: {counts.accepted} ({accepted_rate:.1%})")
    print(f"  correct accepted: {counts.correct}")
    print(f"  wrong accepted: {counts.wrong}")
    print(f"  rejected: {counts.rejected}")
    print(f"  accepted precision: {precision:.1%}")
    print(f"  accepted recall: {recall:.1%}")


def print_background_summary(total: int, false_positives: int) -> None:
    false_positive_rate = false_positives / total if total else 0.0
    print("Background crops")
    print(f"  total: {total}")
    print(f"  false positives: {false_positives} ({false_positive_rate:.1%})")


def main() -> int:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed.")

    args = parse_args()
    runtime = live.load_classifier_runtime(build_live_args(args))
    live_args = build_live_args(args)

    positive_paths = iter_positive_images(args.positive_dir)
    if args.positive_limit > 0:
        positive_paths = positive_paths[: args.positive_limit]

    positive_counts = BacktestCounts()
    background_total = 0
    background_false_positives = 0

    for image_path in positive_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        expected_label = expected_label_for_path(image_path, args.positive_dir)
        predictions = live.classify_objects_in_frame(image, runtime, live_args)
        accepted = accepted_predictions(predictions)

        positive_counts.total += 1
        if not accepted:
            positive_counts.rejected += 1
        else:
            positive_counts.accepted += 1
            if accepted[0].label == expected_label:
                positive_counts.correct += 1
            else:
                positive_counts.wrong += 1

        for crop in corner_crops(image, args.corner_scale):
            if args.background_limit and background_total >= args.background_limit:
                continue
            crop_predictions = live.classify_objects_in_frame(crop, runtime, live_args)
            background_total += 1
            if accepted_predictions(crop_predictions):
                background_false_positives += 1

    print_positive_summary(positive_counts)
    print_background_summary(background_total, background_false_positives)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
