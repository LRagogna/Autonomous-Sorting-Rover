"""Train and validate a small image classifier for rover object sorting.

This is a starter training pipeline for the project while the dataset is still
small. It uses OpenCV instead of a heavy deep-learning framework so the code can
run before the project commits to a final model format.

Supported dataset layouts:

1. Raw extracted frames, grouped by source video:

       data/raw/photos/washer/pan_01.avi/frame_000000.png
       data/raw/photos/bolt/pan_01.avi/frame_000000.png

   In this layout, the script splits by video folder so near-identical frames
   from one video do not leak into both train and validation.

2. Processed classification folders:

       data/processed/classification/train/washer/image_001.png
       data/processed/classification/val/washer/image_002.png

   In this layout, the existing train/val split is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PHOTO_DIR = PROJECT_ROOT / "data" / "raw" / "photos"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "classification"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "object_classifier.yml"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "object_classifier_metadata.json"
DEFAULT_METRICS_PATH = DEFAULT_MODEL_DIR / "object_classifier_metrics.json"
DEFAULT_PREDICTIONS_PATH = DEFAULT_MODEL_DIR / "object_classifier_validation.csv"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ImageSample:
    """One labeled image and the group it came from."""

    path: Path
    label: str
    group: str


@dataclass(frozen=True)
class DatasetSplits:
    """Train, validation, and optional test samples."""

    train: list[ImageSample]
    val: list[ImageSample]
    test: list[ImageSample]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and validate an OpenCV object classifier."
    )
    parser.add_argument(
        "--dataset",
        choices=("auto", "raw", "processed"),
        default="auto",
        help="Dataset layout to use. Defaults to auto.",
    )
    parser.add_argument(
        "--raw-photo-dir",
        type=Path,
        default=DEFAULT_RAW_PHOTO_DIR,
        help="Raw extracted photos directory. Defaults to data/raw/photos.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Processed classification directory. Defaults to data/processed/classification.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path for the trained OpenCV SVM model.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path for label and normalization metadata.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path for validation metrics JSON.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path for per-image validation predictions CSV.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Square image size used for features. Defaults to 64.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio for raw-photo auto splitting. Defaults to 0.2.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Optional test ratio for raw-photo auto splitting. Defaults to 0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for raw-photo splitting. Defaults to 42.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_raw_samples(raw_photo_dir: Path) -> list[ImageSample]:
    """Collect samples from data/raw/photos/<label>/<source_video>/<image>."""
    if not raw_photo_dir.exists():
        return []

    samples: list[ImageSample] = []
    for label_dir in sorted(raw_photo_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("."):
            continue

        label = label_dir.name
        for source_dir in sorted(label_dir.iterdir()):
            if not source_dir.is_dir() or source_dir.name.startswith("."):
                continue

            group = f"{label}/{source_dir.name}"
            for image_path in sorted(source_dir.iterdir()):
                if is_image_file(image_path):
                    samples.append(ImageSample(image_path.resolve(), label, group))

    return samples


def collect_processed_split(processed_dir: Path, split_name: str) -> list[ImageSample]:
    """Collect samples from data/processed/classification/<split>/<label>/<image>."""
    split_dir = processed_dir / split_name
    if not split_dir.exists():
        return []

    samples: list[ImageSample] = []
    for label_dir in sorted(split_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("."):
            continue

        label = label_dir.name
        for image_path in sorted(label_dir.rglob("*")):
            if is_image_file(image_path):
                group = f"{split_name}/{label}/{image_path.parent.name}"
                samples.append(ImageSample(image_path.resolve(), label, group))

    return samples


def load_processed_splits(processed_dir: Path) -> DatasetSplits:
    return DatasetSplits(
        train=collect_processed_split(processed_dir, "train"),
        val=collect_processed_split(processed_dir, "val"),
        test=collect_processed_split(processed_dir, "test"),
    )


def split_raw_samples(
    samples: list[ImageSample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetSplits:
    """Split raw samples by source group and class to avoid frame leakage."""
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be at least 0 and less than 1.")

    if not 0 <= test_ratio < 1:
        raise ValueError("--test-ratio must be at least 0 and less than 1.")

    if val_ratio + test_ratio >= 1:
        raise ValueError("--val-ratio plus --test-ratio must be less than 1.")

    groups_by_label: dict[str, dict[str, list[ImageSample]]] = {}
    for sample in samples:
        label_groups = groups_by_label.setdefault(sample.label, {})
        label_groups.setdefault(sample.group, []).append(sample)

    train: list[ImageSample] = []
    val: list[ImageSample] = []
    test: list[ImageSample] = []

    rng = random.Random(seed)

    for label in sorted(groups_by_label):
        label_groups = groups_by_label[label]
        group_names = sorted(label_groups)
        rng.shuffle(group_names)

        total_groups = len(group_names)
        test_count = int(round(total_groups * test_ratio))
        val_count = int(round(total_groups * val_ratio))

        # Keep one validation group per class when possible. Without this, a
        # random split can make validation accuracy meaningless for small data.
        if val_ratio > 0 and total_groups >= 2 and val_count == 0:
            val_count = 1

        if test_count + val_count >= total_groups and total_groups > 1:
            val_count = max(1, total_groups - test_count - 1)

        test_groups = set(group_names[:test_count])
        val_groups = set(group_names[test_count : test_count + val_count])

        for group_name in group_names:
            destination = train
            if group_name in test_groups:
                destination = test
            elif group_name in val_groups:
                destination = val

            destination.extend(label_groups[group_name])

    return DatasetSplits(train=train, val=val, test=test)


def choose_dataset(args: argparse.Namespace) -> tuple[str, DatasetSplits]:
    processed_splits = load_processed_splits(args.processed_dir)
    raw_samples = collect_raw_samples(args.raw_photo_dir)

    if args.dataset == "processed":
        return "processed", processed_splits

    if args.dataset == "raw":
        return (
            "raw",
            split_raw_samples(raw_samples, args.val_ratio, args.test_ratio, args.seed),
        )

    if processed_splits.train or processed_splits.val:
        return "processed", processed_splits

    return (
        "raw",
        split_raw_samples(raw_samples, args.val_ratio, args.test_ratio, args.seed),
    )


def validate_splits(splits: DatasetSplits, dataset_name: str) -> None:
    all_samples = splits.train + splits.val + splits.test
    if not all_samples:
        raise ValueError(
            "No training images found yet.\n\n"
            "Add raw extracted frames like:\n"
            f"  {DEFAULT_RAW_PHOTO_DIR}/washer/pan_01.avi/frame_000000.png\n\n"
            "Then run:\n"
            "  python ml/train_classifier.py --dataset raw\n\n"
            "Or create processed folders like:\n"
            f"  {DEFAULT_PROCESSED_DIR}/train/washer/image_001.png\n"
            f"  {DEFAULT_PROCESSED_DIR}/val/washer/image_002.png"
        )

    train_labels = {sample.label for sample in splits.train}
    val_labels = {sample.label for sample in splits.val}

    if not splits.train:
        raise ValueError(f"The {dataset_name} dataset has no training images.")

    if len(train_labels) < 2:
        raise ValueError(
            "At least two object classes are needed for classifier training. "
            f"Found: {', '.join(sorted(train_labels)) or 'none'}."
        )

    if not splits.val:
        raise ValueError(
            "No validation images were found. Add another source video per class "
            "or create data/processed/classification/val/<label>/ folders."
        )

    missing_val_labels = train_labels - val_labels
    if missing_val_labels:
        raise ValueError(
            "Validation data is missing class(es): "
            f"{', '.join(sorted(missing_val_labels))}."
        )


def extract_features(image_path: Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    gray_features = gray.astype(np.float32).reshape(-1) / 255.0

    histograms = []
    for channel_index, bins, value_range in (
        (0, 32, [0, 180]),
        (1, 32, [0, 256]),
        (2, 32, [0, 256]),
    ):
        histogram = cv2.calcHist([hsv], [channel_index], None, [bins], value_range)
        histogram = cv2.normalize(histogram, None).reshape(-1)
        histograms.append(histogram.astype(np.float32))

    return np.concatenate([gray_features, *histograms]).astype(np.float32)


def build_feature_matrix(
    samples: list[ImageSample],
    class_to_index: dict[str, int],
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = [extract_features(sample.path, image_size) for sample in samples]
    labels = [class_to_index[sample.label] for sample in samples]
    return np.vstack(features).astype(np.float32), np.array(labels, dtype=np.int32)


def standardize_train(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    return ((features - mean) / std).astype(np.float32), mean, std


def standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


def train_svm(features: np.ndarray, labels: np.ndarray) -> cv2.ml_SVM:
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

    if not svm.train(features, cv2.ml.ROW_SAMPLE, labels):
        raise RuntimeError("OpenCV failed to train the classifier.")

    return svm


def predict(svm: cv2.ml_SVM, features: np.ndarray) -> np.ndarray:
    _, raw_predictions = svm.predict(features)
    return raw_predictions.reshape(-1).astype(np.int32)


def calculate_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    total = int(len(true_labels))
    correct = int((true_labels == predicted_labels).sum())
    confusion = np.zeros((len(class_names), len(class_names)), dtype=int)

    for expected, predicted in zip(true_labels, predicted_labels):
        confusion[int(expected), int(predicted)] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for index, class_name in enumerate(class_names):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        support = int(confusion[index, :].sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        per_class[class_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "correct": correct,
        "total": total,
        "classes": class_names,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def write_predictions(
    output_path: Path,
    samples: list[ImageSample],
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("path", "source_group", "expected", "predicted", "correct"),
        )
        writer.writeheader()
        for sample, expected, predicted in zip(samples, true_labels, predicted_labels):
            writer.writerow(
                {
                    "path": str(sample.path),
                    "source_group": sample.group,
                    "expected": class_names[int(expected)],
                    "predicted": class_names[int(predicted)],
                    "correct": bool(expected == predicted),
                }
            )


def write_json(output_path: Path, data: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)
        json_file.write("\n")


def main() -> int:
    args = parse_args()

    try:
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is not installed. Run: pip install -r requirements.txt"
            )

        dataset_name, splits = choose_dataset(args)
        validate_splits(splits, dataset_name)

        class_names = sorted({sample.label for sample in splits.train})
        class_to_index = {class_name: index for index, class_name in enumerate(class_names)}

        train_features, train_labels = build_feature_matrix(
            splits.train, class_to_index, args.image_size
        )
        val_features, val_labels = build_feature_matrix(
            splits.val, class_to_index, args.image_size
        )

        train_features, mean, std = standardize_train(train_features)
        val_features = standardize(val_features, mean, std)

        svm = train_svm(train_features, train_labels)
        val_predictions = predict(svm, val_features)
        metrics = calculate_metrics(val_labels, val_predictions, class_names)

        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        svm.save(str(args.model_path))

        metadata = {
            "dataset": dataset_name,
            "class_names": class_names,
            "image_size": args.image_size,
            "feature_kind": "grayscale_pixels_plus_hsv_histograms",
            "normalization": {
                "mean": mean.astype(float).tolist(),
                "std": std.astype(float).tolist(),
            },
            "sample_counts": {
                "train": len(splits.train),
                "val": len(splits.val),
                "test": len(splits.test),
            },
        }

        write_json(args.metadata_path, metadata)
        write_json(args.metrics_path, metrics)
        write_predictions(
            args.predictions_path,
            splits.val,
            val_labels,
            val_predictions,
            class_names,
        )

    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Dataset: {dataset_name}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Train images: {len(splits.train)}")
    print(f"Validation images: {len(splits.val)}")
    print(f"Validation accuracy: {metrics['accuracy']:.2%}")
    print(f"Saved model: {args.model_path}")
    print(f"Saved metadata: {args.metadata_path}")
    print(f"Saved metrics: {args.metrics_path}")
    print(f"Saved validation predictions: {args.predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
