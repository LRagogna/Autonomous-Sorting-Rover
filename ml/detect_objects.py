"""Classify object pictures with the trained rover object classifier.

This is the inference side of ml/train_classifier.py. It loads:

    models/object_classifier.yml
    models/object_classifier_metadata.json

Then it predicts one object label for each input image.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "object_classifier.yml"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "object_classifier_metadata.json"
DEFAULT_WRENCH_OVERRIDE_MODEL_PATH = PROJECT_ROOT / "models" / "wrench_override.yml"
DEFAULT_WRENCH_OVERRIDE_METADATA_PATH = PROJECT_ROOT / "models" / "wrench_override_metadata.json"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "models" / "object_detections.csv"
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "models" / "object_detections_summary.json"
DEFAULT_ANNOTATED_DIR = PROJECT_ROOT / "models" / "annotated_detections"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class Detection:
    path: Path
    primary_predicted: str
    predicted: str
    wrench_override: bool
    override_applied: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect/classify rover object pictures with the trained model."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Image file or directory of images to classify.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Trained OpenCV SVM model path.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Classifier metadata JSON path.",
    )
    parser.add_argument(
        "--wrench-override-model-path",
        type=Path,
        default=DEFAULT_WRENCH_OVERRIDE_MODEL_PATH,
        help="Optional binary wrench override model path.",
    )
    parser.add_argument(
        "--wrench-override-metadata-path",
        type=Path,
        default=DEFAULT_WRENCH_OVERRIDE_METADATA_PATH,
        help="Optional binary wrench override metadata path.",
    )
    parser.add_argument(
        "--disable-wrench-override",
        action="store_true",
        help="Do not apply the optional binary wrench override model.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Where to write prediction results.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="Where to write count and majority-vote summary results.",
    )
    parser.add_argument(
        "--annotated-dir",
        type=Path,
        default=DEFAULT_ANNOTATED_DIR,
        help="Directory for labeled image copies.",
    )
    parser.add_argument(
        "--no-annotated-images",
        action="store_true",
        help="Only write the CSV and terminal output.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directories recursively.",
    )
    parser.add_argument(
        "--crop-mode",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "Center-crop phone/video frames before classification. "
            "Auto crops non-square images and leaves processed square images alone."
        ),
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=0.85,
        help="Crop size as a fraction of the shorter image side.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_input_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path.resolve()] if is_image_file(input_path) else []

    if not input_path.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    return sorted(path.resolve() for path in input_path.glob(pattern) if is_image_file(path))


def load_metadata(metadata_path: Path) -> dict[str, object]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def load_model(model_path: Path):
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Run: pip install -r requirements.txt")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = cv2.ml.SVM_load(str(model_path))
    if model.empty():
        raise RuntimeError(f"Could not load model: {model_path}")

    return model


def load_optional_wrench_override(
    model_path: Path,
    metadata_path: Path,
    disabled: bool,
):
    if disabled:
        return None

    if not model_path.exists() or not metadata_path.exists():
        return None

    return load_model(model_path), load_metadata(metadata_path)


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


def should_crop(image, crop_mode: str) -> bool:
    if crop_mode == "always":
        return True
    if crop_mode == "never":
        return False

    height, width = image.shape[:2]
    return height != width


def extract_features_from_image(image, image_size: int) -> np.ndarray:
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


def standardize_feature(feature: np.ndarray, metadata: dict[str, object]) -> np.ndarray:
    normalization = metadata.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Metadata is missing normalization values.")

    mean = np.array(normalization["mean"], dtype=np.float32)
    std = np.array(normalization["std"], dtype=np.float32)
    return ((feature - mean) / std).reshape(1, -1).astype(np.float32)


def predict_label_from_image(
    model,
    metadata: dict[str, object],
    image,
) -> str:
    image_size = int(metadata["image_size"])
    class_names = metadata["class_names"]
    if not isinstance(class_names, list):
        raise ValueError("Metadata class_names must be a list.")

    feature = extract_features_from_image(image, image_size)
    standardized = standardize_feature(feature, metadata)
    _, raw_prediction = model.predict(standardized)
    predicted_index = int(raw_prediction.reshape(-1)[0])

    return str(class_names[predicted_index])


def predict_image(
    model,
    metadata: dict[str, object],
    wrench_override,
    image_path: Path,
    crop_mode: str,
    crop_scale: float,
) -> Detection:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if should_crop(image, crop_mode):
        image = center_square_crop(image, crop_scale)

    primary_predicted = predict_label_from_image(model, metadata, image)
    predicted = primary_predicted
    wrench_override_result = False
    override_applied = False

    if wrench_override is not None:
        override_model, override_metadata = wrench_override
        override_prediction = predict_label_from_image(
            override_model,
            override_metadata,
            image,
        )
        wrench_override_result = override_prediction == "wrench"
        override_candidates = set(override_metadata.get("override_candidates", []))
        if wrench_override_result and primary_predicted in override_candidates:
            predicted = "wrench"
            override_applied = True

    return Detection(
        path=image_path,
        primary_predicted=primary_predicted,
        predicted=predicted,
        wrench_override=wrench_override_result,
        override_applied=override_applied,
    )


def write_csv(output_path: Path, detections: list[Detection]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "path",
                "primary_predicted",
                "wrench_override",
                "override_applied",
                "predicted",
            ),
        )
        writer.writeheader()
        for detection in detections:
            writer.writerow(
                {
                    "path": str(detection.path),
                    "primary_predicted": detection.primary_predicted,
                    "wrench_override": detection.wrench_override,
                    "override_applied": detection.override_applied,
                    "predicted": detection.predicted,
                }
            )


def summarize_detections(detections: list[Detection]) -> dict[str, object]:
    counts = Counter(detection.predicted for detection in detections)
    majority_label, majority_count = counts.most_common(1)[0]
    total = len(detections)

    return {
        "total_images": total,
        "majority_prediction": majority_label,
        "majority_count": majority_count,
        "majority_fraction": round(majority_count / total, 4) if total else 0.0,
        "counts": dict(sorted(counts.items())),
    }


def write_summary(output_path: Path, summary: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")


def annotated_name(input_path: Path, index: int) -> str:
    safe_stem = input_path.stem.replace(" ", "_")
    return f"{index:05d}_{safe_stem}_{input_path.suffix.lstrip('.').lower()}.jpg"


def write_annotated_images(
    output_dir: Path,
    detections: list[Detection],
    crop_mode: str,
    crop_scale: float,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for index, detection in enumerate(detections):
        image = cv2.imread(str(detection.path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        if should_crop(image, crop_mode):
            image = center_square_crop(image, crop_scale)

        max_width = 900
        height, width = image.shape[:2]
        if width > max_width:
            scale = max_width / width
            image = cv2.resize(
                image,
                (max_width, int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        label = f"Detected: {detection.predicted}"
        cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), (0, 180, 0), 8)
        cv2.rectangle(image, (0, 0), (image.shape[1], 48), (0, 120, 0), -1)
        cv2.putText(
            image,
            label,
            (16, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        output_path = output_dir / annotated_name(detection.path, index)
        if cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            written += 1

    return written


def main() -> int:
    args = parse_args()

    try:
        if not 0 < args.crop_scale <= 1:
            raise ValueError("--crop-scale must be greater than 0 and at most 1.")

        model = load_model(args.model_path)
        metadata = load_metadata(args.metadata_path)
        wrench_override = load_optional_wrench_override(
            args.wrench_override_model_path,
            args.wrench_override_metadata_path,
            args.disable_wrench_override,
        )
        image_paths = iter_input_images(args.input, args.recursive)

        if not image_paths:
            raise ValueError(f"No image files found at: {args.input}")

        detections = [
            predict_image(
                model,
                metadata,
                wrench_override,
                image_path,
                args.crop_mode,
                args.crop_scale,
            )
            for image_path in image_paths
        ]
        summary = summarize_detections(detections)

        write_csv(args.output_csv, detections)
        write_summary(args.summary_json, summary)
        annotated_count = 0
        if not args.no_annotated_images:
            annotated_count = write_annotated_images(
                args.annotated_dir, detections, args.crop_mode, args.crop_scale
            )

    except (FileNotFoundError, RuntimeError, ValueError, KeyError, IndexError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    for detection in detections:
        detail = ""
        if detection.override_applied:
            detail = f" (primary: {detection.primary_predicted}, wrench override)"
        print(f"{detection.predicted}{detail}\t{detection.path}")

    print(
        "Majority prediction: "
        f"{summary['majority_prediction']} "
        f"({summary['majority_count']}/{summary['total_images']})"
    )
    print(f"Saved predictions: {args.output_csv}")
    print(f"Saved summary: {args.summary_json}")
    if not args.no_annotated_images:
        print(f"Saved annotated image(s): {annotated_count} in {args.annotated_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
