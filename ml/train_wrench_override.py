"""Train a small binary wrench override model.

The main classifier predicts one of the project classes. Early data shows the
most common wrench failure is predicting a wrench as another long metal object,
especially wire or steel_tape. This binary model is a targeted second opinion:

    is this image a wrench or not?

At detection time, ml/detect_objects.py can override wire/steel_tape predictions
to wrench when this model agrees strongly enough.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from detect_objects import center_square_crop, extract_features_from_image
from train_classifier import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_DIR,
    calculate_metrics,
    collect_processed_split,
    standardize,
    standardize_train,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_DIR = PROJECT_ROOT / "data" / "external" / "classification"
DEFAULT_OVERRIDE_MODEL_PATH = PROJECT_ROOT / "models" / "wrench_override.yml"
DEFAULT_OVERRIDE_METADATA_PATH = PROJECT_ROOT / "models" / "wrench_override_metadata.json"
DEFAULT_OVERRIDE_METRICS_PATH = PROJECT_ROOT / "models" / "wrench_override_metrics.json"
DEFAULT_OVERRIDE_PREDICTIONS_PATH = PROJECT_ROOT / "models" / "wrench_override_validation.csv"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
WRENCH_LABEL = "wrench"
NOT_WRENCH_LABEL = "not_wrench"
DEFAULT_OVERRIDE_CANDIDATES = ("steel_tape", "wire")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the wrench override model.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Processed classification dataset directory.",
    )
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=DEFAULT_EXTERNAL_DIR,
        help="Optional external reference images shaped as data/external/classification/<label>/.",
    )
    parser.add_argument(
        "--primary-model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Existing multiclass model used for combined validation metrics.",
    )
    parser.add_argument(
        "--primary-metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Existing multiclass metadata used for combined validation metrics.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_OVERRIDE_MODEL_PATH,
        help="Output path for the binary wrench override model.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_OVERRIDE_METADATA_PATH,
        help="Output path for the binary wrench override metadata.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_OVERRIDE_METRICS_PATH,
        help="Output path for combined validation metrics.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_OVERRIDE_PREDICTIONS_PATH,
        help="Output path for combined validation predictions.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Square image size used for features. Match the primary classifier.",
    )
    parser.add_argument(
        "--external-crop-scale",
        type=float,
        default=0.95,
        help="Centered crop scale for external reference images.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_external_wrench_images(external_dir: Path) -> list[Path]:
    wrench_dir = external_dir / WRENCH_LABEL
    if not wrench_dir.exists():
        return []

    return sorted(path.resolve() for path in wrench_dir.rglob("*") if is_image_file(path))


def extract_features_from_path(
    image_path: Path,
    image_size: int,
    crop_scale: float | None = None,
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if crop_scale is not None:
        image = center_square_crop(image, crop_scale)

    return extract_features_from_image(image, image_size)


def build_binary_training_data(
    processed_dir: Path,
    external_dir: Path,
    image_size: int,
    external_crop_scale: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    samples = collect_processed_split(processed_dir, "train")
    if not samples:
        raise ValueError(f"No processed train images found under {processed_dir}.")

    features: list[np.ndarray] = []
    labels: list[int] = []

    for sample in samples:
        features.append(extract_features_from_path(sample.path, image_size))
        labels.append(1 if sample.label == WRENCH_LABEL else 0)

    external_paths = collect_external_wrench_images(external_dir)
    for image_path in external_paths:
        features.append(
            extract_features_from_path(
                image_path,
                image_size,
                crop_scale=external_crop_scale,
            )
        )
        labels.append(1)

    if not any(labels):
        raise ValueError("No wrench examples found for override training.")

    return (
        np.vstack(features).astype(np.float32),
        np.array(labels, dtype=np.int32),
        len(external_paths),
    )


def train_binary_svm(features: np.ndarray, labels: np.ndarray):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

    if not svm.train(features, cv2.ml.ROW_SAMPLE, labels):
        raise RuntimeError("OpenCV failed to train the wrench override model.")

    return svm


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)
        json_file.write("\n")


def predict_label(model, metadata: dict[str, object], image_path: Path) -> str:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_size = int(metadata["image_size"])
    feature = extract_features_from_image(image, image_size)
    normalization = metadata["normalization"]
    mean = np.array(normalization["mean"], dtype=np.float32)
    std = np.array(normalization["std"], dtype=np.float32)
    standardized = ((feature - mean) / std).reshape(1, -1).astype(np.float32)
    _, raw_prediction = model.predict(standardized)
    class_names = metadata["class_names"]
    return str(class_names[int(raw_prediction.reshape(-1)[0])])


def validate_combined_predictions(
    processed_dir: Path,
    primary_model_path: Path,
    primary_metadata_path: Path,
    override_model,
    override_mean: np.ndarray,
    override_std: np.ndarray,
    image_size: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    val_samples = collect_processed_split(processed_dir, "val")
    if not val_samples:
        raise ValueError(f"No processed validation images found under {processed_dir}.")

    primary_model = cv2.ml.SVM_load(str(primary_model_path))
    primary_metadata = load_json(primary_metadata_path)
    class_names = list(primary_metadata["class_names"])

    true_labels: list[int] = []
    predicted_labels: list[int] = []
    rows: list[dict[str, object]] = []

    for sample in val_samples:
        primary_prediction = predict_label(primary_model, primary_metadata, sample.path)
        feature = extract_features_from_path(sample.path, image_size)
        override_feature = standardize(
            feature.reshape(1, -1).astype(np.float32),
            override_mean,
            override_std,
        )
        _, raw_override = override_model.predict(override_feature)
        is_wrench = int(raw_override.reshape(-1)[0]) == 1

        predicted = primary_prediction
        override_applied = False
        if is_wrench and primary_prediction in DEFAULT_OVERRIDE_CANDIDATES:
            predicted = WRENCH_LABEL
            override_applied = True

        true_labels.append(class_names.index(sample.label))
        predicted_labels.append(class_names.index(predicted))
        rows.append(
            {
                "path": str(sample.path),
                "expected": sample.label,
                "primary_predicted": primary_prediction,
                "wrench_override": is_wrench,
                "override_applied": override_applied,
                "predicted": predicted,
                "correct": sample.label == predicted,
            }
        )

    metrics = calculate_metrics(
        np.array(true_labels, dtype=np.int32),
        np.array(predicted_labels, dtype=np.int32),
        class_names,
    )
    metrics["override_candidates"] = list(DEFAULT_OVERRIDE_CANDIDATES)
    metrics["override_applied_count"] = sum(bool(row["override_applied"]) for row in rows)
    return metrics, rows


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = (
            "path",
            "expected",
            "primary_predicted",
            "wrench_override",
            "override_applied",
            "predicted",
            "correct",
        )
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    try:
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is not installed. Run: pip install -r requirements.txt"
            )

        features, labels, external_count = build_binary_training_data(
            args.processed_dir,
            args.external_dir,
            args.image_size,
            args.external_crop_scale,
        )
        standardized_features, mean, std = standardize_train(features)
        override_model = train_binary_svm(standardized_features, labels)

        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        override_model.save(str(args.model_path))

        metadata = {
            "class_names": [NOT_WRENCH_LABEL, WRENCH_LABEL],
            "image_size": args.image_size,
            "feature_kind": "grayscale_pixels_plus_hsv_histograms",
            "positive_label": WRENCH_LABEL,
            "override_candidates": list(DEFAULT_OVERRIDE_CANDIDATES),
            "normalization": {
                "mean": mean.astype(float).tolist(),
                "std": std.astype(float).tolist(),
            },
            "sample_counts": {
                "train": int(len(labels)),
                "positive": int(labels.sum()),
                "negative": int((labels == 0).sum()),
                "external_positive": external_count,
            },
        }
        write_json(args.metadata_path, metadata)

        metrics, rows = validate_combined_predictions(
            args.processed_dir,
            args.primary_model_path,
            args.primary_metadata_path,
            override_model,
            mean,
            std,
            args.image_size,
        )
        write_json(args.metrics_path, metrics)
        write_predictions(args.predictions_path, rows)

    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Train images: {len(labels)}")
    print(f"External wrench references: {external_count}")
    print(f"Combined validation accuracy: {metrics['accuracy']:.2%}")
    print(f"Wrench recall: {metrics['per_class'][WRENCH_LABEL]['recall']:.2%}")
    print(f"Saved override model: {args.model_path}")
    print(f"Saved override metadata: {args.metadata_path}")
    print(f"Saved override metrics: {args.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
