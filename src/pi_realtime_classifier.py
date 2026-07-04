"""Real-time Raspberry Pi camera classifier for rover objects.

Run on the Raspberry Pi:

    ./scripts/run_pi_classifier.sh

Press q in the preview window to quit. When running over SSH without a desktop:

    ./scripts/run_pi_classifier.sh --headless
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from detect_objects import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_WRENCH_OVERRIDE_METADATA_PATH,
    DEFAULT_WRENCH_OVERRIDE_MODEL_PATH,
    center_square_crop,
    load_metadata,
    load_model,
    load_optional_wrench_override,
    predict_label_from_image,
)
from edgetpu_classifier import (  # noqa: E402
    DEFAULT_EDGETPU_LABELS_PATH,
    DEFAULT_EDGETPU_MODEL_PATH,
    EdgeTpuClassifier,
)


DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_BACKEND = "edgetpu"
DEFAULT_CLASSIFY_EVERY_FRAMES = 2
DEFAULT_SMOOTHING_WINDOW = 7
DEFAULT_CROP_SCALE = 0.85
DEFAULT_MAX_OBJECTS = 3
DEFAULT_MIN_OBJECT_AREA_RATIO = 0.01
DEFAULT_MAX_OBJECT_AREA_RATIO = 0.70
DEFAULT_BOX_PADDING = 18
DEFAULT_PROPOSAL_WIDTH = 480


@dataclass(frozen=True)
class FramePrediction:
    label: str
    primary_label: str
    wrench_override: bool
    override_applied: bool
    variant_counts: dict[str, int]
    backend: str
    score: float | None = None
    bbox: tuple[int, int, int, int] | None = None
    proposal_score: float = 0.0
    source: str = "frame"


@dataclass(frozen=True)
class ObjectCandidate:
    bbox: tuple[int, int, int, int]
    score: float
    source: str


@dataclass
class ClassifierRuntime:
    backend: str
    model: Any = None
    metadata: dict[str, object] | None = None
    wrench_override: object | None = None
    edgetpu_classifier: EdgeTpuClassifier | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify rover objects live from the Raspberry Pi camera."
    )
    parser.add_argument(
        "--backend",
        choices=("opencv", "edgetpu"),
        default=DEFAULT_BACKEND,
        help=(
            "Inference backend. opencv uses the current .yml SVM on CPU; "
            "edgetpu requires a compiled .tflite model and runs inference on Coral. "
            f"Defaults to {DEFAULT_BACKEND}."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Trained multiclass OpenCV SVM model.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Multiclass model metadata JSON.",
    )
    parser.add_argument(
        "--wrench-override-model-path",
        type=Path,
        default=DEFAULT_WRENCH_OVERRIDE_MODEL_PATH,
        help="Optional wrench override model.",
    )
    parser.add_argument(
        "--wrench-override-metadata-path",
        type=Path,
        default=DEFAULT_WRENCH_OVERRIDE_METADATA_PATH,
        help="Optional wrench override metadata JSON.",
    )
    parser.add_argument(
        "--edgetpu-model-path",
        type=Path,
        default=DEFAULT_EDGETPU_MODEL_PATH,
        help="Edge TPU compiled .tflite classifier model.",
    )
    parser.add_argument(
        "--edgetpu-labels-path",
        type=Path,
        default=DEFAULT_EDGETPU_LABELS_PATH,
        help="Labels file for the Edge TPU classifier.",
    )
    parser.add_argument(
        "--disable-wrench-override",
        action="store_true",
        help="Only use the multiclass classifier.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Camera frame width. Defaults to {DEFAULT_WIDTH}.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Camera frame height. Defaults to {DEFAULT_HEIGHT}.",
    )
    parser.add_argument(
        "--classify-every-frames",
        type=int,
        default=DEFAULT_CLASSIFY_EVERY_FRAMES,
        help=(
            "Run classification every N camera frames. "
            f"Defaults to {DEFAULT_CLASSIFY_EVERY_FRAMES}."
        ),
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW,
        help=(
            "Number of recent predictions used for majority-vote smoothing. "
            f"Defaults to {DEFAULT_SMOOTHING_WINDOW}."
        ),
    )
    parser.add_argument(
        "--detection-mode",
        choices=("objects", "frame"),
        default="objects",
        help=(
            "objects finds likely object regions, classifies those crops, and "
            "draws boxes. frame keeps the old whole-frame classifier."
        ),
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=DEFAULT_MAX_OBJECTS,
        help=f"Maximum object boxes to classify per pass. Defaults to {DEFAULT_MAX_OBJECTS}.",
    )
    parser.add_argument(
        "--min-object-area-ratio",
        type=float,
        default=DEFAULT_MIN_OBJECT_AREA_RATIO,
        help=(
            "Ignore proposed boxes smaller than this fraction of the frame. "
            f"Defaults to {DEFAULT_MIN_OBJECT_AREA_RATIO}."
        ),
    )
    parser.add_argument(
        "--max-object-area-ratio",
        type=float,
        default=DEFAULT_MAX_OBJECT_AREA_RATIO,
        help=(
            "Ignore proposed boxes larger than this fraction of the frame. "
            f"Defaults to {DEFAULT_MAX_OBJECT_AREA_RATIO}."
        ),
    )
    parser.add_argument(
        "--box-padding",
        type=int,
        default=DEFAULT_BOX_PADDING,
        help=f"Pixels to pad around each proposed object box. Defaults to {DEFAULT_BOX_PADDING}.",
    )
    parser.add_argument(
        "--proposal-width",
        type=int,
        default=DEFAULT_PROPOSAL_WIDTH,
        help=(
            "Downscaled width used for fast CPU region proposals. "
            f"Defaults to {DEFAULT_PROPOSAL_WIDTH}."
        ),
    )
    parser.add_argument(
        "--crop-mode",
        choices=("auto", "always", "never", "vote"),
        default="vote",
        help=(
            "How to crop before classification. "
            "vote tries full-frame and center crops, then votes. Defaults to vote."
        ),
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=DEFAULT_CROP_SCALE,
        help=f"Center crop scale. Defaults to {DEFAULT_CROP_SCALE}.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a preview window; print predictions to the terminal.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames. Useful for smoke tests.",
    )
    return parser.parse_args()


def require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Run: pip install -r requirements.txt")


def import_picamera2():
    try:
        from picamera2 import Picamera2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Picamera2 is not installed. On Raspberry Pi OS run: "
            "sudo apt install python3-picamera2"
        ) from exc

    return Picamera2


def crop_variants(frame_bgr, crop_mode: str, crop_scale: float) -> list:
    if crop_mode == "never":
        return [frame_bgr]

    if crop_mode == "always":
        return [center_square_crop(frame_bgr, crop_scale)]

    height, width = frame_bgr.shape[:2]
    if crop_mode == "auto":
        if height == width:
            return [frame_bgr]
        return [center_square_crop(frame_bgr, crop_scale)]

    # Vote mode is intentionally a little redundant. The full frame helps when
    # the object is large or off-center; center crops help when the table/floor
    # background dominates the frame.
    return [
        frame_bgr,
        center_square_crop(frame_bgr, crop_scale),
        center_square_crop(frame_bgr, 1.0),
    ]


def apply_wrench_override(
    label: str,
    frame_bgr,
    wrench_override,
) -> tuple[str, bool, bool]:
    if wrench_override is None:
        return label, False, False

    override_model, override_metadata = wrench_override
    override_label = predict_label_from_image(override_model, override_metadata, frame_bgr)
    wrench_override_result = override_label == "wrench"
    override_candidates = set(override_metadata.get("override_candidates", []))

    if wrench_override_result and label in override_candidates:
        return "wrench", True, True

    return label, wrench_override_result, False


def choose_vote_label(labels: list[str], preferred_label: str) -> str:
    counts = Counter(labels)
    most_common = counts.most_common()
    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        return most_common[0][0]

    # If all variants disagree, prefer the center-crop result because the rover
    # workflow usually places the object near the middle of the camera view.
    return preferred_label


def box_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def clamp_box(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(x1 + 1, min(frame_width, x2))
    y2 = max(y1 + 1, min(frame_height, y2))
    return x1, y1, x2, y2


def pad_box(
    bbox: tuple[int, int, int, int],
    padding: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return clamp_box(
        (x1 - padding, y1 - padding, x2 + padding, y2 + padding),
        frame_width,
        frame_height,
    )


def box_overlap_ratio(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    intersection = box_area((ix1, iy1, ix2, iy2))
    if intersection == 0:
        return 0.0

    smaller_area = max(1, min(box_area(left), box_area(right)))
    return intersection / smaller_area


def crop_box(frame_bgr, bbox: tuple[int, int, int, int]):
    x1, y1, x2, y2 = bbox
    return frame_bgr[y1:y2, x1:x2]


def center_fallback_candidate(frame_bgr, crop_scale: float) -> ObjectCandidate:
    height, width = frame_bgr.shape[:2]
    side = int(min(height, width) * crop_scale)
    side = max(1, min(side, height, width))
    center_x = width // 2
    center_y = height // 2
    x1 = max(0, center_x - side // 2)
    y1 = max(0, center_y - side // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)
    bbox = clamp_box((x1, y1, x2, y2), width, height)
    return ObjectCandidate(bbox=bbox, score=float(box_area(bbox)), source="center-fallback")


def propose_object_candidates(
    frame_bgr,
    max_objects: int,
    min_area_ratio: float,
    max_area_ratio: float,
    box_padding: int,
    proposal_width: int,
) -> list[ObjectCandidate]:
    height, width = frame_bgr.shape[:2]
    frame_area = float(width * height)
    min_area = frame_area * min_area_ratio
    max_area = frame_area * max_area_ratio

    scale = 1.0
    work_bgr = frame_bgr
    if proposal_width > 0 and width > proposal_width:
        scale = proposal_width / width
        work_height = max(1, int(round(height * scale)))
        work_bgr = cv2.resize(
            frame_bgr,
            (proposal_width, work_height),
            interpolation=cv2.INTER_AREA,
        )

    work_height, work_width = work_bgr.shape[:2]
    gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    local_contrast = cv2.absdiff(gray, cv2.medianBlur(gray, 31))
    _, contrast_mask = cv2.threshold(local_contrast, 18, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 40, 120)
    mask = cv2.bitwise_or(edges, contrast_mask)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    large_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.dilate(mask, small_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, large_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[ObjectCandidate] = []
    scale_back = 1.0 / scale

    for contour in contours:
        contour_area = cv2.contourArea(contour) * scale_back * scale_back
        if contour_area < min_area * 0.15:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)
        x1 = int(round(x * scale_back))
        y1 = int(round(y * scale_back))
        x2 = int(round((x + box_width) * scale_back))
        y2 = int(round((y + box_height) * scale_back))
        bbox = pad_box((x1, y1, x2, y2), box_padding, width, height)

        area = box_area(bbox)
        if area < min_area or area > max_area:
            continue

        candidate_width = bbox[2] - bbox[0]
        candidate_height = bbox[3] - bbox[1]
        if candidate_width < 18 or candidate_height < 18:
            continue

        aspect_ratio = max(candidate_width, candidate_height) / max(
            1,
            min(candidate_width, candidate_height),
        )
        if aspect_ratio > 14:
            continue

        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        dx = abs(center_x - width / 2.0) / max(1.0, width / 2.0)
        dy = abs(center_y - height / 2.0) / max(1.0, height / 2.0)
        center_bias = max(0.0, 1.0 - min(1.0, (dx * dx + dy * dy) ** 0.5))
        score = contour_area + area * (0.35 + 0.35 * center_bias)
        candidates.append(ObjectCandidate(bbox=bbox, score=score, source="contour"))

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)

    merged: list[ObjectCandidate] = []
    for candidate in candidates:
        if any(box_overlap_ratio(candidate.bbox, kept.bbox) > 0.62 for kept in merged):
            continue
        merged.append(candidate)
        if len(merged) >= max_objects:
            break

    return merged


def classify_frame(
    frame_bgr,
    runtime: ClassifierRuntime,
    crop_mode: str,
    crop_scale: float,
    bbox: tuple[int, int, int, int] | None = None,
    proposal_score: float = 0.0,
    source: str = "frame",
) -> FramePrediction:
    variants = crop_variants(frame_bgr, crop_mode, crop_scale)
    primary_labels: list[str] = []
    final_labels: list[str] = []
    wrench_override_results: list[bool] = []
    override_applied_results: list[bool] = []
    scores: list[float] = []

    for variant in variants:
        if runtime.backend == "edgetpu":
            if runtime.edgetpu_classifier is None:
                raise RuntimeError("Edge TPU backend selected without a classifier.")
            primary_label, score = runtime.edgetpu_classifier.predict(variant)
            final_label = primary_label
            wrench_override_result = False
            override_applied = False
            scores.append(score)
        else:
            if runtime.model is None or runtime.metadata is None:
                raise RuntimeError("OpenCV backend selected without a classifier.")
            primary_label = predict_label_from_image(runtime.model, runtime.metadata, variant)
            final_label, wrench_override_result, override_applied = apply_wrench_override(
                primary_label,
                variant,
                runtime.wrench_override,
            )

        primary_labels.append(primary_label)
        final_labels.append(final_label)
        wrench_override_results.append(wrench_override_result)
        override_applied_results.append(override_applied)

    center_index = min(1, len(final_labels) - 1)
    final_label = choose_vote_label(final_labels, final_labels[center_index])
    primary_label = choose_vote_label(primary_labels, primary_labels[center_index])

    return FramePrediction(
        label=final_label,
        primary_label=primary_label,
        wrench_override=any(wrench_override_results),
        override_applied=any(override_applied_results),
        variant_counts=dict(sorted(Counter(final_labels).items())),
        backend=runtime.backend,
        score=max(scores) if scores else None,
        bbox=bbox,
        proposal_score=proposal_score,
        source=source,
    )


def classify_objects_in_frame(
    frame_bgr,
    runtime: ClassifierRuntime,
    args: argparse.Namespace,
) -> list[FramePrediction]:
    if args.detection_mode == "frame":
        return [
            classify_frame(
                frame_bgr,
                runtime,
                args.crop_mode,
                args.crop_scale,
            )
        ]

    candidates = propose_object_candidates(
        frame_bgr,
        args.max_objects,
        args.min_object_area_ratio,
        args.max_object_area_ratio,
        args.box_padding,
        args.proposal_width,
    )
    if not candidates:
        candidates = [center_fallback_candidate(frame_bgr, args.crop_scale)]

    predictions: list[FramePrediction] = []
    for candidate in candidates[: args.max_objects]:
        crop = crop_box(frame_bgr, candidate.bbox)
        if crop.size == 0:
            continue
        predictions.append(
            classify_frame(
                crop,
                runtime,
                "never",
                args.crop_scale,
                bbox=candidate.bbox,
                proposal_score=candidate.score,
                source=candidate.source,
            )
        )

    return predictions


def smooth_label(recent_labels: deque[str]) -> str:
    return Counter(recent_labels).most_common(1)[0][0]


def draw_crop_box(frame_bgr, crop_scale: float) -> None:
    height, width = frame_bgr.shape[:2]
    side = int(min(height, width) * crop_scale)
    center_x = width // 2
    center_y = height // 2
    x1 = max(0, center_x - side // 2)
    y1 = max(0, center_y - side // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 180, 0), 2)


def format_prediction_label(prediction: FramePrediction) -> str:
    label = prediction.label
    if prediction.score is not None:
        label += f" {prediction.score:.2f}"
    if prediction.override_applied:
        label += f" ({prediction.primary_label})"
    return label


def draw_object_box(frame_bgr, prediction: FramePrediction) -> None:
    if prediction.bbox is None:
        return

    x1, y1, x2, y2 = prediction.bbox
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)

    label = format_prediction_label(prediction)
    text_scale = 0.62
    text_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_thickness,
    )
    label_height = text_height + baseline + 8
    label_y1 = y1 - label_height
    label_y2 = y1
    text_y = y1 - baseline - 4
    if label_y1 < 0:
        label_y1 = y1
        label_y2 = min(frame_bgr.shape[0], y1 + label_height)
        text_y = min(frame_bgr.shape[0] - 4, y1 + text_height + 4)

    label_x2 = min(frame_bgr.shape[1], x1 + text_width + 12)
    cv2.rectangle(frame_bgr, (x1, label_y1), (label_x2, label_y2), (0, 255, 0), -1)
    cv2.putText(
        frame_bgr,
        label,
        (x1 + 6, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (0, 35, 0),
        text_thickness,
        cv2.LINE_AA,
    )


def summarize_predictions(predictions: list[FramePrediction]) -> str:
    if not predictions:
        return "warming up"

    counts = Counter(prediction.label for prediction in predictions)
    return ", ".join(
        label if count == 1 else f"{label} x{count}"
        for label, count in counts.most_common()
    )


def format_headless_predictions(predictions: list[FramePrediction]) -> str:
    if not predictions:
        return "none"

    parts = []
    for prediction in predictions:
        if prediction.bbox is None:
            box_text = "frame"
        else:
            x1, y1, x2, y2 = prediction.bbox
            box_text = f"{x1},{y1},{x2},{y2}"
        parts.append(f"{format_prediction_label(prediction)} [{box_text}]")

    return "; ".join(parts)


def draw_prediction_overlay(
    frame_bgr,
    predictions: list[FramePrediction],
    smoothed_label: str | None,
    fps: float,
    detection_mode: str,
    crop_mode: str,
    crop_scale: float,
) -> None:
    if detection_mode == "frame" and crop_mode in {"always", "auto", "vote"}:
        draw_crop_box(frame_bgr, crop_scale)

    for prediction in predictions:
        draw_object_box(frame_bgr, prediction)

    label = smoothed_label or summarize_predictions(predictions)
    text = f"Objects: {label}"
    backend = predictions[0].backend if predictions else "loading"

    cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], 78), (0, 95, 0), -1)
    cv2.putText(
        frame_bgr,
        text,
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"Backend: {backend}   Mode: {detection_mode}   FPS: {fps:.1f}   q quits",
        (14, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 255, 230),
        1,
        cv2.LINE_AA,
    )


def configure_camera(width: int, height: int):
    Picamera2 = import_picamera2()
    camera = Picamera2()
    camera.configure(
        camera.create_preview_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
    )
    return camera


def load_classifier_runtime(args: argparse.Namespace) -> ClassifierRuntime:
    if args.backend == "edgetpu":
        edgetpu_classifier = EdgeTpuClassifier.load(
            args.edgetpu_model_path,
            args.edgetpu_labels_path,
        )
        print(
            "Using Edge TPU backend: "
            f"{args.edgetpu_model_path} via {edgetpu_classifier.runtime_name}",
            flush=True,
        )
        return ClassifierRuntime(
            backend="edgetpu",
            edgetpu_classifier=edgetpu_classifier,
        )

    model = load_model(args.model_path)
    metadata = load_metadata(args.metadata_path)
    wrench_override = load_optional_wrench_override(
        args.wrench_override_model_path,
        args.wrench_override_metadata_path,
        args.disable_wrench_override,
    )
    print("Using OpenCV CPU backend.", flush=True)
    return ClassifierRuntime(
        backend="opencv",
        model=model,
        metadata=metadata,
        wrench_override=wrench_override,
    )


def run_live_classifier(args: argparse.Namespace) -> int:
    require_cv2()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be greater than 0.")
    if args.classify_every_frames < 1:
        raise ValueError("--classify-every-frames must be at least 1.")
    if args.smoothing_window < 1:
        raise ValueError("--smoothing-window must be at least 1.")
    if not 0 < args.crop_scale <= 1:
        raise ValueError("--crop-scale must be greater than 0 and at most 1.")
    if args.max_objects < 1:
        raise ValueError("--max-objects must be at least 1.")
    if not 0 < args.min_object_area_ratio < 1:
        raise ValueError("--min-object-area-ratio must be greater than 0 and less than 1.")
    if not 0 < args.max_object_area_ratio <= 1:
        raise ValueError("--max-object-area-ratio must be greater than 0 and at most 1.")
    if args.min_object_area_ratio >= args.max_object_area_ratio:
        raise ValueError("--min-object-area-ratio must be smaller than --max-object-area-ratio.")
    if args.box_padding < 0:
        raise ValueError("--box-padding must be 0 or greater.")
    if args.proposal_width < 0:
        raise ValueError("--proposal-width must be 0 or greater.")

    runtime = load_classifier_runtime(args)

    camera = configure_camera(args.width, args.height)
    recent_labels: deque[str] = deque(maxlen=args.smoothing_window)
    last_predictions: list[FramePrediction] = []
    frame_count = 0
    fps = 0.0
    last_time = time.monotonic()
    last_printed_summary: str | None = None

    camera.start()
    time.sleep(1.0)
    if args.headless:
        print("Rover object classifier running. Press Ctrl+C to stop.", flush=True)
    else:
        print("Rover object classifier running. Press q in the preview window to stop.")
    try:
        while True:
            frame_rgb = camera.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_count += 1

            if frame_count % args.classify_every_frames == 0:
                last_predictions = classify_objects_in_frame(
                    frame_bgr,
                    runtime,
                    args,
                )
                if last_predictions:
                    recent_labels.append(last_predictions[0].label)

            now = time.monotonic()
            elapsed = now - last_time
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed
            last_time = now

            smoothed = smooth_label(recent_labels) if recent_labels else None
            if args.headless:
                current_summary = format_headless_predictions(last_predictions)
                if current_summary != last_printed_summary:
                    print(f"Detected: {current_summary}", flush=True)
                    last_printed_summary = current_summary
            else:
                draw_prediction_overlay(
                    frame_bgr,
                    last_predictions,
                    smoothed,
                    fps,
                    args.detection_mode,
                    args.crop_mode,
                    args.crop_scale,
                )
                cv2.imshow("Rover Object Classifier", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames is not None and frame_count >= args.max_frames:
                break
    finally:
        camera.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = parse_args()
    try:
        return run_live_classifier(args)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError, IndexError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
