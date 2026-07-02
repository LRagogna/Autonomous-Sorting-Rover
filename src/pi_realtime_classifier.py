"""Real-time Raspberry Pi camera classifier for rover objects.

Run on the Raspberry Pi:

    python3 src/pi_realtime_classifier.py

Press q in the preview window to quit. When running over SSH without a desktop:

    python3 src/pi_realtime_classifier.py --headless
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
DEFAULT_CLASSIFY_EVERY_FRAMES = 5
DEFAULT_SMOOTHING_WINDOW = 7
DEFAULT_CROP_SCALE = 0.85


@dataclass(frozen=True)
class FramePrediction:
    label: str
    primary_label: str
    wrench_override: bool
    override_applied: bool
    variant_counts: dict[str, int]
    backend: str
    score: float | None = None


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
        default="opencv",
        help=(
            "Inference backend. opencv uses the current .yml SVM on CPU; "
            "edgetpu requires a compiled .tflite model and runs inference on Coral."
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


def classify_frame(
    frame_bgr,
    runtime: ClassifierRuntime,
    crop_mode: str,
    crop_scale: float,
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
    )


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


def draw_prediction_overlay(
    frame_bgr,
    prediction: FramePrediction | None,
    smoothed_label: str | None,
    fps: float,
    crop_mode: str,
    crop_scale: float,
) -> None:
    if crop_mode in {"always", "auto", "vote"}:
        draw_crop_box(frame_bgr, crop_scale)

    label = smoothed_label or "warming up"
    text = f"Object: {label}"
    if prediction and prediction.override_applied:
        text += f" (primary: {prediction.primary_label})"

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
        f"Backend: {prediction.backend if prediction else 'loading'}   FPS: {fps:.1f}   q quits",
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

    runtime = load_classifier_runtime(args)

    camera = configure_camera(args.width, args.height)
    recent_labels: deque[str] = deque(maxlen=args.smoothing_window)
    last_prediction: FramePrediction | None = None
    frame_count = 0
    fps = 0.0
    last_time = time.monotonic()
    last_printed_label: str | None = None

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
                last_prediction = classify_frame(
                    frame_bgr,
                    runtime,
                    args.crop_mode,
                    args.crop_scale,
                )
                recent_labels.append(last_prediction.label)

            now = time.monotonic()
            elapsed = now - last_time
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed
            last_time = now

            smoothed = smooth_label(recent_labels) if recent_labels else None
            if args.headless:
                if smoothed and smoothed != last_printed_label:
                    detail = ""
                    if last_prediction and last_prediction.override_applied:
                        detail = f" (primary: {last_prediction.primary_label})"
                    print(f"Detected: {smoothed}{detail}", flush=True)
                    last_printed_label = smoothed
            else:
                draw_prediction_overlay(
                    frame_bgr,
                    last_prediction,
                    smoothed,
                    fps,
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
