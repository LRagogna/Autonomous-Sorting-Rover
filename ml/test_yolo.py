"""Shared YOLO inference helpers.

Used by the desktop webcam detector and (in Phase 2) the GUI's Test Detector
tab, so model loading + box formatting live in exactly one place.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml import dataset_utils as du  # noqa: E402


_model_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


def resolve_weights(weights: str | Path | None = None) -> Path:
    """Return the weights path to use, defaulting to the active model."""
    if weights:
        path = Path(weights)
        if not path.is_absolute():
            path = du.MODELS_DIR / path.name
        return path
    active = du.active_model_path()
    if active is None:
        raise RuntimeError("No trained model yet. Train a model first, then test it.")
    return active


def load_model(weights: str | Path | None = None):
    """Load and cache a YOLO model by weights path."""
    path = resolve_weights(weights)
    if not path.exists():
        raise RuntimeError(f"Model file not found: {path}")
    key = str(path.resolve())
    with _cache_lock:
        if key in _model_cache:
            return _model_cache[key]
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("The 'ultralytics' package is not installed.") from error
        model = YOLO(str(path))
        _model_cache[key] = model
        return model


def detect_array(frame, conf: float, weights: str | Path | None = None) -> dict:
    """Run detection on a BGR numpy frame; boxes are returned as 0-1 fractions."""
    model = load_model(weights)
    height, width = frame.shape[:2]
    results = model.predict(frame, conf=conf, verbose=False)
    result = results[0]
    names = result.names
    detections: list[dict] = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        detections.append({
            "x": x1 / width,
            "y": y1 / height,
            "w": (x2 - x1) / width,
            "h": (y2 - y1) / height,
            "name": names[int(box.cls[0])],
            "conf": float(box.conf[0]),
        })
    return {"detections": detections, "width": width, "height": height}


def detect_jpeg(image_bytes: bytes, conf: float, weights: str | Path | None = None) -> dict:
    """Decode a JPEG frame and run detection on it."""
    import cv2
    import numpy as np

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode the image frame.")
    return detect_array(frame, conf, weights)
