"""Edge TPU image classification helper.

This module is intentionally separate from the OpenCV SVM classifier. The Coral
TPU cannot run OpenCV .yml models; it runs TensorFlow Lite models compiled for
the Edge TPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EDGETPU_MODEL_PATH = PROJECT_ROOT / "models" / "object_classifier_edgetpu.tflite"
DEFAULT_EDGETPU_LABELS_PATH = PROJECT_ROOT / "models" / "object_classifier_labels.txt"


def load_labels(labels_path: Path) -> dict[int, str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"Edge TPU labels file not found: {labels_path}")

    labels: dict[int, str] = {}
    next_index = 0
    with labels_path.open("r", encoding="utf-8") as labels_file:
        for raw_line in labels_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.replace(":", " ", 1).split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1].strip()
            else:
                labels[next_index] = line
                next_index += 1

    if not labels:
        raise ValueError(f"No labels found in: {labels_path}")

    return labels


def make_interpreter(model_path: Path):
    """Create a TensorFlow Lite interpreter with the Edge TPU delegate."""
    if not model_path.exists():
        raise FileNotFoundError(
            "Edge TPU model file not found: "
            f"{model_path}\n\n"
            "The current OpenCV .yml model cannot run on the Coral TPU. "
            "Train/export a quantized TensorFlow Lite model, compile it with "
            "edgetpu_compiler, and place it at this path."
        )

    try:
        from pycoral.utils.edgetpu import make_interpreter as pycoral_make_interpreter

        interpreter = pycoral_make_interpreter(str(model_path))
        interpreter.allocate_tensors()
        return interpreter, "pycoral"
    except ModuleNotFoundError:
        pass

    try:
        import tflite_runtime.interpreter as tflite
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Edge TPU runtime is not installed. On Raspberry Pi OS, install "
            "the Coral runtime and PyCoral, then run this again."
        ) from exc

    try:
        interpreter = tflite.Interpreter(
            model_path=str(model_path),
            experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")],
        )
    except ValueError as exc:
        raise RuntimeError(
            "Could not load the Edge TPU delegate. Make sure the Coral USB/M.2 "
            "accelerator is connected and the libedgetpu runtime is installed."
        ) from exc

    interpreter.allocate_tensors()
    return interpreter, "tflite_runtime+libedgetpu"


@dataclass
class EdgeTpuClassifier:
    interpreter: object
    labels: dict[int, str]
    runtime_name: str

    @classmethod
    def load(cls, model_path: Path, labels_path: Path) -> "EdgeTpuClassifier":
        if cv2 is None:
            raise RuntimeError("OpenCV is needed for camera frame resizing.")

        interpreter, runtime_name = make_interpreter(model_path)
        return cls(
            interpreter=interpreter,
            labels=load_labels(labels_path),
            runtime_name=runtime_name,
        )

    def input_size(self) -> tuple[int, int]:
        input_details = self.interpreter.get_input_details()[0]
        shape = input_details["shape"]
        return int(shape[2]), int(shape[1])

    def _prepare_input(self, frame_bgr) -> np.ndarray:
        input_details = self.interpreter.get_input_details()[0]
        input_width, input_height = self.input_size()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            frame_rgb,
            (input_width, input_height),
            interpolation=cv2.INTER_AREA,
        )

        dtype = input_details["dtype"]
        if dtype == np.float32:
            tensor = resized.astype(np.float32) / 255.0
        elif dtype == np.int8:
            # Most Edge TPU image classifiers use uint8 input. This branch keeps
            # int8 models usable by mapping image bytes into the signed range.
            tensor = (resized.astype(np.int16) - 128).astype(np.int8)
        else:
            tensor = resized.astype(dtype)

        return np.expand_dims(tensor, axis=0)

    def predict(self, frame_bgr) -> tuple[str, float]:
        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]

        self.interpreter.set_tensor(input_details["index"], self._prepare_input(frame_bgr))
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(output_details["index"])[0]
        scale, zero_point = output_details.get("quantization", (0.0, 0))
        if scale:
            scores = scale * (output.astype(np.float32) - zero_point)
        else:
            scores = output.astype(np.float32)

        class_index = int(np.argmax(scores))
        score = float(scores[class_index])
        return self.labels.get(class_index, str(class_index)), score
