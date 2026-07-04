"""Train and export the object classifier used by the Coral Edge TPU.

The older rover classifier in ``ml/train_classifier.py`` saves an OpenCV
``.yml`` model. That model is useful as a CPU fallback, but the Coral TPU cannot
run it. This script trains a small TensorFlow image classifier, converts it to a
fully quantized TensorFlow Lite model, then asks ``edgetpu_compiler`` to make
the Coral-specific file used by ``src/pi_realtime_classifier.py``.

Typical use on a training machine:

    python ml/train_coral_classifier.py --epochs 30

Successful output:

    models/object_classifier_coral.keras
    models/object_classifier.tflite
    models/object_classifier_edgetpu.tflite
    models/object_classifier_labels.txt
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "classification"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_KERAS_MODEL_PATH = DEFAULT_MODEL_DIR / "object_classifier_coral.keras"
DEFAULT_TFLITE_MODEL_PATH = DEFAULT_MODEL_DIR / "object_classifier.tflite"
DEFAULT_EDGETPU_MODEL_PATH = DEFAULT_MODEL_DIR / "object_classifier_edgetpu.tflite"
DEFAULT_LABELS_PATH = DEFAULT_MODEL_DIR / "object_classifier_labels.txt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class DatasetPaths:
    train_dir: Path
    val_dir: Path
    class_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export a Coral Edge TPU object classifier."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Processed dataset root with train/ and val/ folders.",
    )
    parser.add_argument(
        "--keras-model-path",
        type=Path,
        default=DEFAULT_KERAS_MODEL_PATH,
        help="Where to save the trained Keras model.",
    )
    parser.add_argument(
        "--tflite-model-path",
        type=Path,
        default=DEFAULT_TFLITE_MODEL_PATH,
        help="Where to save the quantized TensorFlow Lite model before Coral compile.",
    )
    parser.add_argument(
        "--edgetpu-model-path",
        type=Path,
        default=DEFAULT_EDGETPU_MODEL_PATH,
        help="Where to save the compiled Coral Edge TPU model.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="Where to save class labels for the Coral runtime.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square input size for the neural classifier. Defaults to 224.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size. Defaults to 16.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Training epochs. Defaults to 25.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Adam learning rate. Defaults to 0.001.",
    )
    parser.add_argument(
        "--representative-samples",
        type=int,
        default=120,
        help="Training images used to calibrate int8 quantization. Defaults to 120.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by TensorFlow dataset loading. Defaults to 42.",
    )
    parser.add_argument(
        "--edgetpu-compiler",
        default="edgetpu_compiler",
        help="Name or path of the Coral compiler command.",
    )
    parser.add_argument(
        "--skip-edgetpu-compile",
        action="store_true",
        help="Only write the quantized .tflite file; do not run edgetpu_compiler.",
    )
    return parser.parse_args()


def import_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required to train/export the Coral model.\n\n"
            "Install it on a development machine, then run this script again:\n"
            "  python3 -m pip install tensorflow\n\n"
            "The Raspberry Pi runtime still stays lightweight; it only needs "
            "OpenCV, Picamera2, and the Coral runtime/PyCoral."
        ) from exc

    return tf


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def count_images(label_dir: Path) -> int:
    return sum(1 for path in label_dir.rglob("*") if is_image_file(path))


def load_dataset_paths(processed_dir: Path) -> DatasetPaths:
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation folder not found: {val_dir}")

    class_names = [
        path.name
        for path in sorted(train_dir.iterdir())
        if path.is_dir() and not path.name.startswith(".") and count_images(path) > 0
    ]
    if len(class_names) < 2:
        raise ValueError(
            "At least two training classes are needed for the Coral classifier. "
            f"Found: {', '.join(class_names) or 'none'}"
        )

    missing_val = [
        class_name
        for class_name in class_names
        if not (val_dir / class_name).is_dir() or count_images(val_dir / class_name) == 0
    ]
    if missing_val:
        raise ValueError(
            "Validation images are missing for class(es): " + ", ".join(missing_val)
        )

    return DatasetPaths(train_dir=train_dir, val_dir=val_dir, class_names=class_names)


def make_image_dataset(
    tf,
    directory: Path,
    class_names: list[str],
    image_size: int,
    batch_size: int,
    seed: int,
    shuffle: bool,
):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=shuffle,
        seed=seed,
    )
    return dataset


def build_coral_friendly_model(tf, image_size: int, class_count: int, learning_rate: float):
    keras = tf.keras
    layers = keras.layers

    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)

    for filters in (24, 32, 48, 64):
        x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)
        x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(96, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(class_count, activation="softmax", name="class_scores")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="rover_coral_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def train_model(
    tf,
    dataset_paths: DatasetPaths,
    image_size: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
):
    train_ds = make_image_dataset(
        tf,
        dataset_paths.train_dir,
        dataset_paths.class_names,
        image_size,
        batch_size,
        seed,
        shuffle=True,
    )
    val_ds = make_image_dataset(
        tf,
        dataset_paths.val_dir,
        dataset_paths.class_names,
        image_size,
        batch_size,
        seed,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = build_coral_friendly_model(
        tf,
        image_size,
        len(dataset_paths.class_names),
        learning_rate,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
        )
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
    return model, history, float(val_loss), float(val_accuracy)


def make_representative_dataset(
    tf,
    dataset_paths: DatasetPaths,
    image_size: int,
    sample_count: int,
    seed: int,
):
    dataset = make_image_dataset(
        tf,
        dataset_paths.train_dir,
        dataset_paths.class_names,
        image_size,
        batch_size=1,
        seed=seed,
        shuffle=True,
    )

    def representative_dataset():
        for image_batch, _ in dataset.take(sample_count):
            yield [tf.cast(image_batch, tf.float32)]

    return representative_dataset


def write_labels(labels_path: Path, class_names: list[str]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8") as labels_file:
        for index, class_name in enumerate(class_names):
            labels_file.write(f"{index} {class_name}\n")


def save_keras_model(model, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)


def export_quantized_tflite(
    tf,
    model,
    dataset_paths: DatasetPaths,
    image_size: int,
    representative_samples: int,
    seed: int,
    output_path: Path,
) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(
        tf,
        dataset_paths,
        image_size,
        representative_samples,
        seed,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)


def compiled_model_name(tflite_model_path: Path) -> str:
    return f"{tflite_model_path.stem}_edgetpu.tflite"


def compile_for_edgetpu(
    compiler: str,
    tflite_model_path: Path,
    edgetpu_model_path: Path,
) -> str:
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise RuntimeError(
            "edgetpu_compiler was not found, so the Coral-specific model was "
            "not created.\n\n"
            "Install the Coral compiler on a Linux development machine or on "
            "the Raspberry Pi, then rerun this script:\n"
            "  sudo apt install edgetpu-compiler\n\n"
            "If you only want the uncompiled .tflite file for inspection, rerun "
            "with --skip-edgetpu-compile."
        )

    edgetpu_model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rover_edgetpu_") as temp_dir:
        command = [
            compiler_path,
            "-s",
            "-o",
            temp_dir,
            str(tflite_model_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        compiler_output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            raise RuntimeError(
                "edgetpu_compiler failed.\n\n"
                f"Command: {' '.join(command)}\n\n"
                f"{compiler_output}"
            )

        compiled_path = Path(temp_dir) / compiled_model_name(tflite_model_path)
        if not compiled_path.exists():
            raise RuntimeError(
                "edgetpu_compiler finished but did not create the expected file: "
                f"{compiled_path}"
            )

        shutil.move(str(compiled_path), edgetpu_model_path)
        return compiler_output


def main() -> int:
    args = parse_args()

    try:
        if args.image_size <= 0:
            raise ValueError("--image-size must be greater than 0.")
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be greater than 0.")
        if args.epochs <= 0:
            raise ValueError("--epochs must be greater than 0.")
        if args.learning_rate <= 0:
            raise ValueError("--learning-rate must be greater than 0.")
        if args.representative_samples <= 0:
            raise ValueError("--representative-samples must be greater than 0.")

        tf = import_tensorflow()
        tf.keras.utils.set_random_seed(args.seed)

        dataset_paths = load_dataset_paths(args.processed_dir)
        print(f"Classes: {', '.join(dataset_paths.class_names)}", flush=True)

        model, _, val_loss, val_accuracy = train_model(
            tf,
            dataset_paths,
            args.image_size,
            args.batch_size,
            args.epochs,
            args.learning_rate,
            args.seed,
        )
        save_keras_model(model, args.keras_model_path)
        write_labels(args.labels_path, dataset_paths.class_names)
        export_quantized_tflite(
            tf,
            model,
            dataset_paths,
            args.image_size,
            args.representative_samples,
            args.seed,
            args.tflite_model_path,
        )

        compiler_output = ""
        if not args.skip_edgetpu_compile:
            compiler_output = compile_for_edgetpu(
                args.edgetpu_compiler,
                args.tflite_model_path,
                args.edgetpu_model_path,
            )

    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Validation accuracy: {val_accuracy:.2%}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Saved Keras model: {args.keras_model_path}")
    print(f"Saved quantized TFLite model: {args.tflite_model_path}")
    print(f"Saved labels: {args.labels_path}")
    if args.skip_edgetpu_compile:
        print("Skipped Edge TPU compile.")
    else:
        print(f"Saved Coral Edge TPU model: {args.edgetpu_model_path}")
        if compiler_output:
            print("\nedgetpu_compiler output:")
            print(compiler_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
