"""Teach a YOLO model to find the rover's objects (wrench, bit, ...).

HOW TO USE THIS FILE

1. First build the dataset (this makes the boxes for every photo):

       ./scripts/process.sh

2. Then train the detector:

       ./scripts/train.sh

   The first run downloads a tiny starter model called "yolov8n.pt" (~6 MB),
   so you need internet the first time.

WHAT THIS FILE DOES

- YOLO is a model that looks at a whole picture and draws boxes around the
  objects it recognizes. It is different from the old classifier, which only
  guessed one label for a cropped square.
- We start from "yolov8n.pt" (n = nano = the smallest, fastest YOLO). It already
  knows how to see edges and shapes, so it learns our specific objects quickly.
- Training shows the model our pictures many times (each pass is one "epoch")
  until it can place boxes on wrenches and bits by itself.
- "nano" plus a small dataset trains in a few minutes on a normal laptop CPU.

WHERE THE RESULT GOES

The best trained weights are copied to:

    models/yolo_detector.pt

That single file is what the live webcam detector loads:

    python src/desktop_yolo_detector.py

USEFUL OPTIONS

    ./scripts/train.sh --epochs 60      # train longer for better boxes
    ./scripts/train.sh --imgsz 512      # smaller pictures = faster, rougher
    DEVICE=cpu ./scripts/train.sh       # force CPU
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


# This file lives in ml/, so parents[1] is the project folder above ml/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The dataset that scripts/process.sh creates.
DATASET_YAML = PROJECT_ROOT / "data" / "labels" / "dataset.yaml"

# Where we save training runs, and the final easy-to-find weights file.
RUNS_DIR = PROJECT_ROOT / "models" / "yolo_runs"
FINAL_WEIGHTS = PROJECT_ROOT / "models" / "yolo_detector.pt"

# The small pretrained starter model we fine-tune from.
BASE_MODEL = "yolov8n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO object detector.")
    parser.add_argument(
        "--epochs", type=int, default=40,
        help="How many times to show the model the whole dataset.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Picture size the model trains on. Bigger is slower but sharper.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Where to train: 'cpu', or a GPU id like '0' if you have one.",
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="How many pictures to study at once. Lower this if you run out of memory.",
    )
    parser.add_argument(
        "--scale", type=float, default=0.8,
        help="How much to randomly zoom pictures in and out during training "
             "(0-1). Higher makes the model see the object at more sizes, which "
             "helps it recognize objects that are far away (small). Default 0.8.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not DATASET_YAML.exists():
        sys.exit(
            f"Dataset not found: {DATASET_YAML}\n"
            "Build it first with:\n"
            "    ./scripts/process.sh"
        )

    # Import here so the friendly error messages above can show even if
    # ultralytics is not installed yet.
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "The 'ultralytics' package is not installed.\n"
            "Install the training tools with:\n"
            "    pip install -r requirements.txt"
        )

    print(f"Starting from the small pretrained model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # This is the actual training. Ultralytics saves everything under RUNS_DIR.
    # 'scale' controls random zoom so the model learns the object at many sizes
    # (important for spotting a far-away, small object).
    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        scale=args.scale,
        project=str(RUNS_DIR),
        name="detector",
        exist_ok=True,
    )

    # After training, ultralytics leaves the best weights at
    # <save_dir>/weights/best.pt. We copy that to one predictable spot.
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        sys.exit(
            f"Training finished but best weights were not found at {best_weights}."
        )

    FINAL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_weights, FINAL_WEIGHTS)

    print("\nTraining done.")
    print(f"  detailed run:   {results.save_dir}")
    print(f"  ready-to-use:   {FINAL_WEIGHTS}")
    print("\nNow point your webcam at an object:")
    print("    ./scripts/run_desktop_detector.sh")


if __name__ == "__main__":
    main()
