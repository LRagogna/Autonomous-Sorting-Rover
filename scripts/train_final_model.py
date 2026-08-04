"""Train the FINAL YOLO detector.

Same pipeline as ml/train_yolo.py (same dataset, same base weights, same
registry bookkeeping), but tuned for maximum accuracy on this machine and
saved under a fixed name instead of an auto-incremented version:

    models/yolo_detector_final.pt   (best weights)

It is registered in models/registry.json as "yolo_detector_final.pt" and
made the active model.

Power comes from: MPS acceleration (Apple Silicon), a long training budget
with early stopping (patience), a larger batch, cosine LR decay, and the
project's standard zoom augmentation. Base model stays yolov8n per the
current pipeline so it still deploys to the rover's Pi.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml import dataset_utils as du  # noqa: E402
from ml.train_yolo import read_final_metrics  # noqa: E402

FINAL_NAME = "yolo_detector_final.pt"

# Maximum-power settings for this dataset on an M4 Pro.
EPOCHS = 200          # generous budget; early stopping ends it when it plateaus
PATIENCE = 60         # stop if no val improvement for this many epochs
IMGSZ = 640
BATCH = 32
DEVICE = "mps"
SCALE = 0.8           # project-standard random zoom augmentation
BASE_MODEL = "yolov8n.pt"


def main() -> int:
    if not du.DATASET_YAML.exists():
        print(f"Dataset not found: {du.DATASET_YAML}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    print(f"Starting from pretrained model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    results = model.train(
        data=str(du.DATASET_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        device=DEVICE,
        batch=BATCH,
        scale=SCALE,
        cos_lr=True,          # smooth cosine LR decay over the long schedule
        project=str(du.RUNS_DIR),
        name="detector_final",
        exist_ok=True,
        seed=0,
        plots=True,
    )

    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"Training finished but best weights missing at {best_weights}.", file=sys.stderr)
        return 1

    target = du.MODELS_DIR / FINAL_NAME
    du.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(best_weights), str(target))

    metrics = read_final_metrics(run_dir)
    du.register_model(FINAL_NAME, {
        "version": "final",
        "final": True,
        "created": time.time(),
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "base_model": BASE_MODEL,
        "device": DEVICE,
        "cos_lr": True,
        "run_dir": du.repo_relative(run_dir),
        "metrics": metrics,
    })

    du.set_active_model(FINAL_NAME)

    print("\nFINAL training done.")
    print(f"  saved:        {target}")
    print(f"  detailed run: {run_dir}")
    if metrics:
        print(f"  precision {metrics.get('precision', '?')}  recall {metrics.get('recall', '?')}"
              f"  mAP50 {metrics.get('map50', '?')}  mAP50-95 {metrics.get('map50_95', '?')}")
    print(f"  active model: updated to {FINAL_NAME}")
    print(f"TRAIN_RESULT file={FINAL_NAME} active=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
