"""Train a YOLO object detector and save it as a new versioned model.

The dataset is built by the Process Dataset step:

    data/yolo_dataset/dataset.yaml

Each training run saves its best weights as the NEXT version:

    models/yolo_detector_v1.pt, v2, v3, ...

so a new run never overwrites a previous good model. The active model (the one
the detector and rover use) is a separate copy, models/active_model.pt, and is
only replaced when you explicitly mark a version active (or for the very first
model, when there is nothing to protect).

USAGE

    python ml/train_yolo.py --epochs 50 --device cpu
    python ml/train_yolo.py --epochs 10 --set-active      # also make it active
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml import dataset_utils as du  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a versioned YOLO object detector.")
    parser.add_argument("--epochs", type=int, default=50, help="Times to show the model the dataset.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training picture size.")
    parser.add_argument("--device", default="cpu", help="'cpu', 'mps', or a GPU id like '0'.")
    parser.add_argument("--batch", type=int, default=8, help="Pictures studied at once.")
    parser.add_argument("--scale", type=float, default=0.8, help="Random zoom augmentation (0-1).")
    parser.add_argument("--base-model", default="yolov8n.pt", help="Pretrained starter model.")
    parser.add_argument("--set-active", action="store_true",
                        help="Also copy the new model to models/active_model.pt.")
    return parser.parse_args()


def read_final_metrics(run_dir: Path) -> dict:
    """Read precision / recall / mAP from a run's results.csv last row."""
    csv_path = run_dir / "results.csv"
    metrics: dict[str, float] = {}
    if not csv_path.exists():
        return metrics
    lines = [line for line in csv_path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        return metrics
    headers = [h.strip() for h in lines[0].split(",")]
    values = [v.strip() for v in lines[-1].split(",")]
    row = dict(zip(headers, values))
    wanted = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    }
    for key, column in wanted.items():
        if column in row:
            try:
                metrics[key] = round(float(row[column]), 4)
            except ValueError:
                pass
    return metrics


def main() -> int:
    args = parse_args()

    if not du.DATASET_YAML.exists():
        print(f"Dataset not found: {du.DATASET_YAML}\nBuild it first in Process Dataset.",
              file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("The 'ultralytics' package is not installed. Run: pip install -r requirements.txt",
              file=sys.stderr)
        return 1

    print(f"Starting from pretrained model: {args.base_model}")
    model = YOLO(args.base_model)

    results = model.train(
        data=str(du.DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        scale=args.scale,
        project=str(du.RUNS_DIR),
        name="detector",
        exist_ok=True,
    )

    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"Training finished but best weights were not found at {best_weights}.",
              file=sys.stderr)
        return 1

    version, target = du.next_model_version()
    du.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(best_weights), str(target))

    metrics = read_final_metrics(run_dir)
    du.register_model(target.name, {
        "version": version,
        "created": time.time(),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "base_model": args.base_model,
        "device": args.device,
        "run_dir": du.repo_relative(run_dir),
        "metrics": metrics,
    })

    # Set active on request, or automatically for the very first model (nothing
    # to protect yet).
    made_active = False
    if args.set_active or du.active_model_path() is None:
        du.set_active_model(target.name)
        made_active = True

    print("\nTraining done.")
    print(f"  saved version:  {target.name}")
    print(f"  detailed run:   {run_dir}")
    if metrics:
        print(f"  precision {metrics.get('precision', '?')}  recall {metrics.get('recall', '?')}"
              f"  mAP50 {metrics.get('map50', '?')}  mAP50-95 {metrics.get('map50_95', '?')}")
    print(f"  active model:   {'updated to this version' if made_active else 'unchanged'}")
    print(f"TRAIN_RESULT version={version} file={target.name} active={made_active}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
