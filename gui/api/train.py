"""Tab 4 - Train Model: build the split, train a versioned model, view metrics."""

from __future__ import annotations

import sys

from gui import jobs
from gui import state as state_module
from gui.server import route, send_image
from ml import dataset_utils as du


# Training-run plot images the GUI is allowed to display.
ALLOWED_PLOTS = {
    "results.png", "BoxPR_curve.png", "BoxF1_curve.png",
    "confusion_matrix.png", "labels.jpg",
}


@route("GET", "/api/models")
def list_models(req):
    registry = du.load_registry()
    models = []
    for path in du.model_version_files():
        info = registry.get("models", {}).get(path.name, {})
        models.append({
            "file": path.name,
            "version": info.get("version"),
            "created": info.get("created"),
            "epochs": info.get("epochs"),
            "metrics": info.get("metrics", {}),
            "active": path.name == registry.get("active"),
            "size": du.format_size(path.stat().st_size),
        })
    models.sort(key=lambda m: m.get("version") or 0, reverse=True)
    available_plots = [
        name for name in ALLOWED_PLOTS
        if (du.RUNS_DIR / "detector" / name).exists()
    ]
    return {"models": models, "active": registry.get("active"), "plots": available_plots}


@route("GET", "/api/models/plot/{name}")
def model_plot(req):
    name = du.safe_filename(req.params["name"])
    if name not in ALLOWED_PLOTS:
        send_image(req.handler, du.RUNS_DIR / "detector" / "missing.png")  # -> 404
        return
    send_image(req.handler, du.RUNS_DIR / "detector" / name)


@route("POST", "/api/train")
def start_training(req):
    body = req.json()
    if du.count_images(du.DATASET_IMAGES_DIR / "train") == 0:
        raise ValueError("No training images yet. Process and review a dataset first.")

    val_fraction = float(body.get("valFraction", 0.2) or 0.2)
    val_fraction = min(max(val_fraction, 0.05), 0.5)
    du.resplit_dataset(val_fraction)
    du.write_dataset_yaml()

    epochs = max(1, int(body.get("epochs", 50) or 50))
    imgsz = int(body.get("imgsz", 640) or 640)
    batch = int(body.get("batch", 8) or 8)
    device = str(body.get("device", "cpu") or "cpu").strip()
    base_model = du.safe_filename(str(body.get("baseModel", "yolov8n.pt") or "yolov8n.pt"))

    command = [
        sys.executable, "-u", "ml/train_yolo.py",
        "--epochs", str(epochs), "--imgsz", str(imgsz), "--batch", str(batch),
        "--device", device, "--base-model", base_model,
    ]
    if body.get("setActive"):
        command.append("--set-active")

    job = jobs.start_job("train_model", command)
    return {"ok": True, "job": jobs.public_job(job), "split": {"valFraction": val_fraction}}


@route("POST", "/api/models/activate")
def activate_model(req):
    filename = str(req.json().get("file", ""))
    du.set_active_model(filename)
    return {"ok": True, "active": filename, "state": state_module.build_state()}
