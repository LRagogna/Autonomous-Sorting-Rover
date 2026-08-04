"""Tab 5 - Test Detector: run a model live and capture failure frames."""

from __future__ import annotations

from gui import state as state_module
from gui.server import route
from ml import dataset_utils as du
from ml import label_utils as lu
from ml import test_yolo


_tested_marked = False


@route("GET", "/api/detect/models")
def detect_models(req):
    """Model versions available to test, newest first, plus the active one."""
    registry = du.load_registry()
    active_name = registry.get("active")
    models = []
    for path in du.model_version_files():
        info = registry.get("models", {}).get(path.name, {})
        models.append({
            "file": path.name,
            "version": info.get("version"),
            "active": path.name == active_name,
            "metrics": info.get("metrics", {}),
        })
    models.sort(key=lambda m: m.get("version") or 0, reverse=True)

    # Include the active model even when it isn't a version-numbered file
    # (e.g. yolo_detector_final.pt), so it stays selectable here and is
    # preselected. Listed first as the most relevant choice.
    listed = {m["file"] for m in models}
    if active_name and active_name not in listed and (du.MODELS_DIR / active_name).exists():
        info = registry.get("models", {}).get(active_name, {})
        models.insert(0, {
            "file": active_name,
            "version": info.get("version"),
            "active": True,
            "metrics": info.get("metrics", {}),
        })

    return {"models": models, "active": active_name}


@route("POST", "/api/detect")
def run_detect(req):
    """Run detection on one posted JPEG frame; boxes come back as 0-1 fractions."""
    global _tested_marked
    try:
        conf = float(req.q("conf", "0.25"))
    except ValueError:
        conf = 0.25
    conf = min(max(conf, 0.01), 0.99)
    model = req.q("model", "").strip() or None

    result = test_yolo.detect_jpeg(req.body(), conf, weights=model)

    if not _tested_marked:
        du.set_project_flag("tested", True)
        _tested_marked = True
    return result


@route("POST", "/api/detect/save-failure")
def save_failure(req):
    """Save the current frame as a retraining example with failure metadata."""
    fields, files = req.multipart()
    if not files:
        raise ValueError("No frame image was provided.")
    meta = {
        "failureType": fields.get("failureType", "failure"),
        "model": fields.get("model", ""),
        "note": fields.get("note", ""),
    }
    entry = lu.save_retrain_frame(files[0]["content"], meta)
    return {"ok": True, "saved": entry["image"], "counts": lu.retrain_counts(),
            "state": state_module.build_state()}
