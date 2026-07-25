"""Tab 7 - Deploy Model: pick the active model and stage it for the rover."""

from __future__ import annotations

import shutil

from gui import state as state_module
from gui.server import route
from ml import dataset_utils as du


DEPLOY_MODEL = du.DEPLOY_DIR / "active_model.pt"
DEPLOY_CLASSES = du.DEPLOY_DIR / "classes.txt"

# Checklist items the user ticks manually (the rest are derived from disk).
MANUAL_CHECKS = ("cameraReady", "inferenceReady", "roverControlReady")


@route("GET", "/api/deploy/status")
def deploy_status(req):
    registry = du.load_registry()
    versions = [{
        "file": p.name,
        "version": registry.get("models", {}).get(p.name, {}).get("version"),
        "active": p.name == registry.get("active"),
    } for p in du.model_version_files()]
    versions.sort(key=lambda m: m.get("version") or 0, reverse=True)

    active_path = du.active_model_path()
    project = du.load_project()
    manual = project.get("deploy_checklist", {})
    checklist = {
        "modelSelected": active_path is not None,
        "cameraReady": bool(manual.get("cameraReady")),
        "inferenceReady": bool(manual.get("inferenceReady")),
        "roverControlReady": bool(manual.get("roverControlReady")),
        "modelCopied": DEPLOY_MODEL.exists(),
    }
    return {
        "active": du.active_model_name(),
        "activePath": du.repo_relative(active_path) if active_path else None,
        "versions": versions,
        "deployDir": du.repo_relative(du.DEPLOY_DIR),
        "deployedModel": du.repo_relative(DEPLOY_MODEL) if DEPLOY_MODEL.exists() else None,
        "checklist": checklist,
    }


@route("POST", "/api/deploy/activate")
def deploy_activate(req):
    filename = str(req.json().get("file", ""))
    du.set_active_model(filename)
    return {"ok": True, "active": filename, "state": state_module.build_state()}


@route("POST", "/api/deploy/copy")
def deploy_copy(req):
    """Copy the active model (+ class list) into the deploy/ bundle."""
    active = du.active_model_path()
    if active is None:
        raise ValueError("No active model to deploy. Pick one first.")
    du.DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(active), str(DEPLOY_MODEL))
    DEPLOY_CLASSES.write_text("\n".join(du.class_names()) + "\n")

    project = du.load_project()
    project["deployed"] = True
    du.save_project(project)
    return {"ok": True, "deployedModel": du.repo_relative(DEPLOY_MODEL),
            "state": state_module.build_state()}


@route("POST", "/api/deploy/checklist")
def deploy_checklist(req):
    body = req.json()
    key = str(body.get("key", ""))
    if key not in MANUAL_CHECKS:
        raise ValueError("Unknown checklist item.")
    project = du.load_project()
    checklist = project.setdefault("deploy_checklist", {})
    checklist[key] = bool(body.get("value"))
    du.save_project(project)
    return {"ok": True, "checklist": checklist}
