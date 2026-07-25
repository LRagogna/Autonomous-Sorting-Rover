"""Danger Zone - Wipe Data / Start Over (guarded, multi-confirmation)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from gui import state as state_module
from gui.server import route
from ml import dataset_utils as du


WIPE_PHRASE = "WIPE DATA"

# Folders cleared on a wipe (the dataset + everything derived from it).
DATA_TARGETS = [
    du.RAW_VIDEOS_DIR, du.FRAMES_DIR, du.DATASET_DIR,
    du.REJECTED_DIR, du.RETRAIN_DIR, du.DEPLOY_DIR,
]
# Small state files cleared on a wipe (object_classes.txt is kept).
STATE_FILES = [du.REVIEW_STATE_FILE, du.PROCESSED_CLIPS_FILE, du.DURATIONS_FILE]


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


@route("GET", "/api/maintenance/targets")
def wipe_targets(req):
    targets = [{
        "path": du.repo_relative(p),
        "size": du.format_size(_dir_size(p)),
        "exists": p.exists(),
    } for p in DATA_TARGETS]
    models = du.model_version_files()
    return {
        "phrase": WIPE_PHRASE,
        "targets": targets,
        "keeps": [du.repo_relative(du.CLASSES_FILE) + " (class list)"],
        "models": {
            "count": len(models),
            "size": du.format_size(sum(p.stat().st_size for p in models)),
        },
    }


def _backup(delete_models: bool) -> str:
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    backup_dir = du.BACKUPS_DIR / f"wipe_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for target in DATA_TARGETS:
        if target.exists():
            shutil.copytree(target, backup_dir / target.name, dirs_exist_ok=True)
    du.META_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(du.META_DIR, backup_dir / "meta", dirs_exist_ok=True)
    if delete_models:
        (backup_dir / "models").mkdir(exist_ok=True)
        for path in du.model_version_files() + [du.ACTIVE_MODEL, du.MODEL_REGISTRY]:
            if path.exists():
                shutil.copy2(str(path), str(backup_dir / "models" / path.name))
    return du.repo_relative(backup_dir)


@route("POST", "/api/maintenance/wipe")
def wipe(req):
    body = req.json()
    # Server-side re-validation: the typed phrase must match EXACTLY.
    if str(body.get("confirm", "")) != WIPE_PHRASE:
        raise ValueError(f'Type exactly "{WIPE_PHRASE}" to confirm.')
    delete_models = bool(body.get("deleteModels"))

    backup_path = _backup(delete_models) if body.get("backup") else None

    for target in DATA_TARGETS:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
    for state_file in STATE_FILES:
        if state_file.exists():
            state_file.unlink()

    # Reset pipeline flags but keep the project name.
    project = du.load_project()
    project.update({"tested": False, "deployed": False, "deploy_checklist": {}})
    du.save_project(project)

    if delete_models:
        for path in du.model_version_files() + [du.ACTIVE_MODEL, du.MODEL_REGISTRY]:
            if path.exists():
                path.unlink()
        if du.RUNS_DIR.exists():
            shutil.rmtree(du.RUNS_DIR, ignore_errors=True)

    # Recreate the empty skeleton so the app keeps working.
    du.ensure_core_dirs()
    du.write_dataset_yaml()

    return {
        "ok": True,
        "backup": backup_path,
        "deletedModels": delete_models,
        "state": state_module.build_state(),
    }
