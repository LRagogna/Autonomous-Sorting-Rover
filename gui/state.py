"""Build the global status payload the whole GUI reads (sidebar + checklist).

Everything here is derived live from the filesystem + the small meta JSON files,
so the sidebar always reflects the true state of the project.
"""

from __future__ import annotations

from ml import dataset_utils as du
from gui import jobs


def _dataset_stems() -> dict[str, list[str]]:
    stems: dict[str, list[str]] = {"train": [], "val": []}
    for split in du.SPLITS:
        directory = du.DATASET_IMAGES_DIR / split
        if directory.exists():
            stems[split] = [
                p.stem for p in directory.iterdir()
                if p.is_file() and p.suffix.lower() in du.IMAGE_EXTENSIONS
            ]
    return stems


def _count_prefixed(directory, prefix: str) -> int:
    if not directory.exists():
        return 0
    return sum(1 for p in directory.iterdir() if p.is_file() and p.name.startswith(prefix))


def build_state() -> dict:
    project = du.load_project()
    classes = du.class_names()
    review = du.load_review_state()

    stems_by_split = _dataset_stems()
    all_stems = stems_by_split["train"] + stems_by_split["val"]

    def state_of(stem: str) -> str:
        entry = review.get(stem)
        return entry.get("state", "unreviewed") if isinstance(entry, dict) else "unreviewed"

    per_class = []
    totals = {"images": 0, "reviewed": 0, "unreviewed": 0, "failed": 0,
              "frames": 0, "clips": 0}
    for name in classes:
        prefix = f"{name}__"
        cls_stems = [s for s in all_stems if s.startswith(prefix)]
        reviewed = sum(1 for s in cls_stems if state_of(s) in {"passed", "edited"})
        unreviewed = sum(1 for s in cls_stems if state_of(s) == "unreviewed")
        failed = _count_prefixed(du.REJECTED_IMAGES_DIR, prefix)
        class_video_dir = du.RAW_VIDEOS_DIR / name
        clips = sum(1 for p in class_video_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in du.VIDEO_EXTENSIONS) \
            if class_video_dir.exists() else 0
        frames = du.count_images(du.FRAMES_DIR / name)
        per_class.append({
            "class": name, "images": len(cls_stems), "reviewed": reviewed,
            "unreviewed": unreviewed, "failed": failed, "clips": clips, "frames": frames,
        })
        totals["images"] += len(cls_stems)
        totals["reviewed"] += reviewed
        totals["unreviewed"] += unreviewed
        totals["failed"] += failed
        totals["frames"] += frames
        totals["clips"] += clips

    versions = du.model_version_files()
    latest_version = max((int(du._MODEL_VERSION_RE.match(p.name).group(1)) for p in versions), default=0)
    retrain_pending = du.count_images(du.RETRAIN_IMAGES_DIR)

    labels_exist = any((du.DATASET_LABELS_DIR / s).exists() and any((du.DATASET_LABELS_DIR / s).iterdir())
                       for s in du.SPLITS)
    checklist = {
        "clipsUploaded": totals["clips"] > 0,
        "framesExtracted": totals["frames"] > 0,
        "labelsGenerated": labels_exist,
        "labelsReviewed": totals["images"] > 0 and totals["unreviewed"] == 0,
        "datasetBuilt": du.DATASET_YAML.exists() and totals["images"] > 0,
        "modelTrained": len(versions) > 0,
        "modelTested": bool(project.get("tested")),
        "modelDeployed": bool(project.get("deployed")) or (du.DEPLOY_DIR / "active_model.pt").exists(),
    }

    return {
        "project": {"name": project.get("name", "Autonomous Sorting Rover")},
        "classes": classes,
        "perClass": per_class,
        "totals": {"classes": len(classes), "retrainPending": retrain_pending, **totals},
        "model": {
            "active": du.active_model_name(),
            "latestVersion": latest_version,
            "count": len(versions),
        },
        "checklist": checklist,
        "job": jobs.public_job(jobs.current_job()),
    }
