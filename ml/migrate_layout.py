"""One-time migration from the old data layout to the new control-center layout.

OLD                                   NEW
  data/raw/clips/<class>/<video>   -> data/raw_videos/<class>/<video>
  data/raw/photos/<class>/*.jpg    -> data/frames/<class>/<class>__*.jpg
  data/labels/<class>/images/...   -> data/yolo_dataset/images/...
  data/labels/<class>/labels/...   -> data/yolo_dataset/labels/...
  data/labels/<class>/excluded/... -> data/rejected/...
  data/labels/object_classes.txt   -> data/meta/object_classes.txt
  data/labels/.frame_marks.json    -> data/meta/review_state.json
  models/yolo_detector.pt          -> models/yolo_detector_v1.pt (+ active_model.pt)

Safe to run more than once: it skips anything already migrated. Small metadata
files are backed up to backups/migrate_<timestamp>/ first. Image/video/label
files are MOVED (rename within the same disk = no copy, no data loss), and the
now-empty old folders are removed at the end.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml import dataset_utils as du  # noqa: E402


def _move_file(src: Path, dst: Path) -> bool:
    """Move one file, creating parents. Skip (keep dst) if dst already exists."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return False
    shutil.move(str(src), str(dst))
    return True


def backup_metadata(backup_dir: Path) -> None:
    """Copy the small, hard-to-recreate files before touching anything."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        du.LEGACY_LABELS_DIR / "object_classes.txt",
        du.LEGACY_LABELS_DIR / "dataset.yaml",
        du.LEGACY_LABELS_DIR / ".frame_marks.json",
        du.LEGACY_PHOTOS_DIR / ".processed_clips.json",
        du.LEGACY_PHOTOS_DIR / ".deleted_frames.json",
    ]
    saved = 0
    for path in candidates:
        if path.exists():
            shutil.copy2(str(path), str(backup_dir / path.name))
            saved += 1
    print(f"  backed up {saved} metadata file(s) to {du.repo_relative(backup_dir)}")


def migrate_videos() -> int:
    moved = 0
    if not du.LEGACY_CLIPS_DIR.exists():
        return moved
    for class_dir in sorted(p for p in du.LEGACY_CLIPS_DIR.iterdir() if p.is_dir()):
        for video in sorted(p for p in class_dir.iterdir() if p.is_file() and not p.name.startswith(".")):
            if _move_file(video, du.RAW_VIDEOS_DIR / class_dir.name / video.name):
                moved += 1
    print(f"  moved {moved} video(s) -> data/raw_videos/")
    return moved


def migrate_frames() -> int:
    """Move raw photos into data/frames, adding the '<class>__' name prefix."""
    moved = 0
    if not du.LEGACY_PHOTOS_DIR.exists():
        return moved
    for class_dir in sorted(p for p in du.LEGACY_PHOTOS_DIR.iterdir() if p.is_dir()):
        cls = class_dir.name
        for frame in sorted(p for p in class_dir.rglob("*")
                            if p.is_file() and p.suffix.lower() in du.IMAGE_EXTENSIONS):
            name = frame.name if frame.name.startswith(f"{cls}__") else f"{cls}__{frame.name}"
            if _move_file(frame, du.FRAMES_DIR / cls / name):
                moved += 1
    print(f"  moved {moved} frame(s) -> data/frames/")
    return moved


def migrate_dataset() -> int:
    """Move per-class dataset images/labels into the unified yolo_dataset."""
    moved = 0
    if not du.LEGACY_LABELS_DIR.exists():
        return moved
    for class_dir in sorted(p for p in du.LEGACY_LABELS_DIR.iterdir() if p.is_dir()):
        for split in du.SPLITS:
            for img in sorted((class_dir / "images" / split).glob("*")):
                if img.is_file() and img.suffix.lower() in du.IMAGE_EXTENSIONS:
                    if _move_file(img, du.DATASET_IMAGES_DIR / split / img.name):
                        moved += 1
            for lbl in sorted((class_dir / "labels" / split).glob("*.txt")):
                if lbl.is_file():
                    _move_file(lbl, du.DATASET_LABELS_DIR / split / lbl.name)
    print(f"  moved {moved} dataset image(s) -> data/yolo_dataset/")
    return moved


def migrate_rejected() -> int:
    """Move per-class excluded frames into data/rejected/."""
    moved = 0
    if not du.LEGACY_LABELS_DIR.exists():
        return moved
    for class_dir in sorted(p for p in du.LEGACY_LABELS_DIR.iterdir() if p.is_dir()):
        excluded = class_dir / "excluded"
        for img in sorted((excluded / "images").glob("*")) if (excluded / "images").exists() else []:
            if img.is_file() and img.suffix.lower() in du.IMAGE_EXTENSIONS:
                if _move_file(img, du.REJECTED_IMAGES_DIR / img.name):
                    moved += 1
        for lbl in sorted((excluded / "labels").glob("*.txt")) if (excluded / "labels").exists() else []:
            _move_file(lbl, du.REJECTED_LABELS_DIR / lbl.name)
    print(f"  moved {moved} rejected frame(s) -> data/rejected/")
    return moved


def migrate_classes() -> None:
    src = du.LEGACY_LABELS_DIR / "object_classes.txt"
    if src.exists() and not du.CLASSES_FILE.exists():
        du.CLASSES_FILE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(du.CLASSES_FILE))
        print(f"  class list -> {du.repo_relative(du.CLASSES_FILE)}")


def convert_frame_marks() -> int:
    """Convert old .frame_marks.json into the new review_state.json."""
    if du.REVIEW_STATE_FILE.exists():
        return 0
    old = du.read_json(du.LEGACY_LABELS_DIR / ".frame_marks.json", {})
    marks = old.get("marks", {}) if isinstance(old, dict) else {}
    mapping = {"pass": "passed", "fail": "failed"}
    frames: dict[str, dict] = {}
    for key, entry in marks.items():
        stem = key.split("/", 1)[1] if "/" in key else key
        state = mapping.get((entry or {}).get("state"))
        if not state:
            continue
        # Find where the frame actually lives now to record the right split.
        from ml import label_utils as lu

        _, location = lu.locate_frame(stem)
        split = location if location in du.SPLITS else (entry or {}).get("split", "train")
        frames[stem] = {"state": state, "split": split, "class": du.class_from_stem(stem)}
    if frames:
        du.save_review_state(frames)
    print(f"  converted {len(frames)} review decision(s) -> review_state.json")
    return len(frames)


def rebuild_processed_clips() -> None:
    """Rebuild processed_clips.json by scanning the migrated folders."""
    clips: dict[str, dict] = {}
    if du.RAW_VIDEOS_DIR.exists():
        for class_dir in sorted(p for p in du.RAW_VIDEOS_DIR.iterdir() if p.is_dir()):
            cls = class_dir.name
            for video in sorted(p for p in class_dir.iterdir()
                                if p.is_file() and p.suffix.lower() in du.VIDEO_EXTENSIONS):
                stem = du.clip_stem_for_video(video)
                prefix = f"{cls}__{stem}__frame_"
                frame_count = 0
                fdir = du.FRAMES_DIR / cls
                if fdir.exists():
                    frame_count = sum(1 for p in fdir.iterdir()
                                      if p.is_file() and p.name.startswith(prefix))
                labeled = any(
                    (du.DATASET_IMAGES_DIR / split).exists()
                    and any(p.name.startswith(prefix) for p in (du.DATASET_IMAGES_DIR / split).iterdir())
                    for split in du.SPLITS
                )
                clips[du.clip_key(cls, video.name)] = {
                    "class": cls,
                    "video": video.name,
                    "path": du.repo_relative(video),
                    "frames": frame_count,
                    "duration": round(du.video_duration_seconds(video), 2),
                    "labeled": labeled,
                    "status": "labeled" if labeled else ("extracted" if frame_count else "new"),
                }
    du.save_processed_clips(clips)
    print(f"  rebuilt processed-clips manifest ({len(clips)} clip(s))")


def migrate_models() -> None:
    if du.model_version_files():
        return  # already versioned
    if du.LEGACY_MODEL.exists():
        target = du.MODELS_DIR / "yolo_detector_v1.pt"
        shutil.move(str(du.LEGACY_MODEL), str(target))
        metrics = {}
        run_csv = du.RUNS_DIR / "detector" / "results.csv"
        if run_csv.exists():
            from ml import train_yolo

            metrics = train_yolo.read_final_metrics(du.RUNS_DIR / "detector")
        du.register_model(target.name, {
            "version": 1,
            "created": time.time(),
            "base_model": "yolov8n.pt",
            "run_dir": du.repo_relative(du.RUNS_DIR / "detector"),
            "metrics": metrics,
            "note": "migrated from models/yolo_detector.pt",
        })
        du.set_active_model(target.name)
        print("  models/yolo_detector.pt -> yolo_detector_v1.pt (+ active_model.pt)")


def cleanup_old_dirs() -> None:
    """Remove now-empty legacy folders (review previews are regenerated live)."""
    for legacy in (du.LEGACY_CLIPS_DIR.parent, du.LEGACY_LABELS_DIR):
        if legacy.exists():
            shutil.rmtree(legacy, ignore_errors=True)
    print("  removed legacy data/raw and data/labels folders")


def already_migrated() -> bool:
    return du.CLASSES_FILE.exists() and not du.LEGACY_LABELS_DIR.exists()


def main() -> int:
    if already_migrated():
        print("Already migrated to the new layout. Nothing to do.")
        return 0
    if not du.LEGACY_LABELS_DIR.exists() and not du.LEGACY_CLIPS_DIR.exists():
        print("No legacy data found; creating an empty new-layout skeleton.")
        du.ensure_core_dirs()
        du.write_dataset_yaml()
        return 0

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    backup_dir = du.BACKUPS_DIR / f"migrate_{ts}"
    print(f"Migrating to the new data layout (backup: {du.repo_relative(backup_dir)})")
    du.ensure_core_dirs()
    backup_metadata(backup_dir)
    migrate_classes()
    migrate_videos()
    migrate_frames()
    migrate_dataset()
    migrate_rejected()
    convert_frame_marks()
    rebuild_processed_clips()
    migrate_models()
    du.write_dataset_yaml()
    cleanup_old_dirs()
    print("\nMigration complete. New layout is ready under data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
