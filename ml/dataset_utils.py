"""Central paths, config, and dataset helpers for the rover training workflow.

This module is the single source of truth for WHERE everything lives on disk and
for the small JSON files that remember the project's state between runs. Every
other ``ml/`` module and the whole GUI import their paths and helpers from here,
so the folder layout is defined in exactly one place.

FOLDER LAYOUT (the "control center" data model)

    data/
      raw_videos/<class>/<video>                    original recorded clips
      frames/<class>/<class>__<video>__frame_NNN.jpg extracted still frames
      yolo_dataset/
        images/{train,val}/                          pictures YOLO trains on
        labels/{train,val}/                          one .txt box file per picture
        dataset.yaml                                 tells YOLO the folders + classes
      rejected/{images,labels}/                      frames failed in review (recoverable)
      retrain_queue/images/ + <frame>.json           live-test failures to correct later
      meta/
        object_classes.txt                           master "id name" class list
        review_state.json                            per-frame reviewed/passed/failed/edited
        processed_clips.json                         which clips were already extracted
        project.json                                 project name + pipeline flags
    models/
      yolo_detector_v1.pt, v2, ...                   versioned trained weights
      active_model.pt                                copy of the chosen version
      registry.json                                  metrics + metadata per version
    deploy/                                          Raspberry Pi deployment bundle
    backups/<timestamp>/                             migration + wipe backups
"""

from __future__ import annotations

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths. Everything is anchored to PROJECT_ROOT so the tools work no matter
# which folder the terminal (or the GUI subprocess) is started from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
FRAMES_DIR = DATA_DIR / "frames"

DATASET_DIR = DATA_DIR / "yolo_dataset"
DATASET_IMAGES_DIR = DATASET_DIR / "images"
DATASET_LABELS_DIR = DATASET_DIR / "labels"
DATASET_YAML = DATASET_DIR / "dataset.yaml"

REJECTED_DIR = DATA_DIR / "rejected"
REJECTED_IMAGES_DIR = REJECTED_DIR / "images"
REJECTED_LABELS_DIR = REJECTED_DIR / "labels"

RETRAIN_DIR = DATA_DIR / "retrain_queue"
RETRAIN_IMAGES_DIR = RETRAIN_DIR / "images"

META_DIR = DATA_DIR / "meta"
CLASSES_FILE = META_DIR / "object_classes.txt"
REVIEW_STATE_FILE = META_DIR / "review_state.json"
PROCESSED_CLIPS_FILE = META_DIR / "processed_clips.json"
PROJECT_FILE = META_DIR / "project.json"
DURATIONS_FILE = META_DIR / "durations.json"

MODELS_DIR = PROJECT_ROOT / "models"
ACTIVE_MODEL = MODELS_DIR / "active_model.pt"
MODEL_REGISTRY = MODELS_DIR / "registry.json"
RUNS_DIR = MODELS_DIR / "yolo_runs"

DEPLOY_DIR = PROJECT_ROOT / "deploy"
BACKUPS_DIR = PROJECT_ROOT / "backups"

# Legacy locations, used by ml/migrate_layout.py to move old data forward.
LEGACY_CLIPS_DIR = DATA_DIR / "raw" / "clips"
LEGACY_PHOTOS_DIR = DATA_DIR / "raw" / "photos"
LEGACY_LABELS_DIR = DATA_DIR / "labels"
LEGACY_MODEL = MODELS_DIR / "yolo_detector.pt"

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val")

# Per-frame review decisions. "unreviewed" frames have not been looked at yet;
# "passed" and "edited" stay in the training set; "failed" frames are moved to
# data/rejected/ so YOLO never sees them (but they can be restored).
REVIEW_STATES = ("unreviewed", "passed", "failed", "edited")

# Background (negative) images teach the detector what "nothing" looks like so it
# stops drawing boxes on empty scenes. They live in the dataset with an EMPTY
# label file and use this filename prefix so they are never mistaken for a class.
BG_PREFIX = "bg__"

# The user drops PICTURES of empty scenes / plain backgrounds into
# data/raw_videos/background/. This folder name is RESERVED: it is never an object
# class, its images are folded into the dataset as negatives (empty labels), and
# the model must never learn "background" as something to detect.
BACKGROUND_CLASS = "background"
RAW_BACKGROUNDS_DIR = RAW_VIDEOS_DIR / BACKGROUND_CLASS


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def repo_relative(path: Path) -> str:
    """Return a stable repo-relative path (used as manifest keys)."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path, default):
    """Read a small JSON file, returning ``default`` when missing or corrupt."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, data) -> None:
    """Write a small JSON file in a readable, deterministic format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def validate_class_name(name: str) -> str:
    """Return a filesystem-safe object-class label."""
    name = (name or "").strip().replace(" ", "_")
    if not name:
        raise ValueError("Choose or enter an object class name.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Class names can use letters, numbers, _ and - only.")
    if is_background_class(name):
        raise ValueError(
            '"background" is reserved for negative images (data/raw_videos/background/) '
            "and cannot be an object class."
        )
    return name


def safe_filename(filename: str) -> str:
    """Return a safe filename with no folder components."""
    filename = Path(filename).name.strip()
    filename = re.sub(r"[^A-Za-z0-9._ -]+", "_", filename).replace(" ", "_")
    if not filename or filename in {".", ".."}:
        raise ValueError("File has no usable name.")
    return filename


def unique_path(path: Path) -> Path:
    """Return ``path`` or a numbered sibling so nothing is overwritten."""
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free filename for {path.name}")


# ---------------------------------------------------------------------------
# Frame / clip naming. A frame is named "<class>__<video>__frame_NNN.jpg" so it
# is globally unique inside the merged train/val folders and its source video is
# always recoverable.
# ---------------------------------------------------------------------------
def clip_stem_for_video(video_path: Path) -> str:
    """Return the safe prefix used inside extracted frame filenames."""
    return video_path.stem.strip().replace(" ", "_").replace(".", "_")


def frame_stem(class_name: str, video_path: Path, frame_index: int) -> str:
    """Build the '<class>__<video>__frame_NNN' stem for one extracted frame."""
    return f"{class_name}__{clip_stem_for_video(video_path)}__frame_{frame_index:06d}"


def class_from_stem(stem: str) -> str:
    """Recover the class name from a frame/dataset stem."""
    return stem.split("__", 1)[0] if "__" in stem else stem


def is_background_stem(stem: str) -> bool:
    """Return True for background/negative dataset images (empty labels)."""
    return stem.startswith(BG_PREFIX)


def is_background_class(name: str) -> bool:
    """Return True for the reserved ``background`` folder (negatives, not a class)."""
    return (name or "").strip().lower() == BACKGROUND_CLASS


def find_source_video(class_name: str, video_stem: str) -> Path | None:
    """Find the original video file whose frames use ``video_stem``."""
    class_dir = RAW_VIDEOS_DIR / class_name
    if not class_dir.exists():
        return None
    for video in sorted(p for p in class_dir.iterdir() if p.is_file()):
        if clip_stem_for_video(video) == video_stem:
            return video
    return None


def clip_key_for_stem(class_name: str, video_stem: str) -> str:
    """Return the processed-clips key for a class + source-video stem."""
    video = find_source_video(class_name, video_stem)
    return clip_key(class_name, video.name if video else video_stem)


def source_video_from_stem(stem: str) -> str:
    """Recover the source-video stem from a frame/dataset stem.

    "bit__IMG_1297__frame_000045" -> "IMG_1297".
    """
    body = stem.split("__", 1)[1] if "__" in stem else stem
    return body.split("__frame_", 1)[0] if "__frame_" in body else body


# ---------------------------------------------------------------------------
# Object class list (data/meta/object_classes.txt), one "id name" per line.
# ---------------------------------------------------------------------------
def load_classes() -> list[tuple[int, str]]:
    """Read the class list as ordered (id, name) rows."""
    if not CLASSES_FILE.exists():
        return []
    rows: list[tuple[int, str]] = []
    for line in CLASSES_FILE.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            rows.append((int(parts[0]), parts[1]))
        except ValueError:
            continue
    return sorted(rows)


def class_names() -> list[str]:
    """Return class names ordered by their YOLO id."""
    return [name for _, name in load_classes()]


def class_name_to_id() -> dict[str, int]:
    """Return a {name: id} mapping."""
    return {name: class_id for class_id, name in load_classes()}


def ensure_class(name: str) -> int:
    """Append a new class if needed and return its id."""
    name = validate_class_name(name)
    rows = load_classes()
    for class_id, existing in rows:
        if existing == name:
            return class_id
    next_id = max((cid for cid, _ in rows), default=-1) + 1
    CLASSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing_text = CLASSES_FILE.read_text() if CLASSES_FILE.exists() else ""
    if existing_text and not existing_text.endswith("\n"):
        existing_text += "\n"
    CLASSES_FILE.write_text(f"{existing_text}{next_id} {name}\n")
    return next_id


def list_classes_from_disk() -> list[str]:
    """Return every known class: from the class list plus any data folders."""
    names = set(class_names())
    for root in (RAW_VIDEOS_DIR, FRAMES_DIR):
        if root.exists():
            for path in root.iterdir():
                if (path.is_dir() and not path.name.startswith(".")
                        and not is_background_class(path.name)):
                    names.add(path.name)
    return sorted(names)


# ---------------------------------------------------------------------------
# JSON state stores
# ---------------------------------------------------------------------------
def load_processed_clips() -> dict[str, dict]:
    """Load the manifest of clips already extracted into frames."""
    data = read_json(PROCESSED_CLIPS_FILE, {"clips": {}})
    if isinstance(data, dict) and isinstance(data.get("clips"), dict):
        return data["clips"]
    return {}


def save_processed_clips(clips: dict[str, dict]) -> None:
    write_json(PROCESSED_CLIPS_FILE, {"clips": clips})


def clip_key(class_name: str, video_name: str) -> str:
    """Return the processed-clips manifest key for one clip."""
    return f"{class_name}/{video_name}"


def load_review_state() -> dict[str, dict]:
    """Load per-frame review decisions keyed by dataset stem."""
    data = read_json(REVIEW_STATE_FILE, {"frames": {}})
    if isinstance(data, dict) and isinstance(data.get("frames"), dict):
        return data["frames"]
    return {}


def save_review_state(frames: dict[str, dict]) -> None:
    write_json(REVIEW_STATE_FILE, {"frames": frames})


def load_project() -> dict:
    """Load project metadata (name + pipeline flags)."""
    data = read_json(PROJECT_FILE, {})
    if not isinstance(data, dict):
        data = {}
    data.setdefault("name", "Autonomous Sorting Rover")
    data.setdefault("tested", False)
    data.setdefault("deployed", False)
    data.setdefault("deploy_checklist", {})
    return data


def save_project(project: dict) -> None:
    write_json(PROJECT_FILE, project)


def set_project_flag(key: str, value) -> dict:
    """Set one field in project.json and return the updated project."""
    project = load_project()
    project[key] = value
    save_project(project)
    return project


# ---------------------------------------------------------------------------
# Folder scaffolding
# ---------------------------------------------------------------------------
def ensure_core_dirs() -> None:
    """Create the empty folder skeleton the whole workflow relies on."""
    for split in SPLITS:
        (DATASET_IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (DATASET_LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    for path in (
        RAW_VIDEOS_DIR, RAW_BACKGROUNDS_DIR, FRAMES_DIR, REJECTED_IMAGES_DIR,
        REJECTED_LABELS_DIR, RETRAIN_IMAGES_DIR, META_DIR, MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def reset_dataset() -> None:
    """Clear generated dataset outputs for a clean full rebuild.

    Empties yolo_dataset/images, yolo_dataset/labels, and data/rejected, and
    forgets per-frame review decisions. Leaves data/frames and data/raw_videos
    (the sources) untouched.
    """
    import shutil

    for target in (DATASET_IMAGES_DIR, DATASET_LABELS_DIR, REJECTED_DIR):
        if target.exists():
            shutil.rmtree(target)
    if REVIEW_STATE_FILE.exists():
        REVIEW_STATE_FILE.unlink()
    ensure_core_dirs()


def resplit_dataset(val_fraction: float) -> dict:
    """Reassign whole video clips between train and val by ``val_fraction``.

    The split is always by whole source video (never by individual frame), so a
    near-duplicate frame can't leak from train into val. Only frames currently in
    the dataset are affected; rejected frames are left alone. Returns counts.
    """
    import shutil

    ensure_core_dirs()
    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for split in SPLITS:
        directory = DATASET_IMAGES_DIR / split
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                key = (class_from_stem(path.stem), source_video_from_stem(path.stem))
                groups.setdefault(key, []).append((path.stem, split))

    by_class: dict[str, list[str]] = {}
    for cls, video in groups:
        by_class.setdefault(cls, []).append(video)

    review = load_review_state()
    step = max(2, round(1 / val_fraction)) if val_fraction > 0 else 0
    moved = 0
    counts = {"train": 0, "val": 0}
    for cls, videos in by_class.items():
        for index, video in enumerate(sorted(set(videos)), start=1):
            target = "val" if (step and index % step == 0) else "train"
            for stem, current in groups[(cls, video)]:
                counts[target] += 1
                if current != target:
                    for ext in IMAGE_EXTENSIONS:
                        src = DATASET_IMAGES_DIR / current / f"{stem}{ext}"
                        if src.exists():
                            shutil.move(str(src), str(DATASET_IMAGES_DIR / target / f"{stem}{ext}"))
                            break
                    src_label = dataset_label_path(stem, current)
                    if src_label.exists():
                        shutil.move(str(src_label), str(dataset_label_path(stem, target)))
                    if stem in review and isinstance(review[stem], dict):
                        review[stem]["split"] = target
                    moved += 1
    save_review_state(review)
    return {"moved": moved, **counts}


def dataset_image_path(stem: str, split: str) -> Path:
    return DATASET_IMAGES_DIR / split / f"{stem}.jpg"


def dataset_label_path(stem: str, split: str) -> Path:
    return DATASET_LABELS_DIR / split / f"{stem}.txt"


def find_dataset_image(stem: str) -> tuple[Path | None, str | None]:
    """Locate a dataset image by stem, returning (path, split)."""
    for split in SPLITS:
        for ext in IMAGE_EXTENSIONS:
            candidate = DATASET_IMAGES_DIR / split / f"{stem}{ext}"
            if candidate.exists():
                return candidate, split
    return None, None


# ---------------------------------------------------------------------------
# dataset.yaml for the unified (standard) YOLO layout
# ---------------------------------------------------------------------------
def write_dataset_yaml(names_in_order: list[str] | None = None) -> None:
    """Write yolo_dataset/dataset.yaml for the merged images/labels layout."""
    if names_in_order is None:
        names_in_order = class_names()
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(names_in_order))
    text = (
        "# Tells YOLO where the pictures are and what the classes are.\n"
        "# 'path' is absolute so training works from any folder, and this file\n"
        "# is rewritten whenever the dataset changes so it stays correct.\n"
        f"path: {DATASET_DIR.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(names_in_order)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_YAML.write_text(text)


# ---------------------------------------------------------------------------
# Counting + duration helpers
# ---------------------------------------------------------------------------
def count_images(directory: Path) -> int:
    """Count image files directly (non-recursively is enough here) under a dir."""
    if not directory.exists():
        return 0
    return sum(
        1 for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_raw_background_images() -> int:
    """Count the background PICTURES the user has dropped in raw_videos/background/."""
    if not RAW_BACKGROUNDS_DIR.exists():
        return 0
    return sum(
        1 for p in RAW_BACKGROUNDS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.name.startswith(".")
    )


def count_background_images() -> int:
    """Count background/negative images across the train + val splits."""
    total = 0
    for split in SPLITS:
        directory = DATASET_IMAGES_DIR / split
        if directory.exists():
            total += sum(
                1 for p in directory.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                and is_background_stem(p.stem)
            )
    return total


def video_duration_seconds(video_path: Path) -> float:
    """Return a clip's duration in seconds, cached by path+size in meta/.

    Uses OpenCV frame count / fps. Returns 0.0 when it cannot be determined.
    """
    if not video_path.exists():
        return 0.0
    key = repo_relative(video_path)
    try:
        size = video_path.stat().st_size
    except OSError:
        size = 0
    cache = read_json(DURATIONS_FILE, {})
    if not isinstance(cache, dict):
        cache = {}
    entry = cache.get(key)
    if isinstance(entry, dict) and entry.get("size") == size:
        return float(entry.get("seconds", 0.0))

    seconds = 0.0
    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if capture.isOpened():
            fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            if fps > 0 and frames > 0:
                seconds = float(frames) / float(fps)
        capture.release()
    except Exception:  # noqa: BLE001 - duration is best-effort metadata
        seconds = 0.0

    cache[key] = {"size": size, "seconds": seconds}
    write_json(DURATIONS_FILE, cache)
    return seconds


def format_duration(seconds: float) -> str:
    """Return a short 'M:SS' or 'H:MM:SS' duration string."""
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_size(num_bytes: int) -> str:
    """Return a short human-readable file size like '12.4 MB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# ---------------------------------------------------------------------------
# Model versioning + registry (models/yolo_detector_vN.pt + active_model.pt)
# ---------------------------------------------------------------------------
_MODEL_VERSION_RE = re.compile(r"^yolo_detector_v(\d+)\.pt$")


def load_registry() -> dict:
    """Load the model registry (metrics + which version is active)."""
    data = read_json(MODEL_REGISTRY, {"models": {}, "active": None})
    if not isinstance(data, dict):
        data = {"models": {}, "active": None}
    data.setdefault("models", {})
    data.setdefault("active", None)
    return data


def save_registry(registry: dict) -> None:
    write_json(MODEL_REGISTRY, registry)


def model_version_files() -> list[Path]:
    """Return the versioned weight files, sorted by version number."""
    if not MODELS_DIR.exists():
        return []
    files = [p for p in MODELS_DIR.iterdir()
             if p.is_file() and _MODEL_VERSION_RE.match(p.name)]
    return sorted(files, key=lambda p: int(_MODEL_VERSION_RE.match(p.name).group(1)))


def next_model_version() -> tuple[int, Path]:
    """Return the next version number and its target path."""
    existing = [int(_MODEL_VERSION_RE.match(p.name).group(1)) for p in model_version_files()]
    version = max(existing, default=0) + 1
    return version, MODELS_DIR / f"yolo_detector_v{version}.pt"


def active_model_name() -> str | None:
    """Return the filename of the active model version, if known."""
    return load_registry().get("active")


def active_model_path() -> Path | None:
    """Return the path to the active model weights, if present."""
    if ACTIVE_MODEL.exists():
        return ACTIVE_MODEL
    name = active_model_name()
    if name and (MODELS_DIR / name).exists():
        return MODELS_DIR / name
    return None


def set_active_model(filename: str) -> Path:
    """Make one version the active model (copy to active_model.pt)."""
    import shutil

    filename = Path(filename).name
    source = MODELS_DIR / filename
    if not source.exists():
        raise ValueError(f"Model not found: {filename}")
    shutil.copyfile(str(source), str(ACTIVE_MODEL))
    registry = load_registry()
    registry["active"] = filename
    save_registry(registry)
    return ACTIVE_MODEL


def register_model(filename: str, info: dict) -> None:
    """Record metrics/metadata for a trained model version."""
    registry = load_registry()
    registry["models"][filename] = info
    save_registry(registry)
