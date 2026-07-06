#!/usr/bin/env python3
"""Local browser GUI for the rover training workflow.

Run with:

    python tools/workflow_gui.py

The app intentionally uses only the Python standard library. It serves one local
web page, stores uploaded clips in data/raw/clips/<object>/, runs the existing
shell scripts, and lets the user delete rejected review frames.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CLIPS_DIR = PROJECT_ROOT / "data" / "raw" / "clips"
RAW_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "photos"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
CLASSES_FILE = LABELS_DIR / "object_classes.txt"
PROCESSED_CLIPS_FILE = RAW_PHOTOS_DIR / ".processed_clips.json"
DELETED_FRAMES_FILE = RAW_PHOTOS_DIR / ".deleted_frames.json"
# Per-frame keep/skip decisions made in the review gallery. A frame marked
# "red" is moved out of the training folders (into <object>/excluded/) so YOLO
# never sees it, but its review image stays so the choice can be undone.
FRAME_MARKS_FILE = LABELS_DIR / ".frame_marks.json"
# The trained detector used by the in-browser webcam test tab.
DETECT_WEIGHTS = PROJECT_ROOT / "models" / "yolo_detector.pt"

VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

job_lock = threading.Lock()
current_job: dict | None = None
job_history: list[dict] = []

# The YOLO model is loaded lazily on the first webcam-test detection and then
# cached. Inference is guarded by a lock because the threaded HTTP server can
# receive overlapping detection requests.
detect_model = None
detect_model_lock = threading.Lock()


def repo_relative(path: Path) -> str:
    """Return a stable repo-relative path."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path, default):
    """Read a small JSON file."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def write_json(path: Path, data) -> None:
    """Write a small JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def validate_object_name(name: str) -> str:
    """Return a filesystem-safe object label."""
    name = name.strip().replace(" ", "_")
    if not name:
        raise ValueError("Choose or enter an object name.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Object names can use letters, numbers, _ and - only.")
    return name


def safe_filename(filename: str) -> str:
    """Return a safe filename, without any folder components."""
    filename = Path(filename).name.strip()
    filename = re.sub(r"[^A-Za-z0-9._ -]+", "_", filename)
    filename = filename.replace(" ", "_")
    if not filename or filename in {".", ".."}:
        raise ValueError("Uploaded file has no usable filename.")
    return filename


def unique_path(path: Path) -> Path:
    """Avoid overwriting an existing uploaded clip."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique filename for {path.name}")


def load_classes() -> list[tuple[int, str]]:
    """Read object_classes.txt as ordered (id, name) rows."""
    if not CLASSES_FILE.exists():
        return []
    rows: list[tuple[int, str]] = []
    for line in CLASSES_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((int(parts[0]), parts[1]))
        except ValueError:
            continue
    return sorted(rows)


def ensure_class(object_name: str) -> None:
    """Append a new object class if the upload introduced one."""
    object_name = validate_object_name(object_name)
    rows = load_classes()
    if any(name == object_name for _, name in rows):
        return
    next_id = (max((class_id for class_id, _ in rows), default=-1) + 1)
    CLASSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = CLASSES_FILE.read_text() if CLASSES_FILE.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    CLASSES_FILE.write_text(f"{existing}{next_id} {object_name}\n")


def list_objects() -> list[str]:
    """Return known object labels from classes and data folders."""
    objects = {name for _, name in load_classes()}
    for root in (RAW_CLIPS_DIR, RAW_PHOTOS_DIR, LABELS_DIR):
        if not root.exists():
            continue
        for path in root.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                objects.add(path.name)
    return sorted(objects)


def count_files(root: Path, extensions: set[str] | None = None) -> int:
    """Count files under a folder."""
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if extensions is not None and path.suffix.lower() not in extensions:
            continue
        total += 1
    return total


def clip_stem(video_path: Path) -> str:
    """Match data/extract_video_frames.py clip naming."""
    return video_path.stem.strip().replace(" ", "_").replace(".", "_")


def clip_key(object_name: str, video_path: Path) -> str:
    """Return the processed-clips manifest key."""
    return f"{object_name}/{video_path.name}"


def clip_has_frames(object_name: str, video_path: Path) -> bool:
    """Return True when a clip already has extracted raw frames."""
    object_dir = RAW_PHOTOS_DIR / object_name
    if not object_dir.exists():
        return False
    prefix = f"{clip_stem(video_path)}__frame_"
    return any(
        path.is_file()
        and path.name.startswith(prefix)
        and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in object_dir.iterdir()
    )


def load_processed_clips() -> dict[str, dict]:
    """Load the split-frame processed clip manifest."""
    data = read_json(PROCESSED_CLIPS_FILE, {"clips": {}})
    if isinstance(data, dict) and isinstance(data.get("clips"), dict):
        return data["clips"]
    return {}


def save_processed_clips(clips: dict[str, dict]) -> None:
    """Save the split-frame processed clip manifest."""
    write_json(PROCESSED_CLIPS_FILE, {"clips": clips})


def clip_rows() -> list[dict]:
    """Return clip status rows for the GUI."""
    rows: list[dict] = []
    processed = load_processed_clips()
    if not RAW_CLIPS_DIR.exists():
        return rows
    for object_dir in sorted(path for path in RAW_CLIPS_DIR.iterdir() if path.is_dir()):
        object_name = object_dir.name
        for video_path in sorted(path for path in object_dir.iterdir() if path.is_file()):
            if video_path.name.startswith("."):
                continue
            key = clip_key(object_name, video_path)
            has_frames = clip_has_frames(object_name, video_path)
            is_marked = key in processed
            if is_marked:
                status = "processed"
            elif has_frames:
                status = "frames"
            else:
                status = "new"
            rows.append(
                {
                    "object": object_name,
                    "video": video_path.name,
                    "status": status,
                    "frames": count_clip_frames(object_name, video_path),
                    "path": repo_relative(video_path),
                }
            )
    return rows


def count_clip_frames(object_name: str, video_path: Path) -> int:
    """Count raw frames belonging to one clip."""
    object_dir = RAW_PHOTOS_DIR / object_name
    if not object_dir.exists():
        return 0
    prefix = f"{clip_stem(video_path)}__frame_"
    return sum(
        1
        for path in object_dir.iterdir()
        if path.is_file()
        and path.name.startswith(prefix)
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_state() -> dict:
    """Build the dashboard state payload."""
    objects = list_objects()
    object_stats = []
    for object_name in objects:
        label_dir = LABELS_DIR / object_name
        object_stats.append(
            {
                "object": object_name,
                "clips": count_files(RAW_CLIPS_DIR / object_name, VIDEO_EXTENSIONS),
                "rawFrames": count_files(RAW_PHOTOS_DIR / object_name, IMAGE_EXTENSIONS),
                "review": count_files(label_dir / "review", IMAGE_EXTENSIONS),
                "train": count_files(label_dir / "images" / "train", IMAGE_EXTENSIONS),
                "val": count_files(label_dir / "images" / "val", IMAGE_EXTENSIONS),
            }
        )
    return {
        "classes": [name for _, name in load_classes()],
        "objects": objects,
        "stats": object_stats,
        "clips": clip_rows(),
        "job": public_job(current_job),
        "detector": {"available": DETECT_WEIGHTS.exists()},
    }


def public_job(job: dict | None) -> dict | None:
    """Return a JSON-safe copy of a job."""
    if job is None:
        return None
    return {
        "id": job["id"],
        "kind": job["kind"],
        "status": job["status"],
        "startedAt": job["started_at"],
        "finishedAt": job.get("finished_at"),
        "returnCode": job.get("return_code"),
        "log": job["log"][-250:],
    }


def parse_content_disposition(value: str) -> dict[str, str]:
    """Parse a small Content-Disposition header."""
    result: dict[str, str] = {}
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if "=" not in chunk:
            continue
        key, raw_value = chunk.split("=", 1)
        result[key.strip().lower()] = raw_value.strip().strip('"')
    return result


def parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], list[dict]]:
    """Parse the upload form sent by the browser."""
    match = re.search(r"boundary=(.+)", content_type)
    if not match:
        raise ValueError("Upload request is missing a multipart boundary.")
    boundary = match.group(1).strip().strip('"').encode()
    fields: dict[str, str] = {}
    files: list[dict] = []
    delimiter = b"--" + boundary

    for part in body.split(delimiter):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2]
        if b"\r\n\r\n" not in part:
            continue
        header_blob, data = part.split(b"\r\n\r\n", 1)
        headers: dict[str, str] = {}
        for raw_line in header_blob.decode("utf-8", "replace").split("\r\n"):
            if ":" not in raw_line:
                continue
            key, value = raw_line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        disposition = parse_content_disposition(headers.get("content-disposition", ""))
        name = disposition.get("name")
        filename = disposition.get("filename")
        if not name:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        if filename:
            files.append({"field": name, "filename": filename, "content": data})
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


def save_uploaded_videos(object_name: str, files: list[dict]) -> list[dict]:
    """Save uploaded video files into data/raw/clips/<object>/."""
    object_name = validate_object_name(object_name)
    ensure_class(object_name)
    target_dir = RAW_CLIPS_DIR / object_name
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict] = []

    for item in files:
        original_name = item["filename"]
        filename = safe_filename(original_name)
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise ValueError(f"{original_name} is not a supported video file.")
        target = unique_path(target_dir / filename)
        target.write_bytes(item["content"])
        saved.append(
            {
                "name": target.name,
                "object": object_name,
                "path": repo_relative(target),
                "size": target.stat().st_size,
            }
        )
    return saved


def run_command_job(job: dict, command: list[str], env: dict[str, str] | None = None) -> None:
    """Run one workflow command and collect its output."""
    try:
        job["log"].append(f"$ {' '.join(command)}")
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            job["log"].append(line.rstrip())
        return_code = process.wait()
        job["return_code"] = return_code
        job["status"] = "succeeded" if return_code == 0 else "failed"
        job["log"].append(f"Finished with exit code {return_code}.")
    except Exception as error:  # noqa: BLE001 - surface unexpected job failures
        job["status"] = "failed"
        job["return_code"] = 1
        job["log"].append(f"Error: {error}")
    finally:
        job["finished_at"] = time.time()


def start_job(kind: str, command: list[str], env: dict[str, str] | None = None) -> dict:
    """Start a background job, allowing only one workflow job at a time."""
    global current_job
    with job_lock:
        if current_job is not None and current_job["status"] == "running":
            raise RuntimeError(f"{current_job['kind']} is already running.")
        job = {
            "id": uuid.uuid4().hex[:10],
            "kind": kind,
            "status": "running",
            "started_at": time.time(),
            "log": [],
        }
        current_job = job
        job_history.insert(0, job)
        del job_history[8:]
    thread = threading.Thread(
        target=run_command_job,
        args=(job, command, env),
        daemon=True,
    )
    thread.start()
    return job


def read_body(handler: BaseHTTPRequestHandler) -> bytes:
    """Read the request body."""
    length = int(handler.headers.get("Content-Length", "0"))
    return handler.rfile.read(length)


def json_response(handler: BaseHTTPRequestHandler, payload, status: int = 200) -> None:
    """Send a JSON response."""
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200) -> None:
    """Send a text response."""
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def not_found(handler: BaseHTTPRequestHandler) -> None:
    """Send a 404."""
    json_response(handler, {"error": "Not found"}, 404)


def review_stem_for_raw_frame(object_name: str, raw_path: Path) -> str:
    """Return the review filename stem that auto_label_frames.py creates."""
    object_dir = RAW_PHOTOS_DIR / object_name
    if raw_path.parent == object_dir:
        stem = f"{object_name}__{raw_path.stem}"
    else:
        stem = f"{object_name}__{raw_path.parent.name}__{raw_path.stem}"
    return stem.replace(".", "_")


def find_raw_frames_for_review(object_name: str, review_stem: str) -> list[Path]:
    """Find the raw frame or frames that created one review image."""
    object_dir = RAW_PHOTOS_DIR / object_name
    if not object_dir.exists():
        return []
    matches = [
        path
        for path in object_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and review_stem_for_raw_frame(object_name, path) == review_stem
    ]
    if matches:
        return matches

    prefix = f"{object_name}__"
    if not review_stem.startswith(prefix):
        return []
    raw_stem = review_stem[len(prefix):]
    guessed = [
        object_dir / f"{raw_stem}{extension}"
        for extension in (".jpg", ".jpeg", ".png")
    ]
    return [path for path in guessed if path.exists()]


def record_deleted_frames(paths: list[Path]) -> None:
    """Persist raw frames rejected by the review UI."""
    data = read_json(DELETED_FRAMES_FILE, {"deleted_frames": []})
    entries = data.get("deleted_frames", []) if isinstance(data, dict) else []
    deleted = {str(entry) for entry in entries}
    for path in paths:
        deleted.add(repo_relative(path))
    write_json(DELETED_FRAMES_FILE, {"deleted_frames": sorted(deleted)})


def mark_source_clips_handled(object_name: str, raw_frames: list[Path]) -> None:
    """Make sure rejected frames do not cause an old clip to be split again."""
    if not raw_frames:
        return
    processed = load_processed_clips()
    clips_dir = RAW_CLIPS_DIR / object_name
    if not clips_dir.exists():
        return
    videos = [path for path in clips_dir.iterdir() if path.is_file()]
    changed = False

    for raw_frame in raw_frames:
        if "__frame_" not in raw_frame.stem:
            continue
        source_stem = raw_frame.stem.split("__frame_", 1)[0]
        source_video = next(
            (video for video in videos if clip_stem(video) == source_stem),
            None,
        )
        if source_video is None:
            continue
        key = clip_key(object_name, source_video)
        if key in processed:
            continue
        processed[key] = {
            "clip_stem": clip_stem(source_video),
            "object": object_name,
            "path": repo_relative(source_video),
            "saved_images": count_clip_frames(object_name, source_video),
            "status": "reviewed",
            "video": source_video.name,
            "frame_step": None,
        }
        changed = True

    if changed:
        save_processed_clips(processed)


def delete_if_exists(path: Path, removed: list[str]) -> None:
    """Delete one file and remember it for the response."""
    if path.exists() and path.is_file():
        path.unlink()
        removed.append(repo_relative(path))


def delete_review_frame(object_name: str, filename: str) -> dict:
    """Delete a rejected frame from review, labels, images, and raw photos."""
    object_name = validate_object_name(object_name)
    filename = Path(filename).name
    if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Review file must be an image.")

    review_path = LABELS_DIR / object_name / "review" / filename
    review_stem = review_path.stem
    removed: list[str] = []
    raw_frames = find_raw_frames_for_review(object_name, review_stem)

    delete_if_exists(review_path, removed)
    for split in ("train", "val"):
        images_dir = LABELS_DIR / object_name / "images" / split
        labels_dir = LABELS_DIR / object_name / "labels" / split
        for extension in IMAGE_EXTENSIONS:
            delete_if_exists(images_dir / f"{review_stem}{extension}", removed)
        delete_if_exists(labels_dir / f"{review_stem}.txt", removed)

    for raw_frame in raw_frames:
        delete_if_exists(raw_frame, removed)

    if raw_frames:
        record_deleted_frames(raw_frames)
        mark_source_clips_handled(object_name, raw_frames)

    return {
        "deleted": removed,
        "rawFrames": [repo_relative(path) for path in raw_frames],
    }


def load_frame_marks() -> dict[str, dict]:
    """Load per-frame keep/skip decisions made in the review gallery."""
    data = read_json(FRAME_MARKS_FILE, {"marks": {}})
    if isinstance(data, dict) and isinstance(data.get("marks"), dict):
        return data["marks"]
    return {}


def save_frame_marks(marks: dict[str, dict]) -> None:
    """Persist per-frame keep/skip decisions."""
    write_json(FRAME_MARKS_FILE, {"marks": marks})


def mark_key(object_name: str, stem: str) -> str:
    """Return the frame-marks manifest key for one review frame."""
    return f"{object_name}/{stem}"


def frame_state(marks: dict[str, dict], object_name: str, stem: str) -> str:
    """Return 'pass', 'fail', or 'unmarked' for one review frame.

    A missing entry means the frame has not been reviewed yet ('unmarked').
    Both 'unmarked' and 'pass' frames stay in training; only 'fail' frames are
    moved out.
    """
    entry = marks.get(mark_key(object_name, stem))
    if isinstance(entry, dict) and entry.get("state") in {"pass", "fail"}:
        return entry["state"]
    return "unmarked"


def excluded_dirs(object_name: str) -> tuple[Path, Path]:
    """Return the holding folders for frames left out of training."""
    root = LABELS_DIR / object_name / "excluded"
    return root / "images", root / "labels"


def find_active_image(object_name: str, stem: str) -> tuple[Path | None, str | None, str | None]:
    """Find a frame's image inside the live train/val folders."""
    for split in ("train", "val"):
        images_dir = LABELS_DIR / object_name / "images" / split
        for extension in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{stem}{extension}"
            if candidate.exists():
                return candidate, split, extension
    return None, None, None


def exclude_frame(object_name: str, stem: str) -> dict:
    """Move a frame's image and label out of training into the holding area."""
    image_path, split, extension = find_active_image(object_name, stem)
    excluded_images, excluded_labels = excluded_dirs(object_name)
    if image_path is None:
        return {"split": None, "ext": None}
    excluded_images.mkdir(parents=True, exist_ok=True)
    excluded_labels.mkdir(parents=True, exist_ok=True)
    shutil.move(str(image_path), str(excluded_images / image_path.name))
    label_path = LABELS_DIR / object_name / "labels" / split / f"{stem}.txt"
    if label_path.exists():
        shutil.move(str(label_path), str(excluded_labels / label_path.name))
    return {"split": split, "ext": extension}


def restore_frame(object_name: str, stem: str, entry: dict | None) -> None:
    """Move a previously excluded frame back into training."""
    excluded_images, excluded_labels = excluded_dirs(object_name)
    split = (entry or {}).get("split") or "train"
    extension = (entry or {}).get("ext") or ".jpg"
    source_image = excluded_images / f"{stem}{extension}"
    if source_image.exists():
        dest_dir = LABELS_DIR / object_name / "images" / split
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_image), str(dest_dir / source_image.name))
    source_label = excluded_labels / f"{stem}.txt"
    if source_label.exists():
        dest_dir = LABELS_DIR / object_name / "labels" / split
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_label), str(dest_dir / source_label.name))


def set_frame_mark(object_name: str, filename: str, state: str) -> dict:
    """Mark one review frame 'pass', 'fail', or 'unmarked'.

    'fail' moves the image + label into the holding area so YOLO never trains on
    it. 'pass' and 'unmarked' keep it in training; the only difference is whether
    the user has reviewed it yet.
    """
    object_name = validate_object_name(object_name)
    if state not in {"pass", "fail", "unmarked"}:
        raise ValueError("Mark must be pass, fail, or unmarked.")
    stem = Path(filename).stem
    marks = load_frame_marks()
    key = mark_key(object_name, stem)
    entry = marks.get(key) if isinstance(marks.get(key), dict) else None
    current = frame_state(marks, object_name, stem)

    if state == "fail":
        if current != "fail":
            moved = exclude_frame(object_name, stem)
            marks[key] = {"state": "fail", "split": moved["split"], "ext": moved["ext"]}
    else:
        if current == "fail":
            restore_frame(object_name, stem, entry)
        if state == "pass":
            marks[key] = {"state": "pass"}
        else:  # unmarked: forget the frame entirely
            marks.pop(key, None)

    save_frame_marks(marks)
    return {"object": object_name, "stem": stem, "state": state}


def pass_all_unmarked(object_filter: str = "") -> dict:
    """Move every currently unmarked frame to 'pass'. Leaves 'fail' frames alone."""
    marks = load_frame_marks()
    if object_filter and object_filter not in {"", "__all__", "all"}:
        targets = [validate_object_name(object_filter)]
    else:
        targets = list_objects()

    passed = 0
    for object_name in targets:
        review_dir = LABELS_DIR / object_name / "review"
        if not review_dir.exists():
            continue
        for path in sorted(review_dir.iterdir()):
            if not (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
                continue
            if frame_state(marks, object_name, path.stem) == "unmarked":
                marks[mark_key(object_name, path.stem)] = {"state": "pass"}
                passed += 1

    save_frame_marks(marks)
    return {"passed": passed}


def list_review_items(object_filter: str = "") -> dict:
    """List review frames (all objects or one) with their pass/fail state."""
    marks = load_frame_marks()
    objects = list_objects()
    if object_filter and object_filter not in {"", "__all__", "all"}:
        targets = [validate_object_name(object_filter)]
        scope = targets[0]
    else:
        targets = objects
        scope = "__all__"

    items: list[dict] = []
    counts = {"unmarked": 0, "pass": 0, "fail": 0}
    for object_name in targets:
        review_dir = LABELS_DIR / object_name / "review"
        if not review_dir.exists():
            continue
        for path in sorted(review_dir.iterdir()):
            if not (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
                continue
            state = frame_state(marks, object_name, path.stem)
            counts[state] += 1
            items.append(
                {
                    "object": object_name,
                    "name": path.name,
                    "stem": path.stem,
                    "state": state,
                    "url": (
                        "/review/"
                        + urllib.parse.quote(object_name)
                        + "/"
                        + urllib.parse.quote(path.name)
                    ),
                }
            )
    return {"object": scope, "objects": objects, "items": items, "counts": counts}


def serve_review_image(handler: BaseHTTPRequestHandler, object_name: str, filename: str) -> None:
    """Serve one review image."""
    object_name = validate_object_name(urllib.parse.unquote(object_name))
    filename = Path(urllib.parse.unquote(filename)).name
    image_path = LABELS_DIR / object_name / "review" / filename
    if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        not_found(handler)
        return
    body = image_path.read_bytes()
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def format_size(num_bytes: int) -> str:
    """Return a short human-readable file size like '12.4 MB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def build_raw_tree() -> dict:
    """Describe data/raw so the GUI can show which videos have been added."""
    def video_nodes(object_dir: Path) -> list[dict]:
        nodes: list[dict] = []
        for path in sorted(object_dir.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            nodes.append(
                {
                    "name": path.name,
                    "type": "video",
                    "size": format_size(path.stat().st_size),
                    "url": (
                        "/media/clips/"
                        + urllib.parse.quote(object_dir.name)
                        + "/"
                        + urllib.parse.quote(path.name)
                    ),
                }
            )
        return nodes

    clips: list[dict] = []
    if RAW_CLIPS_DIR.exists():
        for object_dir in sorted(p for p in RAW_CLIPS_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")):
            children = video_nodes(object_dir)
            clips.append(
                {"name": object_dir.name, "type": "dir", "count": len(children), "children": children}
            )

    photos: list[dict] = []
    if RAW_PHOTOS_DIR.exists():
        for object_dir in sorted(p for p in RAW_PHOTOS_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")):
            photos.append(
                {
                    "name": object_dir.name,
                    "type": "dir",
                    "count": count_files(object_dir, IMAGE_EXTENSIONS),
                    "children": [],
                }
            )

    return {
        "children": [
            {"name": "clips", "type": "dir", "children": clips},
            {"name": "photos", "type": "dir", "children": photos},
        ]
    }


def serve_media_file(handler: BaseHTTPRequestHandler, object_name: str, filename: str) -> None:
    """Stream a clip video, honoring HTTP Range requests so it can be scrubbed."""
    object_name = validate_object_name(urllib.parse.unquote(object_name))
    filename = Path(urllib.parse.unquote(filename)).name
    video_path = RAW_CLIPS_DIR / object_name / filename
    if not video_path.exists() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        not_found(handler)
        return

    file_size = video_path.stat().st_size
    content_type = mimetypes.guess_type(video_path.name)[0] or "application/octet-stream"
    range_header = handler.headers.get("Range")
    start, end = 0, file_size - 1
    status = 200

    if range_header:
        match = re.match(r"bytes=(\d+)-(\d*)", range_header.strip())
        if match:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else file_size - 1
            end = min(end, file_size - 1)
            if start > end or start >= file_size:
                handler.send_response(416)
                handler.send_header("Content-Range", f"bytes */{file_size}")
                handler.end_headers()
                return
            status = 206

    length = end - start + 1
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Accept-Ranges", "bytes")
    if status == 206:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
    handler.send_header("Content-Length", str(length))
    handler.end_headers()

    try:
        with video_path.open("rb") as source:
            source.seek(start)
            remaining = length
            while remaining > 0:
                chunk = source.read(min(65536, remaining))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                remaining -= len(chunk)
    except (BrokenPipeError, ConnectionResetError):
        # The browser closed the connection (common when scrubbing video).
        pass


def get_detect_model():
    """Load and cache the trained YOLO detector for the webcam test tab."""
    global detect_model
    if detect_model is not None:
        return detect_model
    if not DETECT_WEIGHTS.exists():
        raise RuntimeError(
            "No trained model yet. Run Process Data, then Train Data, then reload."
        )
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "The 'ultralytics' package is not installed in this environment."
        ) from error
    detect_model = YOLO(str(DETECT_WEIGHTS))
    return detect_model


def run_detection(image_bytes: bytes, conf: float) -> dict:
    """Run the detector on one JPEG frame and return boxes as 0-1 fractions."""
    import cv2
    import numpy as np

    model = get_detect_model()
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode the webcam frame.")
    height, width = frame.shape[:2]

    with detect_model_lock:
        results = model.predict(frame, conf=conf, verbose=False)

    result = results[0]
    names = result.names
    detections: list[dict] = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        detections.append(
            {
                "x": x1 / width,
                "y": y1 / height,
                "w": (x2 - x1) / width,
                "h": (y2 - y1) / height,
                "name": names[int(box.cls[0])],
                "conf": float(box.conf[0]),
            }
        )
    return {"detections": detections, "width": width, "height": height}


class WorkflowHandler(BaseHTTPRequestHandler):
    """HTTP handler for the local workflow app."""

    server_version = "RoverWorkflowGUI/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        """Keep the terminal quieter than the default HTTP handler."""
        print(f"[workflow-gui] {self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if path == "/":
                text_response(self, HTML)
            elif path == "/api/state":
                json_response(self, build_state())
            elif path == "/api/job":
                json_response(
                    self,
                    {
                        "current": public_job(current_job),
                        "history": [public_job(job) for job in job_history],
                    },
                )
            elif path == "/api/review":
                object_name = query.get("object", [""])[0]
                json_response(self, list_review_items(object_name))
            elif path == "/api/raw-tree":
                json_response(self, build_raw_tree())
            elif path.startswith("/review/"):
                parts = path.split("/", 3)
                if len(parts) == 4:
                    serve_review_image(self, parts[2], parts[3])
                else:
                    not_found(self)
            elif path.startswith("/media/clips/"):
                parts = path.split("/", 4)
                if len(parts) == 5:
                    serve_media_file(self, parts[3], parts[4])
                else:
                    not_found(self)
            else:
                not_found(self)
        except Exception as error:  # noqa: BLE001
            json_response(self, {"error": str(error)}, 400)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            if path == "/api/upload":
                content_type = self.headers.get("Content-Type", "")
                fields, files = parse_multipart(read_body(self), content_type)
                object_name = fields.get("object") or fields.get("newObject") or ""
                saved = save_uploaded_videos(object_name, files)
                json_response(self, {"saved": saved, "state": build_state()})
            elif path == "/api/jobs/process":
                job = start_job("process_data", ["./scripts/process.sh"])
                json_response(self, {"job": public_job(job)})
            elif path == "/api/jobs/train":
                body = json.loads(read_body(self) or b"{}")
                env: dict[str, str] = {}
                command = ["./scripts/train.sh"]
                epochs = str(body.get("epochs", "")).strip()
                device = str(body.get("device", "")).strip()
                if epochs:
                    command.extend(["--epochs", epochs])
                if device:
                    env["DEVICE"] = device
                job = start_job("train_model", command, env)
                json_response(self, {"job": public_job(job)})
            elif path == "/api/review/mark":
                body = json.loads(read_body(self) or b"{}")
                result = set_frame_mark(
                    str(body.get("object", "")),
                    str(body.get("filename", "")),
                    str(body.get("state", "")),
                )
                json_response(self, {"result": result, "state": build_state()})
            elif path == "/api/review/pass-all":
                body = json.loads(read_body(self) or b"{}")
                result = pass_all_unmarked(str(body.get("object", "")))
                json_response(self, {"result": result, "state": build_state()})
            elif path == "/api/detect":
                query = urllib.parse.parse_qs(parsed.query)
                try:
                    conf = float(query.get("conf", ["0.25"])[0])
                except ValueError:
                    conf = 0.25
                conf = min(max(conf, 0.01), 0.99)
                result = run_detection(read_body(self), conf)
                json_response(self, result)
            elif path == "/api/review/delete":
                body = json.loads(read_body(self) or b"{}")
                result = delete_review_frame(
                    str(body.get("object", "")),
                    str(body.get("filename", "")),
                )
                json_response(self, {"result": result, "state": build_state()})
            else:
                not_found(self)
        except Exception as error:  # noqa: BLE001
            json_response(self, {"error": str(error)}, 400)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rover Training Workflow</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #64748b;
      --line: #d8dee6;
      --accent: #0f766e;
      --accent-2: #2563eb;
      --danger: #b42318;
      --soft: #eef7f5;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 70px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(12px);
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.1;
      font-weight: 750;
    }
    .subtle {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      max-width: 1500px;
      margin: 0 auto;
      align-items: start;
    }
    main > .left, main > .review { min-width: 0; }
    section, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .left {
      display: grid;
      gap: 14px;
      align-content: start;
    }
    .panel {
      padding: 16px;
    }
    .panel h2 {
      margin: 0 0 14px;
      font-size: 16px;
      line-height: 1.25;
    }
    label {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    select, input[type="text"], input[type="number"] {
      width: 100%;
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 10px;
      background: #fff;
      color: var(--ink);
      font-size: 14px;
    }
    .object-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 12px;
    }
    .dropzone {
      min-height: 136px;
      display: grid;
      place-items: center;
      gap: 8px;
      padding: 18px;
      border: 1.5px dashed #9fb3c8;
      border-radius: 8px;
      background: #f8fafc;
      text-align: center;
      cursor: pointer;
      transition: border-color 140ms ease, background 140ms ease;
    }
    .dropzone.drag {
      background: var(--soft);
      border-color: var(--accent);
    }
    .dropzone strong {
      display: block;
      font-size: 15px;
    }
    .dropzone span {
      color: var(--muted);
      font-size: 13px;
    }
    .button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    button {
      min-height: 38px;
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 0 13px;
      font-size: 14px;
      font-weight: 750;
      color: #fff;
      background: var(--accent);
      cursor: pointer;
    }
    button.secondary { background: var(--accent-2); }
    button.neutral {
      color: var(--ink);
      background: #fff;
      border-color: var(--line);
    }
    button.danger { background: var(--danger); }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .train-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .stat {
      min-height: 72px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfe;
    }
    .stat b {
      display: block;
      font-size: 22px;
      line-height: 1.15;
    }
    .stat span {
      color: var(--muted);
      font-size: 12px;
    }
    .clips {
      max-height: 230px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid #edf1f5;
      text-align: left;
      vertical-align: top;
    }
    th {
      background: #f8fafc;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 0 8px;
      border-radius: 999px;
      background: #e2e8f0;
      color: #334155;
      font-size: 12px;
      font-weight: 750;
    }
    .pill.new { background: #dbeafe; color: #1d4ed8; }
    .pill.frames { background: #fef3c7; color: #92400e; }
    .pill.processed { background: #ccfbf1; color: #0f766e; }
    .job-strip {
      max-width: 1500px;
      margin: 0 auto;
      padding: 0 18px 18px;
    }
    .job-strip .panel { padding: 16px; }
    .job-strip .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }
    .job-strip .panel-head h2 { margin: 0; }
    .job-log {
      height: 300px;
      min-height: 120px;
      resize: vertical;
      overflow: auto;
      margin: 0;
      padding: 12px;
      border-radius: 8px;
      background: #111827;
      color: #d1d5db;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    /* data/raw file tree */
    .tree {
      max-height: 260px;
      overflow: auto;
      font-size: 13px;
    }
    .tree-group { margin-bottom: 8px; }
    .tree-label {
      display: flex;
      align-items: center;
      gap: 6px;
      font-weight: 700;
      color: var(--ink);
      padding: 4px 0;
    }
    .tree-object { margin: 2px 0 2px 12px; }
    .tree-object > .tree-label { color: var(--muted); font-weight: 700; text-transform: none; }
    .tree-video {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-left: 26px;
      padding: 5px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
      cursor: pointer;
      margin-bottom: 4px;
    }
    .tree-video:hover { border-color: var(--accent); background: var(--soft); }
    .tree-video .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .tree-video .size { color: var(--muted); font-size: 11px; flex: none; }
    .tree-empty { margin-left: 26px; color: var(--muted); font-size: 12px; }
    /* video preview modal */
    .modal {
      position: fixed;
      inset: 0;
      z-index: 50;
      display: grid;
      place-items: center;
      padding: 24px;
      background: rgba(15, 23, 42, 0.55);
    }
    .modal[hidden] { display: none; }
    .modal-card {
      width: min(880px, 100%);
      background: var(--panel);
      border-radius: 10px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .modal-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
    }
    .modal-head .title { font-weight: 750; overflow-wrap: anywhere; }
    .modal-card video {
      display: block;
      width: 100%;
      max-height: 70vh;
      background: #000;
    }
    .modal-foot { padding: 10px 16px; }
    .review {
      min-height: calc(100vh - 106px);
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .review-head {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 12px;
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }
    .review-controls {
      display: grid;
      grid-template-columns: minmax(180px, 260px) auto;
      gap: 10px;
      align-items: end;
    }
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(132px, 1fr));
      gap: 8px;
      padding: 12px;
      align-content: start;
    }
    .frame {
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }
    .frame img {
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      background: #e5e7eb;
    }
    .frame-body {
      display: grid;
      gap: 6px;
      padding: 7px;
    }
    .frame-name {
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .frame.state-pass { border-color: #16a34a; box-shadow: 0 0 0 2px rgba(22, 163, 74, 0.3); }
    .frame.state-fail { border-color: #dc2626; box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.25); }
    .frame.state-fail img { opacity: 0.5; }
    .mark-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    button.mark {
      min-height: 30px;
      padding: 0 6px;
      font-size: 12px;
      color: #334155;
      background: #fff;
      border: 1.5px solid var(--line);
    }
    button.mark.pass.active { color: #fff; background: #16a34a; border-color: #16a34a; }
    button.mark.fail.active { color: #fff; background: var(--danger); border-color: var(--danger); }
    .tabs {
      display: flex;
      gap: 8px;
      padding: 12px 16px 0;
      flex-wrap: wrap;
    }
    button.tab {
      color: var(--muted);
      background: #fff;
      border: 1px solid var(--line);
      font-weight: 700;
    }
    button.tab.active { color: var(--ink); border-color: var(--accent); background: var(--soft); }
    .tab .count {
      display: inline-flex;
      align-items: center;
      min-width: 20px;
      height: 18px;
      margin-left: 6px;
      padding: 0 6px;
      border-radius: 999px;
      background: #e2e8f0;
      color: #334155;
      font-size: 11px;
    }
    button.tab.active .count { background: #fff; }
    .empty {
      padding: 32px;
      color: var(--muted);
      text-align: center;
    }
    .status-line {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: #94a3b8;
    }
    .dot.running { background: #f59e0b; }
    .dot.succeeded { background: #059669; }
    .dot.failed { background: #dc2626; }
    /* top-level view tabs */
    .topnav {
      display: flex;
      gap: 8px;
      padding: 12px 18px 0;
      max-width: 1500px;
      margin: 0 auto;
    }
    button.topbtn {
      color: var(--muted);
      background: #fff;
      border: 1px solid var(--line);
      min-height: 40px;
      padding: 0 18px;
      font-size: 15px;
    }
    button.topbtn.active { color: #fff; background: var(--accent); border-color: var(--accent); }
    [hidden] { display: none !important; }
    /* webcam detector view */
    .detector {
      max-width: 1100px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      gap: 16px;
    }
    .detector .panel { padding: 18px; }
    .cam-controls {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 12px;
      margin-top: 12px;
    }
    .cam-controls .conf {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .cam-stage {
      position: relative;
      width: 100%;
      aspect-ratio: 4 / 3;
      background: #0b1220;
      border-radius: 10px;
      overflow: hidden;
    }
    .cam-stage video, .cam-stage canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }
    .cam-stage video { object-fit: contain; }
    .cam-hint {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      color: #94a3b8;
      font-size: 14px;
      text-align: center;
      padding: 24px;
    }
    .cam-status {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .review { min-height: auto; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Rover Training Workflow</h1>
      <p class="subtle">Captured clips to frames, labels, review, and YOLO training.</p>
    </div>
    <div class="status-line">
      <span id="jobDot" class="dot"></span>
      <span id="jobStatus">Idle</span>
    </div>
  </header>

  <nav class="topnav">
    <button class="topbtn active" data-view="workflow">Workflow</button>
    <button class="topbtn" data-view="detector">Test Detector (Webcam)</button>
  </nav>

  <div id="viewWorkflow">
  <main>
    <div class="left">
      <section class="panel">
        <h2>Upload Clips</h2>
        <div class="object-grid">
          <div>
            <label for="objectSelect">Object</label>
            <select id="objectSelect"></select>
          </div>
          <div>
            <label for="newObject">New object</label>
            <input id="newObject" type="text" placeholder="optional">
          </div>
        </div>
        <div id="dropzone" class="dropzone" tabindex="0">
          <div>
            <strong>Drop videos here</strong>
            <span>or click to choose multiple files</span>
          </div>
        </div>
        <input id="fileInput" type="file" accept="video/*,.MOV,.mov,.mp4,.m4v,.avi,.mkv,.webm" multiple hidden>
        <div class="button-row">
          <button id="uploadBtn">Upload Videos</button>
          <button id="clearFilesBtn" class="neutral">Clear Selection</button>
        </div>
        <p id="fileSummary" class="subtle">No files selected.</p>
      </section>

      <section class="panel">
        <div class="button-row" style="justify-content: space-between; margin-top: 0;">
          <h2 style="margin: 0;">data/raw</h2>
          <button id="refreshTreeBtn" class="neutral">Refresh</button>
        </div>
        <p class="subtle">Videos already added. Click a video to preview it.</p>
        <div id="rawTree" class="tree"></div>
      </section>

      <section class="panel">
        <h2>Pipeline</h2>
        <div class="button-row">
          <button id="processBtn">Process Data</button>
          <button id="trainBtn" class="secondary">Train Data</button>
        </div>
        <div class="train-grid">
          <div>
            <label for="epochsInput">Epochs</label>
            <input id="epochsInput" type="number" min="1" placeholder="default">
          </div>
          <div>
            <label for="deviceInput">Device</label>
            <input id="deviceInput" type="text" placeholder="mps or cpu">
          </div>
        </div>
      </section>

      <section class="panel">
        <h2>Dataset Counts</h2>
        <div id="stats" class="stats"></div>
      </section>

      <section class="panel">
        <h2>Clips</h2>
        <div class="clips">
          <table>
            <thead>
              <tr><th>Object</th><th>Video</th><th>Status</th><th>Frames</th></tr>
            </thead>
            <tbody id="clipRows"></tbody>
          </table>
        </div>
      </section>
    </div>

    <section class="review">
      <div class="review-head">
        <div>
          <h2>Training Frames</h2>
          <p id="reviewCount" class="subtle">New frames start as <b>Unmarked</b>. Pass keeps a frame for training; Fail drops it.</p>
        </div>
        <div class="review-controls">
          <div>
            <label for="reviewObject">Object</label>
            <select id="reviewObject"></select>
          </div>
          <button id="refreshReviewBtn" class="neutral">Refresh</button>
        </div>
      </div>
      <div id="reviewTabs" class="tabs">
        <button class="tab active" data-tab="unmarked">Unmarked <span id="countUnmarked" class="count">0</span></button>
        <button class="tab" data-tab="pass">Pass <span id="countPass" class="count">0</span></button>
        <button class="tab" data-tab="fail">Fail <span id="countFail" class="count">0</span></button>
        <button id="passAllBtn" class="secondary" style="margin-left: auto;">Pass All Unmarked</button>
      </div>
      <div id="gallery" class="gallery"></div>
    </section>
  </main>

  <div class="job-strip">
    <section class="panel">
      <div class="panel-head">
        <h2>Job Log</h2>
        <button id="clearLogViewBtn" class="neutral">Jump to Latest</button>
      </div>
      <pre id="jobLog" class="job-log">No job has run yet.</pre>
    </section>
  </div>
  </div>

  <div id="viewDetector" hidden>
    <div class="detector">
      <section class="panel">
        <h2>Test the Detector with Your Webcam</h2>
        <p class="subtle">Uses the latest trained model (<code>models/yolo_detector.pt</code>). Hold a bit or wrench up to the camera and it draws a live box around what it recognizes.</p>
        <div class="cam-controls">
          <button id="camStartBtn">Start Camera</button>
          <button id="camStopBtn" class="neutral" disabled>Stop Camera</button>
          <div class="conf">
            <label for="camConf" style="margin: 0;">Confidence</label>
            <input id="camConf" type="range" min="0.05" max="0.9" step="0.05" value="0.25" style="width: 140px;">
            <span id="camConfValue">0.25</span>
          </div>
        </div>
        <div class="cam-status">
          <span id="camDot" class="dot"></span>
          <span id="camStatus">Camera is off.</span>
        </div>
        <div class="cam-stage" style="margin-top: 14px;">
          <video id="camVideo" autoplay playsinline muted></video>
          <canvas id="camOverlay"></canvas>
          <div id="camHint" class="cam-hint">Click <b>Start Camera</b> to begin. Your browser will ask for camera permission.</div>
        </div>
      </section>
    </div>
  </div>

  <div id="videoModal" class="modal" hidden>
    <div class="modal-card">
      <div class="modal-head">
        <span id="videoTitle" class="title"></span>
        <button id="videoClose" class="neutral">Close</button>
      </div>
      <video id="videoPlayer" controls preload="metadata"></video>
      <div class="modal-foot">
        <a id="videoOpen" class="subtle" href="#" target="_blank" rel="noopener">Open in a new tab</a>
      </div>
    </div>
  </div>

  <script>
    const state = {
      files: [],
      lastJobStatus: null,
      reviewTab: "unmarked",
      reviewItems: [],
      reviewCounts: { unmarked: 0, pass: 0, fail: 0 },
      detectorAvailable: false,
      camStream: null,
      camTimer: null,
      camBusy: false,
    };

    const $ = (id) => document.getElementById(id);

    function activeObject() {
      return $("newObject").value.trim() || $("objectSelect").value;
    }

    function setFiles(files) {
      state.files = Array.from(files || []);
      $("fileSummary").textContent = state.files.length
        ? state.files.map((file) => file.name).join(", ")
        : "No files selected.";
    }

    function setBusy(isBusy) {
      for (const id of ["processBtn", "trainBtn", "uploadBtn"]) {
        $(id).disabled = isBusy;
      }
    }

    async function api(path, options = {}) {
      const response = await fetch(path, options);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Request failed");
      }
      return payload;
    }

    async function refreshState() {
      const payload = await api("/api/state");
      renderObjects(payload.objects);
      renderStats(payload.stats);
      renderClips(payload.clips);
      renderJob(payload.job);
      state.detectorAvailable = !!(payload.detector && payload.detector.available);
      return payload;
    }

    async function loadRawTree() {
      const tree = await api("/api/raw-tree");
      renderRawTree(tree);
    }

    function renderRawTree(tree) {
      const root = $("rawTree");
      root.innerHTML = "";
      const clips = (tree.children || []).find((child) => child.name === "clips");
      const objects = (clips && clips.children) || [];
      if (!objects.length) {
        root.innerHTML = '<div class="tree-empty">No videos in data/raw/clips yet.</div>';
        return;
      }
      for (const object of objects) {
        const group = document.createElement("div");
        group.className = "tree-object";
        const label = document.createElement("div");
        label.className = "tree-label";
        label.textContent = `${object.name} (${object.count})`;
        group.appendChild(label);
        if (!object.children.length) {
          const empty = document.createElement("div");
          empty.className = "tree-empty";
          empty.textContent = "No videos";
          group.appendChild(empty);
        }
        for (const video of object.children) {
          const row = document.createElement("div");
          row.className = "tree-video";
          row.innerHTML = `<span class="name">▶ ${video.name}</span><span class="size">${video.size}</span>`;
          row.addEventListener("click", () => openVideo(video.name, video.url));
          group.appendChild(row);
        }
        root.appendChild(group);
      }
    }

    function openVideo(name, url) {
      $("videoTitle").textContent = name;
      const player = $("videoPlayer");
      player.src = url;
      $("videoOpen").href = url;
      $("videoModal").hidden = false;
      player.load();
    }

    function closeVideo() {
      const player = $("videoPlayer");
      player.pause();
      player.removeAttribute("src");
      player.load();
      $("videoModal").hidden = true;
    }

    function renderObjects(objects) {
      const uploadSelect = $("objectSelect");
      const previousUpload = uploadSelect.value;
      uploadSelect.innerHTML = "";
      for (const object of objects) {
        const option = document.createElement("option");
        option.value = object;
        option.textContent = object;
        uploadSelect.appendChild(option);
      }
      if (objects.includes(previousUpload)) {
        uploadSelect.value = previousUpload;
      }

      const reviewSelect = $("reviewObject");
      const previousReview = reviewSelect.value || "__all__";
      reviewSelect.innerHTML = "";
      const allOption = document.createElement("option");
      allOption.value = "__all__";
      allOption.textContent = "All objects";
      reviewSelect.appendChild(allOption);
      for (const object of objects) {
        const option = document.createElement("option");
        option.value = object;
        option.textContent = object;
        reviewSelect.appendChild(option);
      }
      reviewSelect.value =
        previousReview === "__all__" || objects.includes(previousReview)
          ? previousReview
          : "__all__";
    }

    function renderStats(stats) {
      const root = $("stats");
      root.innerHTML = "";
      if (!stats.length) {
        root.innerHTML = '<div class="empty">No objects yet.</div>';
        return;
      }
      for (const item of stats) {
        const div = document.createElement("div");
        div.className = "stat";
        div.innerHTML = `
          <b>${item.review}</b>
          <span>${item.object} review frames</span>
          <div class="subtle">${item.clips} clips, ${item.rawFrames} raw, ${item.train}/${item.val} train/check</div>
        `;
        root.appendChild(div);
      }
    }

    function renderClips(clips) {
      const root = $("clipRows");
      root.innerHTML = "";
      for (const clip of clips) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${clip.object}</td>
          <td>${clip.video}</td>
          <td><span class="pill ${clip.status}">${clip.status}</span></td>
          <td>${clip.frames}</td>
        `;
        root.appendChild(tr);
      }
      if (!clips.length) {
        root.innerHTML = '<tr><td colspan="4">No clips uploaded yet.</td></tr>';
      }
    }

    function renderJob(job) {
      const dot = $("jobDot");
      const status = $("jobStatus");
      const log = $("jobLog");
      dot.className = "dot";
      if (!job) {
        status.textContent = "Idle";
        setBusy(false);
        return;
      }
      dot.classList.add(job.status);
      status.textContent = `${job.kind}: ${job.status}`;
      // Only auto-scroll if the user is already near the bottom, so scrolling up
      // to read earlier lines is not yanked back down on the next poll.
      const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
      log.textContent = job.log && job.log.length ? job.log.join("\n") : "Starting...";
      if (nearBottom) {
        log.scrollTop = log.scrollHeight;
      }
      setBusy(job.status === "running");
      if (state.lastJobStatus === "running" && job.status !== "running") {
        refreshState().catch(showError);
        loadReview().catch(showError);
        loadRawTree().catch(showError);
      }
      state.lastJobStatus = job.status;
    }

    async function loadReview() {
      const object = $("reviewObject").value || "__all__";
      const payload = await api(`/api/review?object=${encodeURIComponent(object)}`);
      state.reviewItems = payload.items || [];
      state.reviewCounts = payload.counts || { unmarked: 0, pass: 0, fail: 0 };
      renderReview();
    }

    function renderReview() {
      const counts = state.reviewCounts;
      $("countUnmarked").textContent = counts.unmarked;
      $("countPass").textContent = counts.pass;
      $("countFail").textContent = counts.fail;
      $("passAllBtn").disabled = counts.unmarked === 0;
      for (const button of document.querySelectorAll("#reviewTabs .tab")) {
        button.classList.toggle("active", button.dataset.tab === state.reviewTab);
      }

      const items = state.reviewItems;
      const filtered = items.filter((item) => item.state === state.reviewTab);
      $("reviewCount").textContent =
        `${filtered.length} ${state.reviewTab} of ${items.length} frames`;

      const root = $("gallery");
      root.innerHTML = "";
      if (!items.length) {
        root.innerHTML = '<div class="empty">No frames yet. Add clips, then click Process Data.</div>';
        return;
      }
      if (!filtered.length) {
        root.innerHTML = `<div class="empty">No ${state.reviewTab} frames.</div>`;
        return;
      }
      for (const item of filtered) {
        const frame = document.createElement("div");
        frame.className = `frame state-${item.state}`;
        frame.innerHTML = `
          <img src="${item.url}" alt="${item.name}" loading="lazy">
          <div class="frame-body">
            <div class="frame-name" title="${item.object} · ${item.name}">${item.name}</div>
            <div class="mark-row">
              <button class="mark pass ${item.state === "pass" ? "active" : ""}" title="Keep for training">Pass</button>
              <button class="mark fail ${item.state === "fail" ? "active" : ""}" title="Drop from training">Fail</button>
            </div>
          </div>
        `;
        const [passBtn, failBtn] = frame.querySelectorAll("button.mark");
        passBtn.addEventListener("click", () => markFrame(item, "pass"));
        failBtn.addEventListener("click", () => markFrame(item, "fail"));
        root.appendChild(frame);
      }
    }

    async function markFrame(item, target) {
      // Clicking the state a frame already has returns it to unmarked.
      const next = item.state === target ? "unmarked" : target;
      await api("/api/review/mark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ object: item.object, filename: item.name, state: next }),
      });
      await loadReview();
      await refreshState();
    }

    async function passAllUnmarked() {
      const object = $("reviewObject").value || "__all__";
      await api("/api/review/pass-all", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ object }),
      });
      await loadReview();
      await refreshState();
    }

    async function uploadVideos() {
      const object = activeObject();
      if (!state.files.length) {
        throw new Error("Choose at least one video first.");
      }
      const form = new FormData();
      form.append("object", object);
      for (const file of state.files) {
        form.append("files", file, file.name);
      }
      await api("/api/upload", { method: "POST", body: form });
      setFiles([]);
      $("fileInput").value = "";
      $("newObject").value = "";
      await refreshState();
    }

    async function startWorkflowJob(path, body = null) {
      await api(path, {
        method: "POST",
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
      await pollJob();
    }

    async function pollJob() {
      const payload = await api("/api/job");
      renderJob(payload.current);
    }

    function showView(view) {
      $("viewWorkflow").hidden = view !== "workflow";
      $("viewDetector").hidden = view !== "detector";
      for (const button of document.querySelectorAll(".topnav .topbtn")) {
        button.classList.toggle("active", button.dataset.view === view);
      }
      if (view !== "detector") {
        stopCamera();
      }
    }

    function setCamStatus(text, statusClass) {
      $("camStatus").textContent = text;
      $("camDot").className = "dot" + (statusClass ? " " + statusClass : "");
    }

    async function startCamera() {
      if (!state.detectorAvailable) {
        setCamStatus("No trained model found. Run Process Data then Train Data first.", "failed");
        return;
      }
      try {
        state.camStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } catch (error) {
        setCamStatus("Could not open the camera: " + (error.message || error), "failed");
        return;
      }
      const video = $("camVideo");
      video.srcObject = state.camStream;
      await video.play().catch(() => {});
      $("camHint").hidden = true;
      $("camStartBtn").disabled = true;
      $("camStopBtn").disabled = false;
      setCamStatus("Camera running. Detecting...", "running");
      // Run detection a few times per second; the flag avoids piling up requests.
      state.camTimer = setInterval(() => detectFrame().catch(() => {}), 250);
    }

    function stopCamera() {
      if (state.camTimer) {
        clearInterval(state.camTimer);
        state.camTimer = null;
      }
      if (state.camStream) {
        for (const track of state.camStream.getTracks()) {
          track.stop();
        }
        state.camStream = null;
      }
      const video = $("camVideo");
      video.srcObject = null;
      const overlay = $("camOverlay");
      const ctx = overlay.getContext("2d");
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      $("camHint").hidden = false;
      $("camStartBtn").disabled = false;
      $("camStopBtn").disabled = true;
      setCamStatus("Camera is off.");
    }

    async function detectFrame() {
      if (state.camBusy || !state.camStream) {
        return;
      }
      const video = $("camVideo");
      if (!video.videoWidth) {
        return;
      }
      state.camBusy = true;
      try {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.7));
        if (!blob || !state.camStream) {
          return;
        }
        const conf = $("camConf").value;
        const response = await fetch(`/api/detect?conf=${conf}`, { method: "POST", body: blob });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Detection failed");
        }
        drawDetections(payload.detections || []);
        const count = (payload.detections || []).length;
        setCamStatus(count ? `Detecting: ${count} object(s) in view.` : "Detecting: nothing recognized yet.", "running");
      } catch (error) {
        setCamStatus("Detector error: " + (error.message || error), "failed");
      } finally {
        state.camBusy = false;
      }
    }

    function drawDetections(detections) {
      const video = $("camVideo");
      const overlay = $("camOverlay");
      const rect = video.getBoundingClientRect();
      overlay.width = rect.width;
      overlay.height = rect.height;
      const ctx = overlay.getContext("2d");
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      // The video uses object-fit: contain, so compute the letterboxed draw area.
      const vw = video.videoWidth || 4;
      const vh = video.videoHeight || 3;
      const scale = Math.min(overlay.width / vw, overlay.height / vh);
      const drawW = vw * scale;
      const drawH = vh * scale;
      const offsetX = (overlay.width - drawW) / 2;
      const offsetY = (overlay.height - drawH) / 2;
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#16a34a";
      ctx.fillStyle = "#16a34a";
      ctx.font = "600 15px Inter, system-ui, sans-serif";
      for (const det of detections) {
        const x = offsetX + det.x * drawW;
        const y = offsetY + det.y * drawH;
        const w = det.w * drawW;
        const h = det.h * drawH;
        ctx.strokeRect(x, y, w, h);
        const label = `${det.name} ${(det.conf * 100).toFixed(0)}%`;
        const textW = ctx.measureText(label).width + 10;
        ctx.fillStyle = "#16a34a";
        ctx.fillRect(x, Math.max(0, y - 22), textW, 22);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, x + 5, Math.max(14, y - 6));
        ctx.fillStyle = "#16a34a";
      }
    }

    function showError(error) {
      alert(error.message || error);
    }

    $("dropzone").addEventListener("click", () => $("fileInput").click());
    $("dropzone").addEventListener("dragover", (event) => {
      event.preventDefault();
      $("dropzone").classList.add("drag");
    });
    $("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("drag"));
    $("dropzone").addEventListener("drop", (event) => {
      event.preventDefault();
      $("dropzone").classList.remove("drag");
      setFiles(event.dataTransfer.files);
    });
    $("fileInput").addEventListener("change", (event) => setFiles(event.target.files));
    $("clearFilesBtn").addEventListener("click", () => {
      $("fileInput").value = "";
      setFiles([]);
    });
    $("uploadBtn").addEventListener("click", () => uploadVideos().catch(showError));
    $("processBtn").addEventListener("click", () => startWorkflowJob("/api/jobs/process").catch(showError));
    $("trainBtn").addEventListener("click", () => {
      const body = {
        epochs: $("epochsInput").value.trim(),
        device: $("deviceInput").value.trim(),
      };
      startWorkflowJob("/api/jobs/train", body).catch(showError);
    });
    $("refreshReviewBtn").addEventListener("click", () => loadReview().catch(showError));
    $("reviewObject").addEventListener("change", () => loadReview().catch(showError));
    $("passAllBtn").addEventListener("click", () => passAllUnmarked().catch(showError));
    $("refreshTreeBtn").addEventListener("click", () => loadRawTree().catch(showError));
    for (const button of document.querySelectorAll("#reviewTabs .tab")) {
      button.addEventListener("click", () => {
        state.reviewTab = button.dataset.tab;
        renderReview();
      });
    }
    for (const button of document.querySelectorAll(".topnav .topbtn")) {
      button.addEventListener("click", () => showView(button.dataset.view));
    }
    $("camStartBtn").addEventListener("click", () => startCamera());
    $("camStopBtn").addEventListener("click", () => stopCamera());
    $("camConf").addEventListener("input", () => {
      $("camConfValue").textContent = Number($("camConf").value).toFixed(2);
    });
    $("clearLogViewBtn").addEventListener("click", () => {
      const log = $("jobLog");
      log.scrollTop = log.scrollHeight;
    });
    $("videoClose").addEventListener("click", closeVideo);
    $("videoModal").addEventListener("click", (event) => {
      if (event.target === $("videoModal")) {
        closeVideo();
      }
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !$("videoModal").hidden) {
        closeVideo();
      }
    });

    setInterval(() => pollJob().catch(() => {}), 1200);
    refreshState().then(loadReview).catch(showError);
    loadRawTree().catch(showError);
  </script>
</body>
</html>
"""


def build_server(host: str, port: int) -> tuple[ThreadingHTTPServer, int]:
    """Bind a server, trying nearby ports if needed."""
    last_error: OSError | None = None
    for candidate in range(port, port + 20):
        try:
            return ThreadingHTTPServer((host, candidate), WorkflowHandler), candidate
        except OSError as error:
            last_error = error
    assert last_error is not None
    raise last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local rover workflow GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server, port = build_server(args.host, args.port)
    url = f"http://{args.host}:{port}"
    print(f"Workflow GUI running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping workflow GUI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
