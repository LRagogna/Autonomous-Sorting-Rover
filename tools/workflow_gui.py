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


def list_review_images(object_name: str) -> dict:
    """List review images for one object."""
    object_name = validate_object_name(object_name)
    review_dir = LABELS_DIR / object_name / "review"
    images = []
    if review_dir.exists():
        for path in sorted(review_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(
                    {
                        "name": path.name,
                        "url": (
                            "/review/"
                            + urllib.parse.quote(object_name)
                            + "/"
                            + urllib.parse.quote(path.name)
                        ),
                    }
                )
    return {"object": object_name, "images": images}


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
                json_response(self, list_review_images(object_name))
            elif path.startswith("/review/"):
                parts = path.split("/", 3)
                if len(parts) == 4:
                    serve_review_image(self, parts[2], parts[3])
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
    }
    section, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
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
      position: sticky;
      top: 0;
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
    .job-log {
      height: 210px;
      overflow: auto;
      margin: 0;
      padding: 12px;
      border-radius: 8px;
      background: #111827;
      color: #d1d5db;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      white-space: pre-wrap;
    }
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
      grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
      gap: 12px;
      padding: 16px;
      align-content: start;
    }
    .frame {
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
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
      gap: 8px;
      padding: 10px;
    }
    .frame-name {
      min-height: 32px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
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

      <section class="panel">
        <h2>Job Log</h2>
        <pre id="jobLog" class="job-log">No job has run yet.</pre>
      </section>
    </div>

    <section class="review">
      <div class="review-head">
        <div>
          <h2>Review Frames</h2>
          <p id="reviewCount" class="subtle">Select an object to review.</p>
        </div>
        <div class="review-controls">
          <div>
            <label for="reviewObject">Review object</label>
            <select id="reviewObject"></select>
          </div>
          <button id="refreshReviewBtn" class="neutral">Refresh</button>
        </div>
      </div>
      <div id="gallery" class="gallery"></div>
    </section>
  </main>

  <script>
    const state = {
      files: [],
      lastJobStatus: null,
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
      return payload;
    }

    function renderObjects(objects) {
      for (const select of [$("objectSelect"), $("reviewObject")]) {
        const previous = select.value;
        select.innerHTML = "";
        for (const object of objects) {
          const option = document.createElement("option");
          option.value = object;
          option.textContent = object;
          select.appendChild(option);
        }
        if (objects.includes(previous)) {
          select.value = previous;
        }
      }
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
      log.textContent = job.log && job.log.length ? job.log.join("\n") : "Starting...";
      log.scrollTop = log.scrollHeight;
      setBusy(job.status === "running");
      if (state.lastJobStatus === "running" && job.status !== "running") {
        refreshState().then(loadReview).catch(showError);
      }
      state.lastJobStatus = job.status;
    }

    async function loadReview() {
      const object = $("reviewObject").value || $("objectSelect").value;
      if (!object) {
        $("gallery").innerHTML = '<div class="empty">No object selected.</div>';
        return;
      }
      const payload = await api(`/api/review?object=${encodeURIComponent(object)}`);
      $("reviewCount").textContent = `${payload.images.length} review frames for ${payload.object}`;
      const root = $("gallery");
      root.innerHTML = "";
      if (!payload.images.length) {
        root.innerHTML = '<div class="empty">No review images yet. Run Process Data.</div>';
        return;
      }
      for (const image of payload.images) {
        const frame = document.createElement("div");
        frame.className = "frame";
        frame.innerHTML = `
          <img src="${image.url}" alt="${image.name}" loading="lazy">
          <div class="frame-body">
            <div class="frame-name">${image.name}</div>
            <button class="danger" data-name="${image.name}">Delete Frame</button>
          </div>
        `;
        frame.querySelector("button").addEventListener("click", () => deleteFrame(image.name));
        root.appendChild(frame);
      }
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

    async function deleteFrame(filename) {
      const object = $("reviewObject").value;
      if (!confirm(`Delete ${filename}?`)) {
        return;
      }
      await api("/api/review/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ object, filename }),
      });
      await refreshState();
      await loadReview();
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

    setInterval(() => pollJob().catch(() => {}), 1200);
    refreshState().then(loadReview).catch(showError);
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
