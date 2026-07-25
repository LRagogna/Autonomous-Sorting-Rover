"""Tab 1 - Upload Clips: manage classes and raw video clips."""

from __future__ import annotations

import urllib.parse
from pathlib import Path

from gui import state as state_module
from gui.server import route, serve_range
from ml import dataset_utils as du
from ml import extract_frames as ef


def _clip_view(class_name: str, video: Path) -> dict:
    seconds = du.video_duration_seconds(video)
    return {
        "class": class_name,
        "video": video.name,
        "size": du.format_size(video.stat().st_size),
        "durationSeconds": round(seconds, 2),
        "durationLabel": du.format_duration(seconds) if seconds else "—",
        "processed": ef.has_existing_frames(class_name, video),
        "url": f"/media/clips/{urllib.parse.quote(class_name)}/{urllib.parse.quote(video.name)}",
    }


@route("GET", "/api/clips")
def list_clips(req):
    groups = []
    if du.RAW_VIDEOS_DIR.exists():
        for class_dir in sorted(p for p in du.RAW_VIDEOS_DIR.iterdir()
                                if p.is_dir() and not p.name.startswith(".")
                                and not du.is_background_class(p.name)):
            videos = sorted(p for p in class_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in du.VIDEO_EXTENSIONS)
            clips = [_clip_view(class_dir.name, v) for v in videos]
            total = sum(c["durationSeconds"] for c in clips)
            groups.append({
                "class": class_dir.name,
                "clipCount": len(clips),
                "totalDurationLabel": du.format_duration(total) if total else "—",
                "clips": clips,
            })
    return {"classes": du.class_names(), "groups": groups}


@route("POST", "/api/clips/create-class")
def create_class(req):
    name = du.validate_class_name(req.json().get("name", ""))
    du.ensure_class(name)
    (du.RAW_VIDEOS_DIR / name).mkdir(parents=True, exist_ok=True)
    return {"ok": True, "class": name, "state": state_module.build_state()}


@route("POST", "/api/upload")
def upload_clips(req):
    fields, files = req.multipart()
    class_name = du.validate_class_name(fields.get("class") or fields.get("object") or "")
    du.ensure_class(class_name)
    target_dir = du.RAW_VIDEOS_DIR / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    if not files:
        raise ValueError("No video files were uploaded.")
    saved = []
    for item in files:
        filename = du.safe_filename(item["filename"])
        if Path(filename).suffix.lower() not in du.VIDEO_EXTENSIONS:
            raise ValueError(f"{item['filename']} is not a supported video file.")
        target = du.unique_path(target_dir / filename)
        target.write_bytes(item["content"])
        saved.append(target.name)
    return {"ok": True, "saved": saved, "class": class_name, "state": state_module.build_state()}


def _video_path(class_name: str, video: str) -> Path:
    class_name = du.validate_class_name(class_name)
    video = du.safe_filename(video)
    path = du.RAW_VIDEOS_DIR / class_name / video
    if not path.exists() or path.suffix.lower() not in du.VIDEO_EXTENSIONS:
        raise ValueError("Clip not found.")
    return path


@route("POST", "/api/clips/rename")
def rename_clip(req):
    body = req.json()
    path = _video_path(body.get("class", ""), body.get("video", ""))
    new_name = du.safe_filename(body.get("newName", ""))
    if Path(new_name).suffix.lower() not in du.VIDEO_EXTENSIONS:
        new_name = f"{Path(new_name).stem}{path.suffix}"
    target = du.unique_path(path.with_name(new_name))
    path.rename(target)

    clips = du.load_processed_clips()
    old_key = du.clip_key(path.parent.name, path.name)
    if old_key in clips:
        entry = clips.pop(old_key)
        entry["video"] = target.name
        entry["path"] = du.repo_relative(target)
        clips[du.clip_key(target.parent.name, target.name)] = entry
        du.save_processed_clips(clips)
    return {"ok": True, "video": target.name, "state": state_module.build_state()}


@route("POST", "/api/clips/delete")
def delete_clip(req):
    body = req.json()
    path = _video_path(body.get("class", ""), body.get("video", ""))
    key = du.clip_key(path.parent.name, path.name)
    path.unlink()
    clips = du.load_processed_clips()
    if key in clips:
        clips.pop(key)
        du.save_processed_clips(clips)
    return {"ok": True, "state": state_module.build_state()}


@route("GET", "/media/clips/{class_name}/{video}")
def stream_clip(req):
    class_name = du.validate_class_name(req.params["class_name"])
    video = du.safe_filename(req.params["video"])
    serve_range(req.handler, du.RAW_VIDEOS_DIR / class_name / video)
