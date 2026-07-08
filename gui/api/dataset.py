"""Tab 2 - Process Dataset: extract frames from clips, then auto-label."""

from __future__ import annotations

import sys

from gui import jobs
from gui.server import route
from ml import dataset_utils as du
from ml import extract_frames as ef


@route("GET", "/api/dataset/clips")
def processable_clips(req):
    """List every clip with whether it already has extracted frames."""
    groups = []
    if du.RAW_VIDEOS_DIR.exists():
        for class_dir in sorted(p for p in du.RAW_VIDEOS_DIR.iterdir()
                                if p.is_dir() and not p.name.startswith(".")):
            clips = []
            for video in sorted(p for p in class_dir.iterdir()
                                if p.is_file() and p.suffix.lower() in du.VIDEO_EXTENSIONS):
                clips.append({
                    "class": class_dir.name,
                    "video": video.name,
                    "key": du.clip_key(class_dir.name, video.name),
                    "extracted": ef.has_existing_frames(class_dir.name, video),
                })
            if clips:
                groups.append({"class": class_dir.name, "clips": clips})
    return {"groups": groups}


@route("POST", "/api/dataset/process")
def start_processing(req):
    body = req.json()
    frame_step = int(body.get("frameStep", 15) or 15)
    frame_step = max(1, min(frame_step, 300))
    reprocess = bool(body.get("reprocess"))
    selected = body.get("select") or []

    command = [sys.executable, "-u", "ml/process_dataset.py", "--frame-step", str(frame_step)]
    if reprocess:
        command.append("--reprocess")
    if selected and not body.get("all"):
        command.append("--select")
        command.extend(str(key) for key in selected)
    else:
        command.append("--all")

    job = jobs.start_job("process_dataset", command)
    return {"ok": True, "job": jobs.public_job(job)}
