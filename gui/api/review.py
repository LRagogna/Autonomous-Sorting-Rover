"""Tab 3 - Review / Edit Labels: inspect, correct, pass, or fail each frame."""

from __future__ import annotations

import urllib.parse

from gui import state as state_module
from gui.server import route, send_image
from ml import dataset_utils as du
from ml import label_utils as lu


def _all_items() -> list[dict]:
    """Enumerate every frame (dataset + rejected) with its state and boxes."""
    review = du.load_review_state()
    items: list[dict] = []

    def add(stem: str, location: str):
        if location == "rejected":
            state = "failed"
        else:
            entry = review.get(stem)
            state = entry.get("state", "unreviewed") if isinstance(entry, dict) else "unreviewed"
        items.append({
            "stem": stem,
            "class": du.class_from_stem(stem),
            "source": du.source_video_from_stem(stem),
            "state": state,
            "location": location,
            "url": "/api/review/image/" + urllib.parse.quote(stem),
            "boxes": lu.read_boxes(stem),
        })

    for split in du.SPLITS:
        directory = du.DATASET_IMAGES_DIR / split
        if directory.exists():
            for path in sorted(directory.iterdir()):
                if (path.is_file() and path.suffix.lower() in du.IMAGE_EXTENSIONS
                        and not du.is_background_stem(path.stem)):
                    add(path.stem, split)
    if du.REJECTED_IMAGES_DIR.exists():
        for path in sorted(du.REJECTED_IMAGES_DIR.iterdir()):
            if path.is_file() and path.suffix.lower() in du.IMAGE_EXTENSIONS:
                add(path.stem, "rejected")
    return items


@route("GET", "/api/review/frames")
def list_frames(req):
    class_filter = req.q("class")
    video_filter = req.q("video")
    state_filter = req.q("state", "all")

    items = _all_items()
    classes = du.class_names()
    videos = sorted({(it["class"], it["source"]) for it in items})

    scoped = [
        it for it in items
        if (not class_filter or class_filter in ("__all__", "") or it["class"] == class_filter)
        and (not video_filter or video_filter in ("__all__", "") or it["source"] == video_filter)
    ]
    counts = {"unreviewed": 0, "passed": 0, "failed": 0, "edited": 0}
    for it in scoped:
        counts[it["state"]] = counts.get(it["state"], 0) + 1

    if state_filter and state_filter not in ("all", ""):
        visible = [it for it in scoped if it["state"] == state_filter]
    else:
        visible = scoped

    return {
        "items": visible,
        "counts": counts,
        "total": len(scoped),
        "classes": classes,
        "videos": [{"class": c, "video": v} for c, v in videos],
    }


@route("GET", "/api/review/image/{stem}")
def review_image(req):
    stem = du.safe_filename(req.params["stem"])
    stem = stem.rsplit(".", 1)[0] if stem.lower().endswith((".jpg", ".jpeg", ".png")) else stem
    path, _ = lu.locate_frame(stem)
    if path is None:
        send_image(req.handler, du.DATASET_IMAGES_DIR / "train" / f"{stem}.jpg")  # -> 404
        return
    send_image(req.handler, path)


@route("POST", "/api/review/mark")
def mark_frame(req):
    body = req.json()
    result = lu.set_review_decision(str(body.get("stem", "")), str(body.get("decision", "")))
    return {"ok": True, "result": result, "state": state_module.build_state()}


@route("POST", "/api/review/save")
def save_frame(req):
    body = req.json()
    stem = str(body.get("stem", ""))
    boxes = body.get("boxes") or []
    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list.")
    result = lu.save_boxes(stem, boxes)
    return {"ok": True, "result": result, "state": state_module.build_state()}


@route("POST", "/api/review/pass-unreviewed")
def pass_unreviewed(req):
    """Pass every currently unreviewed frame (optionally within a class/video)."""
    body = req.json()
    class_filter = str(body.get("class", "")) or "__all__"
    video_filter = str(body.get("video", "")) or "__all__"
    passed = 0
    for it in _all_items():
        if it["state"] != "unreviewed":
            continue
        if class_filter not in ("__all__", "") and it["class"] != class_filter:
            continue
        if video_filter not in ("__all__", "") and it["source"] != video_filter:
            continue
        lu.set_review_decision(it["stem"], "passed")
        passed += 1
    return {"ok": True, "passed": passed, "state": state_module.build_state()}
