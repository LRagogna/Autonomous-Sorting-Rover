"""Tab 6 - Retraining Queue: correct captured failure frames and reuse them."""

from __future__ import annotations

import urllib.parse

from gui import state as state_module
from gui.server import route, send_image
from ml import dataset_utils as du
from ml import label_utils as lu


@route("GET", "/api/retrain/list")
def list_queue(req):
    items = []
    for entry in lu.list_retrain_frames():
        items.append({
            **entry,
            "url": "/api/retrain/image/" + urllib.parse.quote(entry["name"]),
        })
    # Pending first, then corrected, then discarded; newest first within each.
    order = {"pending": 0, "corrected": 1, "discarded": 2}
    items.sort(key=lambda it: (order.get(it.get("status"), 9), -(it.get("created") or 0)))
    return {"items": items, "counts": lu.retrain_counts(), "classes": du.class_names()}


@route("GET", "/api/retrain/image/{name}")
def queue_image(req):
    send_image(req.handler, lu.retrain_image_path(req.params["name"]))


@route("POST", "/api/retrain/promote")
def promote(req):
    body = req.json()
    boxes = body.get("boxes") or []
    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list.")
    result = lu.promote_retrain_frame(str(body.get("name", "")), boxes,
                                      str(body.get("split", "train")))
    return {"ok": True, "result": result, "counts": lu.retrain_counts(),
            "state": state_module.build_state()}


@route("POST", "/api/retrain/background")
def add_background(req):
    """Turn a false-positive capture into a background (negative) training image."""
    result = lu.background_retrain_frame(str(req.json().get("name", "")))
    return {"ok": True, "result": result, "counts": lu.retrain_counts(),
            "state": state_module.build_state()}


@route("POST", "/api/retrain/discard")
def discard(req):
    result = lu.discard_retrain_frame(str(req.json().get("name", "")))
    return {"ok": True, "result": result, "counts": lu.retrain_counts(),
            "state": state_module.build_state()}
