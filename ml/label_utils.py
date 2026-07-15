"""Read/write YOLO label files and move frames between train / rejected.

A YOLO label file holds one line per box:

    <class_id> <cx> <cy> <w> <h>

where cx, cy, w, h are fractions of the picture width/height (0..1). This module
is the only place that knows that format, plus the rules for what happens when a
frame is passed, failed, or its box is edited during review.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from ml import dataset_utils as du


# ---------------------------------------------------------------------------
# Box <-> YOLO text
# ---------------------------------------------------------------------------
def pixel_box_to_yolo(box: tuple[int, int, int, int], width: int, height: int) -> dict:
    """Turn a pixel box (x, y, w, h) into a normalized YOLO box dict."""
    x, y, w, h = box
    return {
        "cx": (x + w / 2) / width,
        "cy": (y + h / 2) / height,
        "w": w / width,
        "h": h / height,
    }


def clamp_box(box: dict) -> dict:
    """Keep a normalized box inside 0..1 and give it a sane minimum size."""
    w = min(max(float(box.get("w", 0.0)), 0.001), 1.0)
    h = min(max(float(box.get("h", 0.0)), 0.001), 1.0)
    cx = min(max(float(box.get("cx", 0.0)), w / 2), 1 - w / 2)
    cy = min(max(float(box.get("cy", 0.0)), h / 2), 1 - h / 2)
    return {"cls": int(box.get("cls", 0)), "cx": cx, "cy": cy, "w": w, "h": h}


def format_label_lines(boxes: list[dict]) -> str:
    """Turn box dicts into YOLO label file text."""
    lines = []
    for raw in boxes:
        box = clamp_box(raw)
        lines.append(
            f"{box['cls']} {box['cx']:.6f} {box['cy']:.6f} "
            f"{box['w']:.6f} {box['h']:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def read_label_file(path: Path) -> list[dict]:
    """Read a YOLO .txt file into a list of box dicts."""
    if not path.exists():
        return []
    boxes: list[dict] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            boxes.append({
                "cls": int(float(parts[0])),
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            })
        except ValueError:
            continue
    return boxes


# ---------------------------------------------------------------------------
# Where a frame currently lives
# ---------------------------------------------------------------------------
def locate_frame(stem: str) -> tuple[Path | None, str | None]:
    """Return (image_path, location) where location is 'train', 'val', or 'rejected'."""
    image, split = du.find_dataset_image(stem)
    if image is not None:
        return image, split
    for ext in du.IMAGE_EXTENSIONS:
        candidate = du.REJECTED_IMAGES_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate, "rejected"
    return None, None


def label_for_frame(stem: str, location: str) -> Path:
    """Return the label path matching a frame's current location."""
    if location == "rejected":
        return du.REJECTED_LABELS_DIR / f"{stem}.txt"
    return du.dataset_label_path(stem, location)


def read_boxes(stem: str) -> list[dict]:
    """Read the boxes for a frame wherever it currently lives."""
    _, location = locate_frame(stem)
    if location is None:
        return []
    return read_label_file(label_for_frame(stem, location))


# ---------------------------------------------------------------------------
# Move a frame between the training set and the rejected holding area
# ---------------------------------------------------------------------------
def _move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.move(str(src), str(dst))


def move_to_rejected(stem: str, split: str) -> None:
    """Move a dataset frame's image + label into data/rejected/."""
    image, found_split = du.find_dataset_image(stem)
    split = found_split or split
    if image is not None:
        _move(image, du.REJECTED_IMAGES_DIR / image.name)
    _move(du.dataset_label_path(stem, split), du.REJECTED_LABELS_DIR / f"{stem}.txt")


def restore_from_rejected(stem: str, split: str) -> None:
    """Move a rejected frame's image + label back into the training set."""
    split = split if split in du.SPLITS else "train"
    for ext in du.IMAGE_EXTENSIONS:
        image = du.REJECTED_IMAGES_DIR / f"{stem}{ext}"
        if image.exists():
            _move(image, du.DATASET_IMAGES_DIR / split / image.name)
            break
    label = du.REJECTED_LABELS_DIR / f"{stem}.txt"
    if label.exists():
        _move(label, du.dataset_label_path(stem, split))


# ---------------------------------------------------------------------------
# Review decisions (Pass / Fail / Edit / Skip)
# ---------------------------------------------------------------------------
def _stored_split(state: dict[str, dict], stem: str) -> str:
    entry = state.get(stem)
    if isinstance(entry, dict) and entry.get("split") in du.SPLITS:
        return entry["split"]
    _, location = locate_frame(stem)
    return location if location in du.SPLITS else "train"


def set_review_decision(stem: str, decision: str) -> dict:
    """Apply Pass / Fail / Unreviewed to a frame and move files as needed.

    'passed'/'unreviewed' keep the frame in training; 'failed' moves it to
    data/rejected/. Editing boxes is handled by save_boxes(), which marks a
    frame 'edited'.
    """
    if decision not in {"passed", "failed", "unreviewed"}:
        raise ValueError("decision must be passed, failed, or unreviewed.")

    state = du.load_review_state()
    _, location = locate_frame(stem)
    if location is None:
        raise ValueError(f"Frame not found: {stem}")
    split = _stored_split(state, stem)
    cls = du.class_from_stem(stem)

    if decision == "failed":
        if location != "rejected":
            move_to_rejected(stem, split)
        state[stem] = {"state": "failed", "split": split, "class": cls}
    else:
        if location == "rejected":
            restore_from_rejected(stem, split)
        if decision == "passed":
            state[stem] = {"state": "passed", "split": split, "class": cls}
        else:  # unreviewed
            state.pop(stem, None)

    du.save_review_state(state)
    return {"stem": stem, "state": decision, "split": split}


def save_boxes(stem: str, boxes: list[dict]) -> dict:
    """Write edited boxes back to the label file and mark the frame 'edited'.

    Editing always keeps the frame in the training set: a failed frame whose box
    is corrected is restored to training (corrected mistakes are valuable data).
    """
    state = du.load_review_state()
    _, location = locate_frame(stem)
    if location is None:
        raise ValueError(f"Frame not found: {stem}")
    split = _stored_split(state, stem)

    if location == "rejected":
        restore_from_rejected(stem, split)

    du.dataset_label_path(stem, split).parent.mkdir(parents=True, exist_ok=True)
    du.dataset_label_path(stem, split).write_text(format_label_lines(boxes))

    primary = du.class_from_stem(stem)
    if boxes:
        names = du.class_names()
        cls_id = int(boxes[0].get("cls", 0))
        if 0 <= cls_id < len(names):
            primary = names[cls_id]
    state[stem] = {"state": "edited", "split": split, "class": primary}
    du.save_review_state(state)
    return {"stem": stem, "state": "edited", "split": split, "boxes": len(boxes)}


def promote_frame_to_dataset(image_src: Path, stem: str, boxes: list[dict],
                             split: str = "train") -> dict:
    """Copy a corrected frame + write its label into the training dataset.

    Used by the retraining queue to fold a corrected failure frame back into
    yolo_dataset/.
    """
    split = split if split in du.SPLITS else "train"
    dst_image = du.dataset_image_path(stem, split)
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(image_src), str(dst_image))
    du.dataset_label_path(stem, split).parent.mkdir(parents=True, exist_ok=True)
    du.dataset_label_path(stem, split).write_text(format_label_lines(boxes))

    state = du.load_review_state()
    state[stem] = {"state": "edited", "split": split, "class": du.class_from_stem(stem)}
    du.save_review_state(state)
    return {"stem": stem, "split": split, "image": du.repo_relative(dst_image)}


# ---------------------------------------------------------------------------
# Retraining queue: failure frames captured during live testing (Tab 5),
# corrected and folded back into the dataset (Tab 6).
# ---------------------------------------------------------------------------
def retrain_image_path(name: str) -> Path:
    return du.RETRAIN_IMAGES_DIR / du.safe_filename(name)


def save_retrain_frame(image_bytes: bytes, meta: dict) -> dict:
    """Store a captured failure frame + its metadata sidecar in the queue."""
    du.RETRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ftype = du.safe_filename(str(meta.get("failureType", "failure")) or "failure")[:40]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = du.unique_path(du.RETRAIN_IMAGES_DIR / f"{stamp}_{ftype}.jpg")
    image_path.write_bytes(image_bytes)
    entry = {
        "image": image_path.name,
        "created": time.time(),
        "status": "pending",
        "failureType": str(meta.get("failureType", "")),
        "model": str(meta.get("model", "")),
        "note": str(meta.get("note", "")),
        "boxes": meta.get("boxes", []),
    }
    du.write_json(image_path.with_suffix(".json"), entry)
    return entry


def list_retrain_frames() -> list[dict]:
    """List every retrain-queue entry (pending, corrected, discarded)."""
    items: list[dict] = []
    if not du.RETRAIN_IMAGES_DIR.exists():
        return items
    for sidecar in sorted(du.RETRAIN_IMAGES_DIR.glob("*.json")):
        data = du.read_json(sidecar, {})
        if not isinstance(data, dict):
            continue
        name = data.get("image", sidecar.stem + ".jpg")
        data["name"] = name
        data["hasImage"] = (du.RETRAIN_IMAGES_DIR / name).exists()
        items.append(data)
    return items


def retrain_counts() -> dict:
    counts = {"total": 0, "pending": 0, "corrected": 0, "discarded": 0}
    for item in list_retrain_frames():
        counts["total"] += 1
        counts[item.get("status", "pending")] = counts.get(item.get("status", "pending"), 0) + 1
    return counts


def _update_retrain_sidecar(name: str, **changes) -> None:
    sidecar = retrain_image_path(name).with_suffix(".json")
    data = du.read_json(sidecar, {})
    if not isinstance(data, dict):
        data = {}
    data.update(changes)
    du.write_json(sidecar, data)


def discard_retrain_frame(name: str) -> dict:
    """Mark a failure frame useless and remove its image (keep the record)."""
    image = retrain_image_path(name)
    _update_retrain_sidecar(name, status="discarded")
    if image.exists():
        image.unlink()
    return {"ok": True, "status": "discarded"}


def promote_retrain_frame(name: str, boxes: list[dict], split: str = "train") -> dict:
    """Fold a corrected failure frame into the training dataset."""
    image = retrain_image_path(name)
    if not image.exists():
        raise ValueError("Retrain frame not found (already corrected or discarded).")
    if not boxes:
        raise ValueError("Draw at least one box before adding this frame.")
    names = du.class_names()
    cls_id = int(boxes[0].get("cls", 0))
    class_name = names[cls_id] if 0 <= cls_id < len(names) else (names[0] if names else "object")
    stem = f"{class_name}__retrain_{Path(name).stem}"
    result = promote_frame_to_dataset(image, stem, boxes, split)
    _update_retrain_sidecar(name, status="corrected", promotedStem=stem)
    image.unlink()
    return {"ok": True, "status": "corrected", **result}
