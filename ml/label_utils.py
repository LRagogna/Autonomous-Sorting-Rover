"""Read/write YOLO label files and move frames between train / rejected.

A YOLO label file holds one line per box:

    <class_id> <cx> <cy> <w> <h>

where cx, cy, w, h are fractions of the picture width/height (0..1). This module
is the only place that knows that format, plus the rules for what happens when a
frame is passed, failed, or its box is edited during review.
"""

from __future__ import annotations

import shutil
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

    Used by the retraining queue (Phase 2) to fold a corrected failure frame
    back into yolo_dataset/.
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
