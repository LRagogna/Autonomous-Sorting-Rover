"""Remove iCloud "conflict copy" duplicates from the project tree.

This project lives on an iCloud-synced Desktop. When iCloud re-downloads an
evicted file it sometimes leaves a second copy whose name has a space + number
appended before the extension:

    IMG_1297.MOV      <- the real file
    IMG_1297 2.MOV    <- iCloud duplicate (junk)

    gui/web/tabs      <- the real folder
    gui/web/tabs 2    <- iCloud duplicate (junk)

These duplicates confuse the training pipeline (extra "new" clips, duplicate
frames, stale copies of code/config), so they must be swept away. This module
finds and deletes them.

SAFETY

    A ``<name> <N>`` item is only deleted when the CANONICAL original
    (``<name>`` with the same extension) exists as a sibling. Because the
    original is always kept, this can never delete unique data — only redundant
    iCloud copies. VCS, virtualenv, and cache folders are never scanned.

USAGE

    python ml/dedupe.py            # delete duplicates, print what was removed
    python ml/dedupe.py --dry-run  # just list what WOULD be removed
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folders we never descend into (their contents are not project data).
EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".vscode",
}

# The iCloud marker: the stem ends with a space followed by one or more digits.
_DUP_STEM_RE = re.compile(r"^(?P<base>.+) \d+$")


def canonical_sibling(path: Path) -> Path | None:
    """Return the original this path duplicates, or None if it is not a duplicate.

    Splits ``<base> <N><suffix>`` and checks that ``<base><suffix>`` exists next
    to it. Directories and dotfiles (no suffix) are handled too.
    """
    name = path.name
    suffix = "" if path.is_dir() else path.suffix
    stem = name[: len(name) - len(suffix)] if suffix else name
    match = _DUP_STEM_RE.match(stem)
    if not match:
        return None
    canonical = path.with_name(f"{match.group('base')}{suffix}")
    if canonical != path and canonical.exists():
        return canonical
    return None


def find_duplicates(root: Path = PROJECT_ROOT) -> list[Path]:
    """List every iCloud duplicate file/folder under ``root`` (depth-first).

    A duplicate directory is reported but not descended into (the whole folder
    goes), so its children are not listed separately.
    """
    duplicates: list[Path] = []

    def walk(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.is_dir() and entry.name in EXCLUDE_DIRS:
                continue
            if canonical_sibling(entry) is not None:
                duplicates.append(entry)
                continue  # a duplicate folder is removed whole — don't descend
            if entry.is_dir() and not entry.is_symlink():
                walk(entry)

    walk(root)
    return duplicates


def remove_duplicates(root: Path = PROJECT_ROOT, dry_run: bool = False) -> list[str]:
    """Delete iCloud duplicates under ``root``. Returns the repo-relative paths.

    Set ``dry_run=True`` to list what would be removed without deleting anything.
    Deletion failures are skipped rather than raised, so this can run safely at
    GUI startup without ever blocking it.
    """
    removed: list[str] = []
    for path in find_duplicates(root):
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = str(path)
        if not dry_run:
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError as error:
                print(f"  could not remove {rel}: {error}", file=sys.stderr)
                continue
        removed.append(rel)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove iCloud duplicate files/folders.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List duplicates without deleting them.")
    args = parser.parse_args()

    removed = remove_duplicates(dry_run=args.dry_run)
    verb = "Would remove" if args.dry_run else "Removed"
    if not removed:
        print("No iCloud duplicate files or folders found.")
        return 0
    for rel in removed:
        print(f"  {verb}: {rel}")
    print(f"{verb} {len(removed)} iCloud duplicate item(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
