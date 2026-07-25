#!/usr/bin/env bash
# Remove iCloud / Finder "conflict copy" duplicates from the whole repo.
#
# WHY THIS EXISTS
#   This project lives on an iCloud-synced Desktop. When iCloud syncs a folder
#   that is changing, it sometimes saves a second copy with a " 2" / " 3" suffix
#   instead of merging. That litters the repo with duplicate files
#   (e.g. "dataset 2.yaml", "IMG_1297__frame_000000 3.jpg") and empty duplicate
#   folders (e.g. "images 2"). These break training and confuse which file is real.
#
# WHAT IT DOES
#   Deletes files and directories whose name ends in " <number>" (a copy suffix),
#   because none of this project's real files or folders use that pattern (real
#   names use underscores and no spaces before the extension). Non-empty duplicate
#   folders are reported but NOT deleted, so real data is never lost silently.
#
# USAGE
#   ./scripts/clean_icloud_dupes.sh          # delete the duplicates
#   ./scripts/clean_icloud_dupes.sh --dry-run # only list what would be deleted
set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

# Never touch the virtualenv or git internals.
PRUNE=(-path ./.venv -prune -o -path ./.git -prune -o)

echo "==> Scanning for iCloud/Finder copy duplicates..."
found=0

# 1. Duplicate FILES: "name 2.ext" or "name 2" (no extension).
while IFS= read -r -d '' f; do
  found=1
  echo "  file: $f"
  [[ $DRY_RUN -eq 0 ]] && rm -f "$f"
done < <(find . "${PRUNE[@]}" -type f \( -name '* [0-9].*' -o -name '* [0-9]' \) -print0 2>/dev/null)

# 2. Duplicate DIRECTORIES: "name 2". Deepest-first so nested copies collapse.
# Only remove EMPTY copy folders; report any that still hold files so real data
# is never lost silently.
while IFS= read -r d; do
  [[ -d "$d" ]] || continue
  found=1
  if [[ -z "$(find "$d" -type f -print -quit 2>/dev/null)" ]]; then
    echo "  dir (empty): $d"
    [[ $DRY_RUN -eq 0 ]] && rm -rf "$d"
  else
    echo "  dir (NOT EMPTY - left in place, check by hand): $d"
  fi
done < <(find . "${PRUNE[@]}" -type d -name '* [0-9]' -print 2>/dev/null | awk '{print length"\t"$0}' | sort -rn | cut -f2-)

if [[ $found -eq 0 ]]; then
  echo "    No duplicates found. Repo is clean."
  exit 0
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "==> Dry run only. Re-run without --dry-run to delete."
else
  echo "==> Done."
fi
