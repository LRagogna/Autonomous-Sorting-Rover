"""Process Dataset job: extract frames from selected clips, then auto-label.

This is what the GUI's "Process Dataset" tab runs as one streamed job. It:

  1. extracts still frames from the chosen clips (or --all new clips), then
  2. auto-labels every not-yet-labeled clip and rebuilds dataset.yaml.

It prints a machine-readable PROCESS_SUMMARY line at the end that the GUI parses
to show frames extracted / labels created / auto-label failures / needing review.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml import auto_label_frames as al  # noqa: E402
from ml import dataset_utils as du  # noqa: E402
from ml import extract_frames as ef  # noqa: E402
from ml import label_utils as lu  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from clips and auto-label them.")
    parser.add_argument("--all", action="store_true", help="Process every not-yet-extracted clip.")
    parser.add_argument("--select", nargs="*", default=None,
                        help="Clips to process as class/video keys, e.g. bit/IMG_1297.MOV.")
    parser.add_argument("--frame-step", type=int, default=15, help="Save every Nth frame.")
    parser.add_argument("--reprocess", action="store_true", help="Re-extract clips that already have frames.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Share of clips used for validation.")
    parser.add_argument("--pad", type=float, default=0.06, help="Padding around auto-drawn boxes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    du.ensure_core_dirs()

    print("==> Step 1/3: extracting frames from clips")
    try:
        videos = ef.resolve_selection(args.select, args.all)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    frames_extracted = 0
    if not videos:
        print("  no clips selected — skipping extraction")
    for class_name, video_path in videos:
        if not args.reprocess and ef.has_existing_frames(class_name, video_path):
            print(f"  skip {class_name}/{video_path.name}: frames already extracted")
            continue
        print(f"  extracting {class_name}/{video_path.name} (every {args.frame_step} frames)...")
        saved = ef.extract_video(video_path, class_name, args.frame_step)
        print(f"    saved {saved} frame(s)")
        frames_extracted += saved

    print("\n==> Step 2/3: auto-labeling frames and building the dataset")
    result = al.run(overwrite=False, incremental=True,
                    val_fraction=args.val_fraction, pad=args.pad)
    labels = result.get("labels_created", 0)
    no_box = result.get("no_box", 0)
    if not result.get("ok"):
        # Not fatal on its own: there may still be background pictures to fold in
        # and a dataset built on a previous run. Warn and keep going.
        print(f"  note: auto-labeling added no new object labels "
              f"({result.get('reason', 'nothing new')}).", file=sys.stderr)

    print("\n==> Step 3/3: adding background images (negative examples)")
    background = lu.ingest_background_images(val_fraction=args.val_fraction)
    bg_added = background.get("added", 0)
    print(f"  background pictures added: {bg_added} "
          f"(skipped {background.get('skipped', 0)} already in the dataset)")
    print(f"  background images in dataset: {background.get('total', 0)}")

    print("\n==> Done processing.")
    print(f"  frames extracted:      {frames_extracted}")
    print(f"  labels created:        {labels}")
    print(f"  auto-label failures:   {no_box}")
    print(f"  frames needing review: {labels}")
    print(f"  backgrounds added:     {bg_added}")
    print(f"PROCESS_SUMMARY extracted={frames_extracted} labels={labels} "
          f"failed={no_box} review={labels} backgrounds={bg_added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
