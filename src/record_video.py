"""Record a video on the Raspberry Pi camera until you type 'q' in the terminal.

This is a headless data-collection helper for the rover. Run it on the Pi (which
has the OV5647 CSI camera), point the camera at an object, and it records until
you stop it. The clips are meant to be copied over to the training computer and
dropped into ``data/raw_videos/<class>/`` (or ``data/raw_videos/background/``).

BEHAVIOR

    1. Prints "Started Recording" and immediately begins recording.
    2. Keeps recording until you type ``q`` (then Enter) in the terminal.
    3. Saves the clip as an .mp4 in the ``pi_videos/`` folder.

USAGE

    python3 src/record_video.py
    python3 src/record_video.py --outdir pi_videos --width 1280 --height 720 --fps 30

REQUIREMENTS

    Uses Picamera2, which ships with Raspberry Pi OS (Bookworm). If it is missing:

        sudo apt install -y python3-picamera2 ffmpeg

    ffmpeg is what turns the camera's H.264 stream into a playable .mp4.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a Pi camera video until 'q' is entered in the terminal."
    )
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "pi_videos"),
                        help="Folder to save recordings in (default: pi_videos/).")
    parser.add_argument("--width", type=int, default=1280, help="Frame width (default 1280).")
    parser.add_argument("--height", type=int, default=720, help="Frame height (default 720).")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default 30).")
    parser.add_argument("--bitrate", type=int, default=10_000_000,
                        help="H.264 bitrate in bits/sec (default 10 Mbps).")
    return parser.parse_args()


def wait_for_quit() -> None:
    """Block until the user types 'q' (then Enter). Ctrl-C / EOF also stop us."""
    while True:
        try:
            line = input()
        except EOFError:  # stdin closed (e.g. piped input ended)
            return
        if line.strip().lower() == "q":
            return


def main() -> int:
    args = parse_args()

    try:
        from picamera2 import Picamera2
        from picamera2.encoders import H264Encoder
        from picamera2.outputs import FfmpegOutput
    except ModuleNotFoundError:
        print(
            "Picamera2 is not available. Run this on the Raspberry Pi, and if needed:\n"
            "  sudo apt install -y python3-picamera2 ffmpeg",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / time.strftime("rover_%Y%m%d_%H%M%S.mp4")

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (args.width, args.height)},
        controls={"FrameRate": args.fps},
    )
    picam2.configure(config)

    encoder = H264Encoder(bitrate=args.bitrate)
    output = FfmpegOutput(str(out_path))

    picam2.start_recording(encoder, output)
    print("Started Recording", flush=True)
    print(f"Recording to {out_path} — type 'q' then Enter to stop.", flush=True)

    try:
        wait_for_quit()
    except KeyboardInterrupt:
        pass  # Ctrl-C is a normal way to stop; still save the clip below.
    finally:
        picam2.stop_recording()
        picam2.close()

    print(f"Stopped. Saved recording to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
