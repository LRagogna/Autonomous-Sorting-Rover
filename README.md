# Autonomous-Sorting-Rover

Autonomous robotic rover that detects, collects, and sorts user-prompted items.

## Overview

This project is an early-stage autonomous sorting rover. The rover is being designed around a tank-style chassis, Raspberry Pi based control, camera-driven object recognition, and an electromagnet for collecting metal debris.

The current proof of concept focuses on computer vision: using a Raspberry Pi camera and OpenCV to detect green objects in the camera feed.

## Current Hardware Direction

- Raspberry Pi 4 for main control and decision making
- OV5647 Raspberry Pi camera for visual detection
- Tank-style drive layout with each side moving together
- Relay-controlled electromagnet for metal object pickup
- Possible Coral TPU for future object recognition acceleration

## Repository Layout

```text
README.md
requirements.txt
data/
  extract_video_frames.py
  labels/
  processed/
  raw/
    photos/
      washer/
        pan_01.mp4/
    videos/
      washer/
docs/
  images/
  progress_log.md
scripts/
  setup_pi_sparse_checkout.sh
tests/
  rectangle_detect.py
```

## Current Prototype

`tests/rectangle_detect.py` captures frames from the Raspberry Pi camera, converts them to HSV color space, thresholds for green objects, cleans the mask, finds contours, and draws bounding boxes around detected objects.

The script is intended to run on a Raspberry Pi with the camera connected.

## Setup Notes

Install Python dependencies:

```bash
pip install -r requirements.txt
```

On Raspberry Pi OS, install Picamera2 through the system package manager:

```bash
sudo apt install python3-picamera2
```

## Run the Vision Test

```bash
python tests/rectangle_detect.py
```

Press `q` to quit the camera preview windows.

## Extract Training Images From Video

Place source videos in object-specific folders under `data/raw/videos/`:

```text
data/raw/videos/washer/pan_01.mp4
data/raw/videos/washer/pan_02.mp4
```

Then run:

```bash
python data/extract_video_frames.py washer pan_01.mp4
```

This looks for the video in `data/raw/videos/washer/` and writes frames to:

```text
data/raw/photos/washer/pan_01.mp4/
```

Each video gets its own photo folder, so frames from different pans, positions, and lighting conditions stay grouped by source video.

By default, every frame is saved as a PNG. To save fewer frames, use `--frame-step`:

```bash
python data/extract_video_frames.py washer pan_01.mp4 --frame-step 10
```

To process every video under `data/raw/videos/`, use:

```bash
python data/extract_video_frames.py --all --frame-step 15
```

Batch mode skips videos that already have extracted photos in their matching output folder. For example, this video:

```text
data/raw/videos/washer/pan_01.mp4
```

is skipped if this folder already contains image files:

```text
data/raw/photos/washer/pan_01.mp4/
```

## Documentation Images

Project documentation pictures should go in date-labeled folders under `docs/images/`. Use a filename-safe date format with dashes:

```text
docs/images/2026-06-19/
  pi-cooling.jpg
  relay-test.jpg
```

Then reference them from Markdown like:

```markdown
![Raspberry Pi cooling setup](images/2026-06-19/pi-cooling.jpg)
```

Images under `docs/images/` are tracked with Git LFS and are excluded from the Raspberry Pi checkout.

## Dataset Storage

Everything under `data/` and `docs/images/` is tracked with Git LFS so the dataset and documentation images can be visible on GitHub without making every clone or pull download all of the large files immediately.

On development machines that should download dataset files normally, use:

```bash
git lfs install
git pull
```

On the Raspberry Pi, configure sparse checkout once so nothing under `data/` or `docs/images/` appears in the Pi working tree during normal pulls:

```bash
./scripts/setup_pi_sparse_checkout.sh
git pull
```

If the Raspberry Pi does not have that script yet, run the same setup manually before pulling:

```bash
git lfs install --local --skip-smudge
git sparse-checkout init --no-cone
git sparse-checkout set "/*" "!/data/" "!/docs/images/"
git pull
```

This keeps the dataset and documentation images visible on GitHub while keeping the Raspberry Pi checkout focused on runtime code and lightweight text docs.

If the Raspberry Pi needs dataset files later, disable sparse checkout first and then pull the specific LFS paths:

```bash
git sparse-checkout disable
git lfs pull --include="data/raw/photos/washer/**"
```

## Development Status

See `docs/progress_log.md` for dated project progress. Current work is focused on validating individual subsystems before integrating rover movement, vision, and object pickup behavior.

Near-term goals:

1. Tune camera detection under different lighting.
2. Add shape detection in addition to color detection.
3. Build a motor driver proof of concept.
4. Validate electromagnet activation with final hardware.
5. Define a simple rover state machine for search, approach, pickup, and release.
