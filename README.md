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
  photos/
    washer/
  videos/
    washer/
docs/
  progress_log.md
src/
  extract_video_frames.py
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

Place source videos in object-specific folders under `data/videos/`:

```text
data/videos/washer/pan_01.mp4
data/videos/washer/pan_02.mp4
```

Then run:

```bash
python src/extract_video_frames.py washer/pan_01.mp4
```

This infers the object type from the folder name and writes frames to:

```text
data/photos/washer/
```

By default, every frame is saved as a PNG. To save fewer frames, use `--frame-step`:

```bash
python src/extract_video_frames.py washer/pan_01.mp4 --frame-step 10
```

## Dataset Storage

Everything under `data/` is tracked with Git LFS so the dataset can be visible on GitHub without making every clone or pull download all of the large files immediately.

On development machines that should download dataset files normally, use:

```bash
git lfs install
git pull
```

On the Raspberry Pi, configure sparse checkout once so nothing under `data/` appears in the Pi working tree during normal pulls:

```bash
./scripts/setup_pi_sparse_checkout.sh
git pull
```

If the Raspberry Pi does not have that script yet, run the same setup manually before pulling:

```bash
git lfs install --local --skip-smudge
git sparse-checkout init --no-cone
git sparse-checkout set "/*" "!/data/"
git pull
```

This keeps the dataset visible on GitHub while keeping the Raspberry Pi checkout focused on runtime code.

If the Raspberry Pi needs dataset files later, disable sparse checkout first and then pull the specific LFS paths:

```bash
git sparse-checkout disable
git lfs pull --include="data/photos/washer/**"
```

## Development Status

See `docs/progress_log.md` for dated project progress. Current work is focused on validating individual subsystems before integrating rover movement, vision, and object pickup behavior.

Near-term goals:

1. Tune camera detection under different lighting.
2. Add shape detection in addition to color detection.
3. Build a motor driver proof of concept.
4. Validate electromagnet activation with final hardware.
5. Define a simple rover state machine for search, approach, pickup, and release.
