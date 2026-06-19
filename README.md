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
docs/
  progress_log.md
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

## Development Status

See `docs/progress_log.md` for dated project progress. Current work is focused on validating individual subsystems before integrating rover movement, vision, and object pickup behavior.

Near-term goals:

1. Tune camera detection under different lighting.
2. Add shape detection in addition to color detection.
3. Build a motor driver proof of concept.
4. Validate electromagnet activation with final hardware.
5. Define a simple rover state machine for search, approach, pickup, and release.
