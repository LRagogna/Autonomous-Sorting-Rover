# Progress Log

This log tracks major project decisions, hardware milestones, software work, and documentation updates for the autonomous sorting rover.

## May 24, 2026

### Project Definition

- Defined the main project goal: build an autonomous rover that can locate, collect, and sort user-prompted items.
- Discussed sorting methods, hardware constraints, budget, expected capabilities, and overall project scope.
- Identified the major rover subsystems:
  - Mobility
  - Computer vision
  - Object retrieval
  - Control and decision making

### Initial Architecture

- Chose a tank-style drive concept where wheels on each side move together.
- Planned to use an electromagnet for picking up metal debris.
- Planned to use cameras for object recognition based on shape and color.
- Planned to use a Raspberry Pi 4 for rover control and decision making.
- Identified the Coral TPU as a possible future accelerator for object recognition.
- Started sourcing parts.

## May 28, 2026

### Parts And Planning

- Created the final list of items to purchase.
- Purchased initial project parts.
- Created early sketches of the project architecture.
- Entered waiting period for parts to arrive from Amazon between late May and early June.

## June 14, 2026

### Electromagnet Proof Of Concept

- Began breadboarding a proof of concept for microcontroller-powered electromagnet activation.
- Used a relay module and LED as a safe stand-in for the electromagnet.
- Used an Arduino for the demo.
- Noted that the final implementation still needs a controller decision between Arduino and ESP32.
- Finalized a sketch of the electrical layout for the project.

## June 15-16, 2026

### Raspberry Pi Setup

- Received the Raspberry Pi 4.
- Set up the supporting hardware needed to use the Pi, including keyboard, monitor, and related accessories.
- Set up SSH access for the Raspberry Pi.
- Created basic programs while learning Raspberry Pi fundamentals.

## June 18, 2026

### Camera Vision Prototype

- Created a basic object recognition program using the Raspberry Pi camera and OpenCV.
- Used the OV5647 Raspberry Pi camera.
- Verified camera detection with `tests/rectangle_detect.py`.
- Built an early green-object detector that highlights detected green objects in the camera view.
- Cleaned up early documentation.
- Configured GitHub access on the Raspberry Pi.

## June 19, 2026

### Raspberry Pi Hardware

- Set up the Raspberry Pi cooling system.

### Repository Setup

- Added a Python virtual environment for local development.
- Added `requirements.txt` for Python dependencies.
- Added `.gitignore` for Python caches, virtual environments, local context, logs, editor files, and generated noise.
- Added `.gitattributes` for Git LFS dataset tracking.
- Expanded `README.md` with project overview, setup notes, dataset workflow, and Raspberry Pi data workflow.
- Added `docs/context.md` as an AI-only local context document.
- Kept `docs/context.md` ignored by git because it is only for local project memory.
- Added `docs/images/` for project documentation photos.
- Decided documentation image folders should use filesystem-safe date names like `docs/images/2026-06-19/`.

### Dataset Workflow

- Created the `data/` dataset structure:
  - `data/raw/videos/` for original recorded object videos.
  - `data/raw/photos/` for extracted raw image frames.
  - `data/processed/` for future cleaned or model-ready data.
  - `data/labels/` for future labels, annotations, or class maps.
- Added `data/extract_video_frames.py`.
- Designed the extractor so a video like:

```text
data/raw/videos/washer/pan_01.mp4
```

creates frames under:

```text
data/raw/photos/washer/pan_01.mp4/
```

- Added single-video extraction:

```bash
python data/extract_video_frames.py washer pan_01.mp4 --frame-step 15
```

- Added `--all` mode to process every video under `data/raw/videos/`.
- Added skip behavior so `--all` does not reprocess videos that already have extracted photos.
- Chose `--frame-step 15` as a good starting point for short 3-5 second panning videos.

### GitHub And Raspberry Pi Data Strategy

- Set up Git LFS so everything under `data/` can be visible on GitHub without storing large media directly in normal git objects.
- Added an LFS exception for `data/extract_video_frames.py` so the script stays visible as normal source code on GitHub.
- Added `.gitkeep` exceptions so placeholder files stay normal text files.
- Added `scripts/setup_pi_sparse_checkout.sh`.
- Decided the Raspberry Pi should use sparse checkout so normal pulls exclude the entire `data/` folder.
- Added `docs/images/` to the Raspberry Pi sparse checkout exclusions so documentation photos stay on GitHub but do not download to the Pi.
- Confirmed the Pi setup command pattern:

```bash
git lfs install --local --skip-smudge
git sparse-checkout init --no-cone
git sparse-checkout set '/*' '!/data/' '!/docs/images/'
git pull
```

### Documentation Style

- Established a project documentation rule: the GitHub repo should be understandable to a beginner.
- Added the rule to `docs/context.md`: a five-year-old should be able to go through the project and understand what each part is for.
- Added more thorough comments and top-of-file usage instructions to:
  - `data/extract_video_frames.py`
  - `tests/rectangle_detect.py`

## Week Of June 22, 2026

### ML Training Images

- Uploaded object images and videos for machine learning training data.
- Added more source media under `data/raw/videos/` so the object classifier can eventually learn from multiple object categories and camera angles.
- Continued building the dataset needed for the rover's vision system.

## June 29, 2026

### Wheel Power And Electrical Architecture

- Worked through how to power the rover wheels reliably.
- Figured out the electrical architecture for coordinating the Raspberry Pi and Arduino.
- Decided the Raspberry Pi should act as the high-level controller while the Arduino handles low-level motor commands over serial.

### Arduino Motor Control

- Added `src/serial_drive_turns.ino` for the Elegoo Smart Robot Car V4.0.
- The Arduino sketch accepts serial commands for forward, backward, stop, and calibrated in-place turns.
- Implemented non-blocking turn timing in the Arduino sketch so serial input can still be read while a turn is in progress.

### Raspberry Pi Main Program

- Added `src/main.py` as the main Raspberry Pi-side program entry point.
- The main program accepts compact movement sequences such as `FFBB`.
- `FFBB` now maps to two one-second forward movements followed by two one-second backward movements, with a stop command after each movement.
- Added `pyserial` to `requirements.txt` so the Raspberry Pi can send commands to the Arduino over USB serial.

### Raspberry Pi Data Safety

- Confirmed the Raspberry Pi should not pull videos or training data from the `data/` folder during normal operation.
- Updated `scripts/setup_pi_sparse_checkout.sh` to keep `data/` and `docs/images/` out of the Raspberry Pi checkout.
- Added a Git LFS fetch exclusion for `data/**` and `docs/images/**` as an extra guard against downloading large dataset files onto the Raspberry Pi.
- Updated `README.md` with the matching Raspberry Pi sparse checkout and Git LFS setup instructions.

### GitHub Workflow

- Resolved a non-fast-forward push issue by fetching the latest GitHub commits, rebasing the local work on top of them, and pushing the updated `main` branch.
- Confirmed the local branch and GitHub `main` branch were synced after the push.
