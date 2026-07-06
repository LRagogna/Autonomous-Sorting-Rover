ds# Autonomous-Sorting-Rover

Autonomous robotic rover that detects, collects, and sorts user-prompted items.

## Overview

This project is an early-stage autonomous sorting rover. The rover is being designed around a tank-style chassis, Raspberry Pi based control, camera-driven object recognition, and an electromagnet for collecting metal debris.

The current proof of concept focuses on computer vision. There are two vision pieces:

1. A simple color test that draws boxes around green objects (`tests/rectangle_detect.py`).
2. A **YOLO object detector** that learns your real objects (a wrench, a bit, and any objects you add later) and draws a **green box** around them in a live camera feed.

## Current Hardware Direction

- Raspberry Pi 4 for main control and decision making
- OV5647 Raspberry Pi camera for visual detection
- Tank-style drive layout with each side moving together
- Relay-controlled electromagnet for metal object pickup

## Repository Layout

```text
README.md
requirements.txt
data/
  extract_video_frames.py        # turn videos into photos
  auto_label_frames.py           # turn photos into a YOLO dataset (draws the boxes)
  capture_webcam_training_images.py
  raw/
    clips/<object>/<clip>        # original recorded videos
    photos/<object>/             # JPG frames created by split_frames.sh
  labels/
    object_classes.txt           # the master list of objects: "id name"
    dataset.yaml                 # YOLO dataset config created by process.sh
    <object>/                    # processed images, labels, and review images
ml/
  train_yolo.py                  # train the detector
models/
  yolo_detector.pt               # the trained detector (created by train_yolo.py)
src/
  desktop_yolo_detector.py       # live webcam detector (draws the green boxes)
scripts/
  run_workflow_gui.sh            # browser GUI for the training workflow
  split_frames.sh                # manual/debug command to split raw clips into JPG frames
  process.sh                     # split new clips and add new frames to the YOLO dataset
  train.sh                       # train YOLO from the processed dataset
  run_desktop_detector.sh
  setup_pi_sparse_checkout.sh
tests/
  rectangle_detect.py            # Raspberry Pi green-color detection proof of concept
```

## Training Workflow

The normal training workflow is:

```bash
./scripts/process.sh
./scripts/train.sh
```

`process.sh` is the one-button data prep step. It first slices any new clips in
`data/raw/clips/` into JPG frames under `data/raw/photos/<object>/`, then turns
new frames into YOLO labels, processed train/check images, and per-object review
folders under `data/labels/<object>/`.

After a clip has been added to the dataset, it is recorded in
`data/raw/photos/.processed_clips.json`. Future runs skip tracked clips, so old
videos are not split or added again. If you delete a bad review frame, that raw
frame is also recorded in `data/raw/photos/.deleted_frames.json` so it does not
come back later.

`train.sh` trains the detector from `data/labels/dataset.yaml` and writes the
final model to `models/yolo_detector.pt`. Pass training options straight
through, for example `./scripts/train.sh --epochs 60 --scale 0.9`, or force the
CPU with `DEVICE=cpu ./scripts/train.sh`.

To keep more frames per video when splitting:

```bash
FRAME_STEP=10 ./scripts/process.sh
```

You can also manage the same workflow from a local browser GUI:

```bash
./scripts/run_workflow_gui.sh
```

The GUI lets you drag and drop multiple videos into `data/raw/clips/<object>/`,
run Process Data and Train Data from buttons, and review/delete generated
frames. When you delete a review frame, the matching raw frame and YOLO
label/image are also removed, and the deletion is recorded so that frame does
not come back on the next processing run.

## The Vision Pipeline In Steps

The detector learns from your own photos through this short workflow:

```bash
# 1. Split new clips, draw boxes, and add them to the YOLO dataset
./scripts/process.sh

# 2. Train the detector
./scripts/train.sh
```

Then run the live detector on your computer:

```bash
./scripts/run_desktop_detector.sh
```

Hold a wrench or a bit in front of your webcam. When the model is confident, a green box with the object's name appears around it. Press `q` to quit.

## Setup Notes

On a development computer, install the desktop/ML Python dependencies with:

```bash
pip install -r requirements.txt
```

Do not run that file on the Raspberry Pi. The Pi runtime dependency file is
small:

```bash
pip install -r requirements-pi.txt
```

On Raspberry Pi OS, install Picamera2 through the system package manager:

```bash
sudo apt install python3-picamera2
```

## Raspberry Pi Runtime Checkout

The Raspberry Pi should be a runtime checkout, not a development or training
checkout. Today, ML training does **not** run on the Pi: videos, datasets,
auto-labeling, YOLO training, desktop detector experiments, model run outputs,
CAD files, and project docs stay on a development machine. The Pi currently
needs only the control entry point, Arduino sketch, ROS 2 control source, the
Raspberry Pi hardware tests, and small setup/config files used to operate or
update those pieces.

Configure the Pi checkout once:

```bash
./scripts/setup_pi_sparse_checkout.sh
git pull
```

That script allowlists only:

```text
.gitattributes
.gitignore
README.md
requirements-pi.txt
scripts/setup_pi_sparse_checkout.sh
src/main.py
src/serial_drive_turns.ino
tests/
ros2_ws/README.md
ros2_ws/rover_env.sh
ros2_ws/src/
```

It also tells Git LFS not to fetch `data/`, `docs/`, `solidworks/`, `ml/`,
`models/`, the computer `requirements.txt`, or the base YOLO weights during
normal Pi pulls.

## Step 1: Process Training Clips Into Reviewable Data

Place source videos in object-specific folders under `data/raw/clips/`:

```text
data/raw/clips/wrench/clip_01.MOV
data/raw/clips/bit/clip_01.MOV
```

Then process the data:

```bash
./scripts/process.sh
```

This first splits new clips into JPG frames, then creates YOLO labels and review
images. At roughly 30 FPS, the default `FRAME_STEP=15` saves about 2 frames per
second. Photos are written as JPG frames to `data/raw/photos/<object>/`, with
filenames like `clip_01__frame_000000.jpg`.

After a clip has been added to `data/labels`, it is tracked in
`data/raw/photos/.processed_clips.json`, so future Process Data runs skip that
clip instead of splitting or adding it again.

You can also capture photos straight from your webcam:

```bash
python data/capture_webcam_training_images.py wrench --auto-save --max-images 80
```

## Step 2: Review Generated Labels

A YOLO detector needs to know **where** the object is in each photo, not just what it is. Because the objects sit on a plain, high-contrast mat, `data/auto_label_frames.py` can find the object and draw the box for you:

This reads the class list in `data/labels/object_classes.txt` and writes a ready-to-train dataset to:

```text
data/labels/
  object_classes.txt
  dataset.yaml
  wrench/
    images/train/
    images/val/
    labels/train/
    labels/val/
    review/
  bit/
    images/train/
    images/val/
    labels/train/
    labels/val/
    review/
```

**Always spot-check each object's `review/` folder.** Open a handful of images
and confirm the green boxes hug the object. If a class looks wrong, delete those
source photos or adjust the padding and re-run:

```bash
./scripts/process.sh --pad 0.1
```

The dataset is split by video clip, so near-identical frames from one clip never end up in both the training and check sets.

## Step 3: Train The Detector

```bash
./scripts/train.sh
```

This fine-tunes a small pretrained model (`yolov8n.pt`, "nano" = smallest/fastest). On a normal laptop CPU with a small dataset it finishes in a few minutes. The first run downloads the ~6 MB starter model, so you need internet once.

The trained detector is saved to:

```text
models/yolo_detector.pt
```

Useful options:

```bash
./scripts/train.sh --epochs 60     # train longer for tighter boxes
./scripts/train.sh --imgsz 512     # smaller pictures = faster, rougher
./scripts/train.sh --batch 4       # lower if you run out of memory
```

## Step 4: Run The Live Detector On Your Computer

```bash
./scripts/run_desktop_detector.sh
```

This opens your computer's default webcam, runs the trained detector on each frame, and draws a green box with the object's name and confidence. Press `q` to quit.

Useful options:

```bash
./scripts/run_desktop_detector.sh --camera-index 1   # use a second camera
./scripts/run_desktop_detector.sh --conf 0.15        # show more (shakier) boxes
./scripts/run_desktop_detector.sh --conf 0.5         # only very confident boxes
./scripts/run_desktop_detector.sh --smooth-frames 0  # turn off box smoothing
./scripts/run_desktop_detector.sh --headless         # no window; print detections
```

The detector lowers shaky boxes with two helpers: a confidence cutoff (`--conf`)
and box smoothing, which keeps a box on screen for a few frames after it is lost
so it does not flicker.

On macOS, if OpenCV says camera access was denied, allow your terminal app under:

```text
System Settings > Privacy & Security > Camera
```

## Improve The Model With Your Own Captures

Only use videos you capture yourself for this project. If the detector struggles
with distance, lighting, angles, hands, or backgrounds, record more short clips
that show those exact conditions and place them in:

```text
data/raw/clips/<object>/
```

Then rebuild the training data and train again:

```bash
./scripts/split_frames.sh
./scripts/process.sh
./scripts/train.sh
```

Training also randomly zooms pictures in and out (the `--scale` option, default
0.8) so the model learns each object at many sizes. To push size variety even
harder:

```bash
./scripts/train.sh --scale 0.9
```

## How To Add A New Object To Recognize

The pipeline is built to grow. To teach the rover a new object (for example, a washer):

1. Record short videos into `data/raw/clips/washer/`.
2. Split photos: `./scripts/split_frames.sh`
3. Add one line to `data/labels/object_classes.txt`, for example:

   ```text
   0 bit
   1 wrench
   2 washer
   ```

4. Process the photos and labels: `./scripts/process.sh`
5. Train: `./scripts/train.sh`

The new object will now get its own green box in the live detector.

## Color Detection Proof Of Concept

`tests/rectangle_detect.py` is a separate, simpler Raspberry Pi camera test. It captures frames, converts them to HSV color space, thresholds for green pixels, cleans the mask, finds contours, and draws boxes around green objects. It is intended to run on a Raspberry Pi with the camera connected:

```bash
python tests/rectangle_detect.py
```

Press `q` to quit.

## Documentation Images

Project documentation pictures go in date-labeled folders under `docs/images/`. Use a filename-safe date format with dashes:

```text
docs/images/2026-06-19/pi-cooling.jpg
```

Then reference them from Markdown like:

```markdown
![Raspberry Pi cooling setup](images/2026-06-19/pi-cooling.jpg)
```

Images under `docs/images/` are tracked with Git LFS and are excluded from the Raspberry Pi checkout.

## Dataset, CAD, And Documentation Storage

Everything under `data/`, `docs/images/`, and `solidworks/`, plus the computer
`requirements.txt`, is tracked with Git LFS so datasets, documentation media,
CAD/export files, and desktop dependency lists can stay out of the Pi runtime
checkout. Common SolidWorks/CAD extensions (`.sldprt`, `.sldasm`, `.slddrw`,
`.stl`, `.step`, and `.stp`, including uppercase variants) are also tracked
with LFS wherever they are added. Small text files (the frame-extractor code,
`.gitkeep` placeholders, YOLO label `.txt` files, `dataset.yaml`, and
`requirements-pi.txt`) are kept as normal text so they read normally on GitHub.

On development machines that should download dataset files normally, use:

```bash
git lfs install
git pull
```

On the Raspberry Pi, configure sparse checkout once so only runtime files appear
in the Pi working tree during normal pulls:

```bash
./scripts/setup_pi_sparse_checkout.sh
git pull
```

If the Raspberry Pi does not have that script yet, run the same setup manually
before pulling:

```bash
git lfs install --local --skip-smudge
git config --local lfs.fetchexclude "data/**,docs/**,solidworks/**,ml/**,models/**,requirements.txt,yolov8n.pt"
git sparse-checkout init --no-cone
git sparse-checkout set \
  "/.gitattributes" \
  "/.gitignore" \
  "/README.md" \
  "/requirements-pi.txt" \
  "/scripts/setup_pi_sparse_checkout.sh" \
  "/src/main.py" \
  "/src/serial_drive_turns.ino" \
  "/tests/" \
  "/ros2_ws/README.md" \
  "/ros2_ws/rover_env.sh" \
  "/ros2_ws/src/"
git pull
```

This keeps the dataset, documentation media, CAD files, ML training workspace,
model run outputs, and desktop tools visible in the repository while keeping
the Raspberry Pi checkout focused on operational rover code and hardware tests.

## Development Status

See `docs/progress_log.md` for dated project progress. Current work is focused on validating individual subsystems before integrating rover movement, vision, and object pickup behavior.

Near-term goals:

1. Grow the detection dataset with more objects and lighting conditions.
2. Tune the YOLO detector for reliable boxes on the rover's real camera.
3. Run the detector on the Raspberry Pi (or export it for on-device speed).
4. Build a motor driver proof of concept.
5. Validate electromagnet activation with final hardware.
6. Define a simple rover state machine for search, approach, pickup, and release.
