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
- Coral Edge TPU for accelerated object classification on the Pi

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

For the live classifier on Raspberry Pi, the system OpenCV package is usually
the easiest path:

```bash
sudo apt install python3-picamera2 python3-opencv python3-numpy
```

For Coral TPU inference, also install the Coral runtime/PyCoral package on the
Pi:

```bash
sudo apt install python3-pycoral
```

The Coral model is trained/exported with TensorFlow on a development machine.
TensorFlow is intentionally not in `requirements.txt` because it is too large
for the normal Raspberry Pi runtime setup:

```bash
python3 -m pip install tensorflow
```

## Run the Vision Test

```bash
python tests/rectangle_detect.py
```

Press `q` to quit the camera preview windows.

## Run The Live Object Classifier On Raspberry Pi

The default Pi classifier command is Coral-required:

```bash
./scripts/run_pi_classifier.sh
```

It expects an Edge-TPU-compiled TensorFlow Lite model at:

```text
models/object_classifier_edgetpu.tflite
models/object_classifier_labels.txt
```

If that compiled `.tflite` file is missing, the command fails instead of
silently using the CPU. This is intentional, so Coral mode really means Coral
mode.

The preview window shows the camera feed, finds likely object regions, sends
those cropped regions through the classifier, and draws a green box with the
predicted label around each confirmed object. It now rejects the `background`
class, rejects low-certainty crop votes, rejects boxes touching the camera edge,
and waits for the same label across consecutive classification passes before
drawing a green box. Press `q` in the preview window to quit.

If you are connected over SSH or do not have a desktop preview:

```bash
./scripts/run_pi_classifier.sh --headless
```

Headless mode prints the detected label whenever it changes.

The current OpenCV SVM model can still run as a CPU fallback:

```bash
./scripts/run_pi_classifier_cpu.sh
```

CPU fallback uses:

```text
models/object_classifier.yml
models/object_classifier_metadata.json
```

The optional wrench override files are still available, but CPU live mode does
not use them by default because the current HOG-based multiclass model tested
better without the override. Use `--enable-wrench-override` only for comparison.

Useful runtime options:

```bash
./scripts/run_pi_classifier.sh --width 640 --height 480
./scripts/run_pi_classifier.sh --confirm-frames 3
./scripts/run_pi_classifier.sh --min-vote-fraction 0.67
./scripts/run_pi_classifier.sh --show-rejected
./scripts/run_pi_classifier.sh --box-padding 24
./scripts/run_pi_classifier_cpu.sh --enable-wrench-override
```

By default, live mode uses `--detection-mode objects`. OpenCV only does the
lightweight candidate-box proposal work; the actual classification inference is
done by the Coral Edge TPU when you use `./scripts/run_pi_classifier.sh`.
Use `--detection-mode frame` only if you want the older whole-frame behavior for
debugging.

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

## Train And Validate A Starter Classifier

The first training pipeline lives in:

```bash
python ml/train_classifier.py
```

It trains a small OpenCV SVM classifier and validates it before saving model files. This is meant as a lightweight starter model while the dataset is still small.

The script can use either processed classification folders:

```text
data/processed/classification/
  train/
    washer/
      image_001.png
    bolt/
      image_001.png
  val/
    washer/
      image_002.png
    bolt/
      image_002.png
```

or raw extracted frames:

```text
data/raw/photos/
  washer/
    pan_01.mp4/
      frame_000000.png
  bolt/
    pan_01.mp4/
      frame_000000.png
```

When using raw extracted frames, the trainer splits by source video folder instead of randomly mixing individual frames. That keeps similar frames from the same video from appearing in both training and validation.

Run with raw extracted frames:

```bash
python ml/train_classifier.py --dataset raw
```

Run with a prepared train/validation dataset:

```bash
python ml/train_classifier.py --dataset processed
```

The trainer needs at least two object classes and validation images for each class. If no usable dataset exists yet, it exits with setup instructions instead of creating a bad model.

Successful training writes:

```text
models/object_classifier.yml
models/object_classifier_metadata.json
models/object_classifier_metrics.json
models/object_classifier_validation.csv
```

## Prepare A Cleaner Classification Dataset

Raw extracted frames can include a lot of background and repeated near-identical
frames. To build a cleaner train/validation dataset from all extracted frames,
run:

```bash
python data/prepare_classification_dataset.py --overwrite
```

This writes balanced cropped images to:

```text
data/processed/classification/
  train/
  val/
```

Then train from the processed dataset:

```bash
python ml/train_classifier.py --dataset processed
```

To reduce false detections on the camera background, generate a `background`
class from raw frame corners before training:

```bash
python data/generate_background_samples.py --overwrite
python ml/train_classifier.py --dataset processed
```

Backtest the live filter against raw camera frames and generated background
crops:

```bash
python ml/backtest_live_filter.py --positive-dir data/raw/photos
```

## Build The Coral TPU Classifier

The OpenCV `.yml` model cannot run on the Coral TPU. To make the classifier file
that actually runs the heavy neural-network inference on Coral, train and export
the TensorFlow Lite model:

```bash
python ml/train_coral_classifier.py --epochs 30
```

This writes:

```text
models/object_classifier_coral.keras
models/object_classifier.tflite
models/object_classifier_edgetpu.tflite
models/object_classifier_labels.txt
```

`models/object_classifier_edgetpu.tflite` is the file used by:

```bash
./scripts/run_pi_classifier.sh
```

If `edgetpu_compiler` is missing, install it on a Linux development machine or
the Raspberry Pi:

```bash
sudo apt install edgetpu-compiler
```

Optional representative examples can go in:

```text
data/external/classification/<label>/
```

For example, flat product-style wrench references can go in:

```text
data/external/classification/wrench/
```

After training the main classifier, train the targeted wrench override model:

```bash
python ml/train_wrench_override.py
```

This trains a small binary wrench-vs-not-wrench model. During detection, it can
correct common mistakes where a wrench is first classified as `wire` or
`steel_tape`.

## Detect Objects In Pictures

After training, classify one picture:

```bash
python ml/detect_objects.py path/to/image.jpg
```

Classify a folder of pictures:

```bash
python ml/detect_objects.py path/to/folder --recursive
```

The detector writes:

```text
models/object_detections.csv
models/object_detections_summary.json
models/annotated_detections/
```

This starter detector predicts one object label for the whole picture. It does
not yet draw a tight box around the object. If `models/wrench_override.yml`
exists, the detector uses it automatically. To compare the raw multiclass
prediction without that override, run:

```bash
python ml/detect_objects.py path/to/image.jpg --disable-wrench-override
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
git config --local lfs.fetchexclude "data/**,docs/images/**"
git sparse-checkout init --no-cone
git sparse-checkout set "/*" "!/data/" "!/docs/images/"
git pull
```

This keeps the dataset and documentation images visible on GitHub while keeping the Raspberry Pi checkout focused on runtime code and lightweight text docs. The LFS fetch exclusion also prevents the Pi from downloading files under `data/` during normal LFS operations.

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
