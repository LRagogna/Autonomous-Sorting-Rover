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
  labels/
    object_classes.txt           # the master list of objects: "id name"
  processed/
    detection/                   # the YOLO dataset (created by auto_label_frames.py)
  raw/
    videos/<object>/<clip>       # original recorded clips
    photos/<object>/<clip>/      # frames extracted from those clips
ml/
  train_yolo.py                  # train the detector
models/
  yolo_detector.pt               # the trained detector (created by train_yolo.py)
src/
  desktop_yolo_detector.py       # live webcam detector (draws the green boxes)
scripts/
  retrain.sh                     # one command: rebuild dataset + retrain
  run_desktop_detector.sh
  setup_pi_sparse_checkout.sh
tests/
  rectangle_detect.py            # Raspberry Pi green-color detection proof of concept
```

## Retrain In One Command

Whenever you add more training data, rebuild the dataset and retrain with a
single command:

```bash
./scripts/retrain.sh
```

This does everything for you: it **slices any new videos** in `data/raw/videos`
into photos (already-sliced videos are skipped), **rebuilds the dataset** from all
your source folders (old and new together), and **trains** the detector into
`models/yolo_detector.pt`. Pass training options straight through, for example
`./scripts/retrain.sh --epochs 60 --scale 0.9`, force the CPU with
`DEVICE=cpu ./scripts/retrain.sh`, or keep more frames per video with
`FRAME_STEP=10 ./scripts/retrain.sh`.

So if you record a new clip, you can just drop the video file into
`data/raw/videos/<object>/` and run `./scripts/retrain.sh` — no separate extract
step needed.

## The Vision Pipeline In Steps

The one command above is just these steps in a row. The detector learns from your
own photos:

```bash
# 1. Turn your videos into photos (skip if you already have photos)
python data/extract_video_frames.py --all --frame-step 15

# 2. Draw a box around the object in every photo and build the YOLO dataset
python data/auto_label_frames.py --overwrite

# 3. Train the detector
python ml/train_yolo.py
```

Then run the live detector on your computer:

```bash
./scripts/run_desktop_detector.sh
```

Hold a wrench or a bit in front of your webcam. When the model is confident, a green box with the object's name appears around it. Press `q` to quit.

## Setup Notes

Install Python dependencies (this includes `ultralytics` and PyTorch, which is a large download):

```bash
pip install -r requirements.txt
```

On Raspberry Pi OS, install Picamera2 through the system package manager:

```bash
sudo apt install python3-picamera2
```

## Step 1: Extract Training Photos From Video

Place source videos in object-specific folders under `data/raw/videos/`:

```text
data/raw/videos/wrench/clip_01.MOV
data/raw/videos/bit/clip_01.MOV
```

Then extract frames from every video at once:

```bash
python data/extract_video_frames.py --all --frame-step 15
```

At roughly 30 FPS, `--frame-step 15` saves about 2 frames per second. Photos are written to `data/raw/photos/<object>/<clip>/`. Batch mode skips videos that already have photos, so you can add new clips over time and rerun it.

You can also capture photos straight from your webcam:

```bash
python data/capture_webcam_training_images.py wrench --auto-save --max-images 80
```

## Step 2: Auto-Label The Photos (Build The YOLO Dataset)

A YOLO detector needs to know **where** the object is in each photo, not just what it is. Because the objects sit on a plain, high-contrast mat, `data/auto_label_frames.py` can find the object and draw the box for you:

```bash
python data/auto_label_frames.py --overwrite
```

This reads the class list in `data/labels/object_classes.txt` and writes a ready-to-train dataset to:

```text
data/processed/detection/
  dataset.yaml            # class names + folders, for YOLO
  images/train/           # most photos, used for learning
  images/val/             # a few photos, used for checking
  labels/train/           # one box file per training photo
  labels/val/             # one box file per check photo
  review/                 # the same photos with the box drawn on, for a human to check
```

**Always spot-check `data/processed/detection/review/`.** Open a handful of images and confirm the green boxes hug the object. If a class looks wrong, delete those photos or adjust the padding and re-run:

```bash
python data/auto_label_frames.py --overwrite --pad 0.1
```

The dataset is split by video clip, so near-identical frames from one clip never end up in both the training and check sets.

## Step 3: Train The Detector

```bash
python ml/train_yolo.py
```

This fine-tunes a small pretrained model (`yolov8n.pt`, "nano" = smallest/fastest). On a normal laptop CPU with a small dataset it finishes in a few minutes. The first run downloads the ~6 MB starter model, so you need internet once.

The trained detector is saved to:

```text
models/yolo_detector.pt
```

Useful options:

```bash
python ml/train_yolo.py --epochs 60     # train longer for tighter boxes
python ml/train_yolo.py --imgsz 512     # smaller pictures = faster, rougher
python ml/train_yolo.py --batch 4       # lower if you run out of memory
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

## Make It Work On Hands And Different Backgrounds

The automatic labeler only works when the object sits alone on a plain
background, so the trained model is best at that. If the detector struggles when
you **hold the object in your hand** or use a **busy or colored background**, the
fix is to add training pictures of those exact situations and box them by hand.

Full step-by-step instructions are in `data/hand_labeled/README.md`. In short:

1. Capture varied webcam pictures (hand-held, different backgrounds, lighting):

   ```bash
   python data/capture_webcam_training_images.py wrench --output-dir data/hand_labeled --auto-save --max-images 120
   python data/capture_webcam_training_images.py bit    --output-dir data/hand_labeled --auto-save --max-images 120
   ```

2. Draw boxes on them with LabelImg (a free desktop app), saving in **YOLO**
   format:

   ```bash
   pip install labelImg
   labelImg
   ```

3. Rebuild and retrain. The builder automatically folds in every hand-labeled
   picture alongside the plain-background ones:

   ```bash
   python data/auto_label_frames.py --overwrite
   python ml/train_yolo.py
   ```

Repeat the loop (capture more variety, label, retrain) until it recognizes the
objects in the conditions you care about.

## Add Internet Wrench Pictures (Better At A Distance)

Our own photos are all close-up, so the detector can be weak on a wrench that is
far away (small in the frame). Google's free **Open Images** dataset has thousands
of real wrench pictures at many distances and backgrounds, each already boxed.
Download a batch (wrench only — the bit is too unusual to find online):

```bash
pip install fiftyone
python data/fetch_wrench_internet.py --max-samples 600
```

This saves boxed wrench pictures into `data/hand_labeled/wrench_openimages/`, so
they fold into the dataset automatically. Then rebuild and retrain:

```bash
python data/auto_label_frames.py --overwrite
python ml/train_yolo.py
```

Training also randomly zooms pictures in and out (the `--scale` option, default
0.8) so the model learns each object at many sizes. To push size variety even
harder:

```bash
python ml/train_yolo.py --scale 0.9
```

## How To Add A New Object To Recognize

The pipeline is built to grow. To teach the rover a new object (for example, a washer):

1. Record short videos into `data/raw/videos/washer/`.
2. Extract photos: `python data/extract_video_frames.py --all --frame-step 15`
3. Add one line to `data/labels/object_classes.txt`, for example:

   ```text
   0 bit
   1 wrench
   2 washer
   ```

4. Rebuild the dataset: `python data/auto_label_frames.py --overwrite`
5. Retrain: `python ml/train_yolo.py`

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

## Dataset Storage

Everything under `data/` and `docs/images/` is tracked with Git LFS so the dataset and documentation images can be visible on GitHub without making every clone download all of the large files immediately. Small text files (the frame-extractor code, `.gitkeep` placeholders, YOLO label `.txt` files, and `dataset.yaml`) are kept as normal text so they read normally on GitHub.

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

This keeps the dataset visible on GitHub while keeping the Raspberry Pi checkout focused on runtime code and lightweight text docs.

## Development Status

See `docs/progress_log.md` for dated project progress. Current work is focused on validating individual subsystems before integrating rover movement, vision, and object pickup behavior.

Near-term goals:

1. Grow the detection dataset with more objects and lighting conditions.
2. Tune the YOLO detector for reliable boxes on the rover's real camera.
3. Run the detector on the Raspberry Pi (or export it for on-device speed).
4. Build a motor driver proof of concept.
5. Validate electromagnet activation with final hardware.
6. Define a simple rover state machine for search, approach, pickup, and release.
