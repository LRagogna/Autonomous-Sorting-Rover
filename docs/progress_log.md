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
  - `data/raw/clips/` for original recorded object videos.
  - `data/raw/photos/` for extracted raw image frames.
  - `data/labels/` for generated YOLO images, labels, dataset config, and review images.
- Added `data/extract_video_frames.py`.
- Designed the extractor so a video like:

```text
data/raw/clips/washer/pan_01.mp4
```

creates frames under:

```text
data/raw/photos/washer/pan_01__frame_000000.jpg
```

- Added single-video extraction:

```bash
python data/extract_video_frames.py washer pan_01.mp4 --frame-step 15
```

- Added batch mode to process every video under `data/raw/clips/`.
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
- Added more source media under `data/raw/clips/` so the object detector can learn from multiple object categories and camera angles.
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

## July 4, 2026

### Switched Vision From Classifier To YOLO Object Detector

- Replaced the old OpenCV SVM/HOG classifier and Coral pipeline with a proper YOLO object detector. The old approach only guessed one label for a whole crop and never drew a real box around the object; YOLO draws a tight box.
- Added `ultralytics` (which pulls in PyTorch) to `requirements.txt` for training and inference.

### New Three-Step Detection Pipeline

- `data/auto_label_frames.py`: reads `data/labels/object_classes.txt` and automatically draws a bounding box around the object in every extracted frame using OpenCV GrabCut (with a brightness/darkness fallback for the darker bit). It writes a YOLO dataset to `data/labels/` (images, labels, `dataset.yaml`) plus per-object `review/` folders of annotated previews for human spot-checking. Train/val is split by video clip so near-identical frames do not cross the split.
- `ml/train_yolo.py`: fine-tunes `yolov8n.pt` on the dataset and saves the result to `models/yolo_detector.pt`.
- `src/desktop_yolo_detector.py` + `scripts/run_desktop_detector.sh`: live webcam detector that draws a green box, label, and confidence around each detected object. This is the on-computer "simulator" for the rover's eyes.

### Data And Repo Cleanup

- Restructured `data/` around the YOLO dataset. Removed the obsolete classification crops (`data/processed/classification/`), external classifier references (`data/external/`), and `data/labels/object_classes.json` (the plain `object_classes.txt` is now the single source of truth for classes).
- Removed the obsolete classifier code and artifacts: `ml/train_classifier.py`, `ml/train_coral_classifier.py`, `ml/edgetpu_classifier.py`, `ml/detect_objects.py`, `ml/backtest_live_filter.py`, `src/desktop_realtime_classifier.py`, `src/pi_realtime_classifier.py`, `data/prepare_classification_dataset.py`, `data/generate_background_samples.py`, the classifier runner scripts, and the old `models/object_classifier.*` and test-output files.
- Kept the raw videos and photos as the dataset source of truth, plus `data/extract_video_frames.py` and `data/capture_webcam_training_images.py` for growing the dataset.
- The pipeline is designed to scale: adding a new object is drop clips → extract → add a class line → re-label → retrain.
- Updated `README.md` to document the new three-step workflow and how to add new objects. Kept `tests/rectangle_detect.py` as a separate color-detection proof of concept.

### Raspberry Pi Note

- The live detector currently targets a laptop/desktop webcam. Running YOLO on the Raspberry Pi (or exporting it for on-device speed) is a deliberate future step, not part of this desktop-simulator work.

### First Live Test And Robustness Follow-up

- Trained the first detector (40 epochs, Apple MPS): mAP50 ~0.85, and 12/12 held-out images boxed correctly.
- Live webcam test showed missed and flickering boxes. Root cause: the training data only shows the object alone on the dark mat, so the model does not generalize to a hand-held object or busy/colored backgrounds (domain gap).
- Quick tuning fix in `src/desktop_yolo_detector.py`: lowered default `--conf` to 0.25 and added box smoothing (`--smooth-frames`, default 6) so a box persists briefly after being lost instead of flickering.
- Current data rule: use only videos captured for this project. To improve robustness, record more short clips of the objects under the actual distances, lighting, angles, hands, and backgrounds the rover should handle.
- `ml/train_yolo.py`: added a `--scale` option (default 0.8) so training randomly zooms pictures in/out, teaching the model to see each object at many sizes (helps far-away/small detection).
- Rebuilding the dataset now means: add captured clips, run `./scripts/split_frames.sh`, run `./scripts/process.sh`, then train with `./scripts/train.sh`.

### One-Command Retraining

- Replaced the combined retraining command with three explicit steps:
  - `scripts/split_frames.sh`: slices `data/raw/clips/<object>/` videos into JPGs under `data/raw/photos/<object>/`.
  - `scripts/process.sh`: auto-labels captured frames and writes the YOLO dataset under `data/labels/<object>/`.
  - `scripts/train.sh`: trains YOLO from `data/labels/dataset.yaml`.
- Clarified the mental model for the user: the builder always rebuilds the dataset from captured frame folders, and `ml/train_yolo.py` always retrains from the pretrained base on the whole current dataset, so new captures are learned together with old captures.
- Advised on data quantity: roughly 150-250 varied, in-domain images per object, prioritizing the bit, with variety (hands, several backgrounds, near/far, angles, lighting) mattering more than raw count.

## July 5, 2026

### Physical Rover Design

- Created SolidWorks parts for mounting the electronics and camera on the rover.
- Designed a Raspberry Pi holder so the Pi has a planned physical location instead of being loose on the rover body.
- Designed a camera-position holder so the vision system can be mounted at a more controlled angle and height.
- Added the current rover CAD files to the project so the mechanical design can evolve alongside the electronics, code, and vision work.

### Training And Validation GUI

- Built a local workflow GUI to make the machine-learning loop easier to run and check.
- The GUI gives one place to split captured videos into frames, process/auto-label the frames, train the YOLO detector, and review the generated examples.
- Added a job log so long-running tasks like processing and training can be watched without guessing whether they are still running.
- Added review controls so rejected or questionable examples can be separated from the training set instead of quietly hurting model quality.
- Updated the training scripts to work smoothly with this GUI-driven workflow.

### Dataset Review Progress

- Continued cleaning the `bit` and `wrench` training data by moving questionable frames out of the train/val folders and into excluded review areas.
- Removed bad bit examples from the dataset so the detector is trained on cleaner object views.
- Cleaned up old review preview images after they had served their purpose, including the remaining `bit` review previews in the current working tree.
- Re-ran YOLO training outputs after the dataset and workflow cleanup so the detector reflects the cleaner training process.

## Week Of July 13, 2026

### GUI Workflow And Data Collection

- Spent most of the past week improving the training GUI workflow so the whole machine-learning loop is easier and faster to run from one place.
- Continued collecting data for the ML training by recording and adding more object clips, growing the dataset the detector learns from.

## July 20, 2026

### Wheel Distance Sensing Goal

- Set a new goal for the mobility system: make the rover drive a real, measured distance instead of just moving for a guessed amount of time. The target behavior is that a command like `forward 1 ft` actually travels one foot.
- Chose to measure distance with a wheel odometer built from a TCRT5000 infrared reflectance sensor. Strips of white tape are placed on the black rover wheel, and the sensor "sees" each white strip pass by as the wheel spins.
- Each white strip that passes counts as one "pulse." Counting pulses tells the rover how far the wheel has rolled, which is how far the rover has driven.

### IR Sensor Test Sketch

- Added `tests/ir_wheel_tape_pulse_test/ir_wheel_tape_pulse_test.ino`, a standalone Arduino test that drives the car forward a requested distance using tape pulses.
- Reused the same motor wiring as the working `src/serial_drive_turns.ino` (Elegoo Smart Robot Car V4.0), so the motor-driving part was already proven.
- Wired the TCRT5000 sensor to 5V, ground, and digital pin D10, and noted the module's trim potentiometer must be adjusted so the signal flips cleanly between white tape and black wheel.
- Added serial commands (`forward 1 ft`, `forward 36 in`, `f 120`, `stop`) at 9600 baud, plus a live telemetry line that prints the raw sensor value, whether it currently sees white, the pulse count, distance, speed, and RPM for debugging.

### Distance Math

- Measured the wheel at 2.6 inches across and placed 2 tape marks per revolution.
- Worked out that each pulse is about 4.084 inches of travel (wheel circumference divided by 2 marks).
- A `forward 1 ft` command therefore needs about 3 pulses (12 inches divided by 4.084).

### Tuning Problems Found And Fixed

- The car moved far too slowly at first. The drive speed was set low enough that the wheels nearly stalled, so raised the motor power (PWM) from 150 to 200.
- The sensor was counting short flashes of light (reflections and glints) as if they were real tape marks, which made the car stop after barely moving. Added a filter that only counts a pulse when the sensor sees white continuously for at least 15 milliseconds, since real tape gives a long steady signal and stray blips are brief.
- Sorted out a false alarm about the wheels spinning in opposite directions; the forward wiring was correct and matches the turns sketch.

### Decision: Trust The Tape, Not A Timer

- An earlier version also had a backup timer that could stop the car after a calculated amount of time. This timer was firing too early and cutting the drive short before the real tape pulses were counted.
- Decided distance should be measured purely by the white-tape pulses and removed the timer entirely, so the car drives until it has counted enough real pulses.
- Tradeoff accepted: if the sensor ever reads nothing, the car keeps driving until a `stop` command is sent, so the plan is to confirm on each run that the pulse count climbs as the wheel turns.
- Next step is a real-floor test of `forward 1 ft` to confirm the pulse count reaches 3 cleanly, then adjust the white-signal filter if any real pulses are missed.

## July 23-24, 2026

### Data Collection On The Pi Camera

- Spent these two days collecting training data using the Raspberry Pi camera, capturing object clips from the rover's own point of view instead of a phone camera.
- Recorded the rover driving up to each object, filming it from the low, on-floor angle the vision system will actually see, then driving away.
- Gathered Pi-camera footage across the object categories (bit, car, jenga, screwdriver, wrench) to close the domain gap between the training data and the rover's real-world view.

### 3D Printing Troubles

- Worked through a rough stretch on the 3D printer while trying to print the rover's electronics and camera mounts.
- Several prints failed outright: instead of a solid part, the printer laid down loose loops of plastic that piled up into a tangled "spaghetti" mess, and other prints came out covered in thin stray strands (stringing) with messy, poorly-formed top layers.
- Kept adjusting and retrying until a clean part finally came off the bed, shown on the right in the photo below next to the failed attempts on the left and middle.

![Failed 3D prints (spaghetti tangle and stringing) next to a clean printed enclosure](images/2026-07-24/80666396476__CB1A8F8C-2295-499B-8880-369EE9A2B66F.jpeg)
