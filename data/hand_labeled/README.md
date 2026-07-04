# Hand-labeled pictures

Put pictures here that you box **by hand** with a labeling tool. Use this for the
hard scenes the automatic labeler cannot handle:

- the object **held in your hand**
- **busy or colored backgrounds** (not the plain mat)
- unusual lighting, angles, or distances

The automatic labeler (`data/auto_label_frames.py`) only works when the object is
the one high-contrast lump on a plain background. A hand or a busy background
breaks it, so those pictures must be boxed by a person instead.

Any picture in this folder that has a matching YOLO `.txt` box file next to it is
automatically folded into the training dataset when you run
`python data/auto_label_frames.py --overwrite`.

## Step 1: Capture pictures

Use your webcam to grab pictures of one object at a time, into this folder:

```bash
python data/capture_webcam_training_images.py wrench --output-dir data/hand_labeled --auto-save --max-images 120
python data/capture_webcam_training_images.py bit    --output-dir data/hand_labeled --auto-save --max-images 120
```

While it records, move the object around: hold it in your hand, turn it, move it
close and far, and change the background and lighting. Variety is what makes the
model robust. Press `q` when done.

This writes pictures to:

```text
data/hand_labeled/wrench/webcam_<timestamp>/frame_000001.jpg
data/hand_labeled/bit/webcam_<timestamp>/frame_000001.jpg
```

## Step 2: Draw the boxes with LabelImg

Install LabelImg once:

```bash
pip install labelImg
```

Run it:

```bash
labelImg
```

In LabelImg:

1. Click **Open Dir** and choose the folder of pictures you captured
   (for example `data/hand_labeled/wrench/webcam_<timestamp>`).
2. On the left toolbar, click the format button until it says **YOLO**
   (not PascalVOC). This is important — we need YOLO format.
3. For each picture: press **W**, drag a tight box around the object, and type
   the class name exactly `bit` or `wrench`. Press **Ctrl+S** to save, then **D**
   for the next picture.

LabelImg saves one `.txt` box file next to each picture and a `classes.txt`
listing the names you used. Our builder reads that `classes.txt` and lines the
class numbers up automatically, so the order you add classes in does not matter —
just spell the names right.

Tip: only box pictures where you can clearly see the object. Skip blurry ones.

## Step 3: Rebuild the dataset and retrain

```bash
python data/auto_label_frames.py --overwrite
python ml/train_yolo.py
```

The printout will say how many hand-labeled pictures were added. Then test:

```bash
./scripts/run_desktop_detector.sh
```

Repeat this loop (capture more variety → label → retrain) whenever the detector
struggles with a new background or angle.

---

Note: `classes.txt` in this top folder is a copy of the class names in order,
handy to load into LabelImg as its predefined class list.
