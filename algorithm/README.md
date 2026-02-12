# Sylva — What We Need to Build and How

## How the System Works

```
DRONE (flies + records)              WORKSTATION (processes after landing)
┌──────────────────────┐            ┌─────────────────────────────────┐
│ Front camera ──► video│            │                                 │
│ Rear camera  ──► video│──► SSD ──►│  1. Detect trees in each frame  │
│ GPS/IMU ──► telemetry │            │  2. Track trees + map GPS locs  │
│ Obstacle avoidance    │            │  3. Flag broken branches        │
└──────────────────────┘            │  4. Classify diseases           │
                                    └────────────┬────────────────────┘
                                                 ▼
                                          Tree Health Map
                                     (ID, location, status)
```

No ML runs on the drone. It just records video and telemetry. All processing happens post-flight on a workstation.

---

## 1. Get the Dev Environment Running

- Clone the [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) repo
- Create a Python virtual environment and install PyTorch (CPU build is fine to start)
- Download SAM2 and GroundingDINO checkpoints
- Run `grounded_sam2_hf_model_demo.py` on a sample image and confirm it outputs annotated images with bounding boxes and masks
- Collect sample data to work with: drone footage of tree farms from YouTube, Google Images, or photos from your phone walking through a tree line

---

## 2. Build the Tree Detection Pipeline

This is the foundation everything else depends on.

- Use GroundingDINO (via HuggingFace `transformers`) for zero-shot detection with text prompt `tree.` — no training needed
- Use SAM2 to convert bounding boxes into pixel-perfect segmentation masks
- Add SAHI (Slicing Aided Hyper Inference) using the `supervision` library's `InferenceSlicer` — this slices high-res drone images into overlapping patches, runs detection on each, and merges results so small/distant trees aren't missed
- Tune detection thresholds and SAHI parameters (slice size, overlap ratio) on your sample imagery
- Build a batch processing script: video in → extract frames → detect trees per frame → save results as JSON files

**Output:** Per-frame JSON files containing bounding boxes, confidence scores, and optionally segmentation masks for every detected tree.

---

## 3. Add Multi-Object Tracking

Take the per-frame detections and link them into persistent tracks so each tree gets a consistent ID across frames.

- Use ByteTrack via the `supervision` library — it takes per-frame detections and assigns stable tracker IDs
- Verify that a tree entering frame N keeps the same ID through frame N+50+
- For cross-camera matching (front ↔ rear), start with temporal handoff: a tree exiting the front camera FOV at time T should enter the rear camera at approximately T + Δt based on drone speed and camera separation
- Save tracked results: `{track_id, list of (frame_idx, bbox)}`

---

## 4. Geolocate Each Tree

Convert pixel-space detections into real-world GPS coordinates using drone telemetry.

- For each tracked tree, you know its pixel position (u, v) in the frame and the timestamp
- From the flight controller telemetry log, get the drone's GPS coordinates, heading, pitch, and roll at that timestamp
- From camera intrinsics (focal length, sensor size) and drone orientation, compute the bearing from drone to tree
- From rangefinder readings or estimated distance (bounding box size as a proxy), compute distance
- Bearing + distance + drone GPS = tree GPS coordinate
- Tools: `pyproj` for coordinate transforms, OpenCV `solvePnP` for camera projection math

**Output:** A tree catalog — JSON or SQLite database mapping each `tree_id` to `{lat, lon, image_crops[], health_status}`.

---

## 5. Get Training Data for Health Assessment

This needs to happen in parallel with steps 2–4 since data collection takes time.

- Contact forestry department professors. Ask for:
  - A list of the 5–10 most common diseases/pests for the tree species at your target farm
  - Reference images of each disease at various stages
  - Access to any photo archives from field surveys
  - A few hours of their time to validate model predictions later
- For broken branches: collect 200+ images of broken/damaged branches and 200+ healthy trees from Google Images, Flickr, and your own photos
- For diseases: collect 100+ images per disease class plus a healthy class
- Supplement with public datasets: PlantVillage, iNaturalist (filter by species/region)
- Use [Roboflow](https://roboflow.com) (free tier) to upload, annotate (draw bounding boxes/labels), and manage your datasets

---

## 6. Train and Integrate Broken Branch Detection

- Annotate broken branch images in Roboflow — draw bounding boxes around the damage
- Export annotations in YOLOv8 format
- Train with Ultralytics: `yolo detect train data=broken_branches.yaml model=yolov8n.pt epochs=100`
- Use Google Colab for free GPU access if your machine is CPU-only
- Evaluate on a held-out test set — look at precision, recall, and confusion matrix
- Integrate into the pipeline: detection produces tree crops → broken branch model flags damaged trees

**Shortcut for early prototyping:** Before you have enough labeled data to train, use a vision-language model (Claude's vision or open-source LLaVA) to analyze tree crops with a prompt like "Is there visible damage such as broken branches in this image?" — zero training required, just slower.

---

## 7. Train and Integrate Disease Classification

- Finalize disease class list with forestry department input
- Organize images into folders by class (`healthy/`, `leaf_blight/`, `bark_beetle/`, etc.)
- Train a classifier:
  - **Option A (simplest):** YOLOv8 classification mode
  - **Option B:** Fine-tune EfficientNet or ResNet via HuggingFace `transformers` or `torchvision`
- Evaluate per-class accuracy — check what the model confuses with what
- Integrate into the pipeline: tree crop → disease classifier → per-tree health label in the catalog

---

## 8. Build the End-to-End Pipeline

Wire everything together into a single workflow:

```
Input: drone video file + GPS/IMU telemetry log

Step 1: Extract video frames (every Nth frame)
Step 2: Run SAHI + GroundingDINO + SAM2 on each frame → detections
Step 3: Run ByteTrack across frames → tracked trees with persistent IDs
Step 4: Match front + rear camera tracks → deduplicated tree list
Step 5: Project each tree to GPS coordinates using telemetry
Step 6: Crop each tree from its best frame
Step 7: Run broken branch detector on each crop
Step 8: Run disease classifier on each crop
Step 9: Write tree catalog (ID, lat, lon, species, health flags, image crops)

Output: tree catalog + optional map visualization
```

---

## 9. Test and Validate

- Run the full pipeline on the best drone footage you can get
- Spot-check detection: are trees being found reliably? Are false positives (bushes, poles) being filtered?
- Spot-check tracking: do tree IDs stay consistent? Does cross-camera matching work?
- Spot-check geolocation: if you have ground truth, how close are the estimated coordinates?
- Spot-check health assessment: do the models flag actual damage? What do they miss?
- Document failure modes and limitations — you'll need this for your design report

---

## Tools We're Using

| Need | Tool |
|---|---|
| Tree detection (no training) | Grounded-SAM-2 + SAHI via `supervision` |
| Segmentation masks | SAM2 |
| Object tracking | ByteTrack via `supervision` |
| Image annotation | Roboflow (free tier) |
| Training detectors/classifiers | Ultralytics YOLOv8 |
| Free GPU for training | Google Colab |
| AI pair programming | Claude Code |
| GPS/geospatial math | pyproj, GeoPandas |
| Video/image processing | OpenCV |
| Version control | Git + GitHub |
