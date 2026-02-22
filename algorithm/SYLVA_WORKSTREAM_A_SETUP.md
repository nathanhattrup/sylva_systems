# Sylva — Workstream A: Tree Detection & Segmentation

**Last updated:** 2026-02-21  
**Target machine:** Ubuntu/Linux, CPU-only (no NVIDIA GPU required)  
**Authors:** Sylva Team

---

## What This Workstream Does

Workstream A is the **eyes** of the Sylva pipeline. It takes raw drone footage of a tree farm and answers one question per frame: **where are the trees?**

The pipeline chains two models together:

1. **GroundingDINO** — an open-vocabulary object detector. You give it the text prompt `"tree."` and an image; it returns bounding boxes around every tree it finds. We load this through HuggingFace Transformers so there's zero local compilation required.

2. **SAM2 (Segment Anything Model 2)** — a segmentation model from Meta. It takes GroundingDINO's bounding boxes as prompts and produces pixel-perfect masks for each tree. This means downstream workstreams get exact tree outlines, not just rectangles.

Because drone images are high resolution and trees at the edge of frame can be small, we wrap the detection step in **SAHI (Sliced Aided Hyper Inference)** using the `supervision` library's `InferenceSlicer`. This slices the image into overlapping patches, runs GroundingDINO on each patch, then merges the results. The effect is dramatic — SAHI catches distant/small trees that full-image inference misses entirely.

### Pipeline Flow

```
Drone image/frame
        │
        ▼
┌────────────────┐
│  SAHI Slicer   │   supervision.InferenceSlicer
│  (overlapping  │   Slices image into NxN patches with overlap
│   patches)     │
└───────┬────────┘
        │  N overlapping patches
        ▼
┌────────────────┐
│ GroundingDINO  │   Text prompt: "tree."
│ (HuggingFace)  │   Returns bounding boxes per patch
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Merge + NMS   │   Deduplicates boxes from overlapping regions
└───────┬────────┘
        │  Final bounding boxes
        ▼
┌────────────────┐
│     SAM2       │   Box prompts → pixel-perfect masks
└───────┬────────┘
        │
        ▼
  Masks + Boxes + JSON output
```

### What the outputs look like

For each processed frame, the pipeline produces:

- **Annotated image with bounding boxes** — every detected tree outlined with a confidence score
- **Annotated image with segmentation masks** — colored overlays showing exact tree boundaries
- **JSON detection file** — machine-readable results with bounding box coordinates and confidence scores, used by Workstream B (tracking) downstream
- **Comparison image** — side-by-side of standard vs. SAHI detection so you can see the improvement

---

## Scripts Overview

You'll find two main scripts in `~/sylva/Grounded-SAM-2/`:

### `tree_detect_sahi.py` — Single Image Detection

This is the core detection script. Point it at any image and it will detect and segment every tree. Use this for testing, tuning thresholds, and validating on new imagery. It runs both a standard (full-image) pass and a SAHI (sliced) pass so you can compare results.

### `batch_video_trees.py` — Video Batch Processing

Wraps the same detection pipeline around a video file. It extracts every Nth frame (configurable via `FRAME_INTERVAL`), runs SAHI detection on each, saves per-frame JSONs for the tracking pipeline, and reassembles the annotated frames into an output video.

---

## Setup: Replicating the Environment

Since you'll have all the project files from the repo, setup is mostly about getting the Python environment right and downloading model weights.

### Prerequisites

```bash
# Confirm Python 3.10+
python3 --version    # needs to be 3.10, 3.11, or 3.12

# If you need to install Python 3.11:
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev python3-pip

# Install build tools
sudo apt install git build-essential wget
```

### Step 1 — Create project directory and virtual environment

```bash
mkdir -p ~/sylva
cd ~/sylva

python3 -m venv venv
source venv/bin/activate

# Confirm venv is active
which python    # should show ~/sylva/venv/bin/python
```

**Every time you open a new terminal:**

```bash
cd ~/sylva && source venv/bin/activate
```

### Step 2 — Install PyTorch (CPU-only)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output — `CUDA available: False` is correct and expected.

### Step 3 — Clone and set up Grounded-SAM-2

```bash
cd ~/sylva
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
```

### Step 4 — Download model checkpoints

```bash
# SAM2 checkpoints
cd checkpoints
bash download_ckpts.sh
cd ..

# GroundingDINO checkpoints (for the local demo; HF version downloads automatically)
cd gdino_checkpoints
bash download_ckpts.sh
cd ..
```

For CPU, we primarily use `sam2.1_hiera_tiny.pt` (fastest) or `sam2.1_hiera_small.pt` (better accuracy, still reasonable).

### Step 5 — Install SAM2

```bash
# From ~/sylva/Grounded-SAM-2
SAM2_BUILD_CUDA=0 pip install -e .
```

You'll see a warning about skipping post-processing — that's fine on CPU.

### Step 6 — Install GroundingDINO

We use the HuggingFace version to avoid C extension compilation headaches entirely:

```bash
pip install transformers --upgrade
```

That's it. The scripts load GroundingDINO via `AutoModelForZeroShotObjectDetection` from HuggingFace, which downloads weights automatically on first run (~1-2 GB, cached at `~/.cache/huggingface/hub/`).

If you want to also try the local demo scripts that came with the repo:

```bash
unset CUDA_HOME
pip install --no-build-isolation -e grounding_dino
```

If that fails with compilation errors, don't worry — the HuggingFace approach is what our scripts use.

### Step 7 — Install remaining dependencies

```bash
pip install opencv-python
pip install supervision
pip install pycocotools
pip install addict
pip install yapf
pip install timm
```

### Step 8 — Verify everything works

```bash
# Quick test with the bundled HuggingFace demo
cd ~/sylva/Grounded-SAM-2
python grounded_sam2_hf_model_demo.py --text-prompt "tree."

# Check outputs
ls outputs/grounded_sam2_hf_demo/
```

If you see annotated images, you're good. Now test the SAHI script:

```bash
python tree_detect_sahi.py
```

---

## Tuning Parameters

This is the section that matters most once you have things running. All of these are set at the top of `tree_detect_sahi.py` and `batch_video_trees.py`.

### Detection Thresholds

| Parameter | What it controls | Default | Guidance |
|---|---|---|---|
| `BOX_THRESHOLD` | Minimum confidence for a detection to count | `0.30` | Lower (0.20) catches more trees but introduces false positives. Higher (0.40) is stricter — fewer detections but more confident. Start at 0.30 and adjust based on your imagery. |
| `TEXT_THRESHOLD` | How well the detection must match the text prompt | `0.25` | Keep between 0.20–0.30. Rarely needs changing. |
| `TEXT_PROMPT` | What GroundingDINO looks for | `"tree."` | **Must be lowercase and end with a period.** Try variations based on your species: `"tree."`, `"tree canopy."`, `"conifer tree."`, `"evergreen tree."`, `"pine tree."`. Different prompts can significantly change detection behavior. |

### SAHI Slicing Parameters

| Parameter | What it controls | Default | Guidance |
|---|---|---|---|
| `SLICE_WH` | Pixel dimensions of each image slice | `(640, 640)` | For 1080p footage: `(640, 640)`. For 4K: `(1024, 1024)`. Smaller slices catch smaller trees but increase processing time proportionally. |
| `OVERLAP_RATIO_WH` | How much adjacent slices overlap | `(0.2, 0.2)` | 20% is a solid default. If you notice trees on slice boundaries getting missed or cut in half, bump to `(0.3, 0.3)`. |
| `iou_threshold` | IoU above which overlapping boxes from different slices get merged | `0.5` | Lower = more aggressive merging (fewer duplicate boxes). Higher = more conservative. 0.5 is standard. |

### CRITICAL: supervision InferenceSlicer Syntax

We ran into issues with the `supervision` library API. The correct constructor depends on your installed version. **Check your version first:**

```bash
python -c "import supervision; print(supervision.__version__)"
```

**For supervision >= 0.18.0** (current/recommended):

```python
slicer = sv.InferenceSlicer(
    callback=detect_trees_grounding_dino,
    slice_wh=SLICE_WH,
    overlap_ratio_wh=OVERLAP_RATIO_WH,
    overlap_filter_strategy=sv.OverlapFilter.NON_MAX_MERGE,
    iou_threshold=0.5,
)
```

**For older versions of supervision** you may see errors like `OverlapFilter has no attribute NON_MAX_MERGE` or unexpected keyword arguments. In that case, try:

```python
slicer = sv.InferenceSlicer(
    callback=detect_trees_grounding_dino,
    slice_wh=SLICE_WH,
    overlap_ratio_wh=OVERLAP_RATIO_WH,
    overlap_filter=sv.OverlapFilter.NON_MAX_MERGE,
    iou_threshold=0.5,
)
```

Or as a fallback with no merge strategy specified:

```python
slicer = sv.InferenceSlicer(
    callback=detect_trees_grounding_dino,
    slice_wh=SLICE_WH,
    overlap_ratio_wh=OVERLAP_RATIO_WH,
)
```

**If you're hitting errors**, the fastest fix is to pin the version: `pip install supervision==0.21.0` and use the first syntax above. The supervision library has changed its API across versions, so version mismatches are the most common source of syntax errors in this project.

### Video Processing Parameters

| Parameter | What it controls | Default | Guidance |
|---|---|---|---|
| `FRAME_INTERVAL` | Process every Nth frame | `10` | At 30fps source video, `10` gives ~3 processed frames/sec. For faster prototyping use `30` (1 frame/sec). For final output quality, use `5` or lower. |
| `VIDEO_PATH` | Input video file | — | Set this to your drone footage `.mp4` file. |

### SAM2 Model Selection

| Checkpoint | Speed (CPU) | Mask Quality | When to Use |
|---|---|---|---|
| `sam2.1_hiera_tiny.pt` | ~3-5 sec/image | Good | Default for prototyping and batch processing |
| `sam2.1_hiera_small.pt` | ~8-12 sec/image | Better | Final deliverable / presentation outputs |
| `sam2.1_hiera_large.pt` | ~20-30 sec/image | Best | Only if you get GPU access |

### CPU Performance Tips

- **Use `sam2.1_hiera_tiny.pt`** — roughly 4x faster than large with minimal quality loss
- **Increase `FRAME_INTERVAL`** — process every 30th frame instead of every 10th during development
- **Skip SAM2 during prototyping** — bounding boxes alone are sufficient for the tracking pipeline (Workstream B). Add masks later for the final deliverable
- **Resize before inference** — if your source is 4K, resize to 1080p before detection: `cv2.resize(img, (1920, 1080))` gives a major speedup
- **First run is slow** — HuggingFace downloads model weights on first execution (~1-2 GB). Subsequent runs use the local cache

---

## How Everything Connects to Downstream Workstreams

Workstream A's **output** is Workstream B's **input**. Specifically:

- The per-frame `detections/*.json` files contain bounding box coordinates and confidence scores for every tree detected in each frame
- Workstream B (ByteTrack) reads these JSONs, tracks trees across consecutive frames, and assigns persistent IDs
- Workstream C takes tracked trees and geolocates them using GPS telemetry from the drone
- Workstream D uses the segmentation masks and geolocated tree data to assess health (broken branches, disease, etc.)

The JSON format looks like this:

```json
{
  "frame_idx": 120,
  "timestamp_sec": 4.0,
  "num_trees": 7,
  "trees": [
    {
      "bbox_xyxy": [234.5, 112.3, 489.1, 567.8],
      "confidence": 0.87
    }
  ]
}
```

`bbox_xyxy` is `[x_min, y_min, x_max, y_max]` in pixel coordinates.

---

## File Structure After Setup

```
~/sylva/
├── venv/                          # Python virtual environment
└── Grounded-SAM-2/                # The cloned repo
    ├── checkpoints/               # SAM2 model weights
    │   ├── sam2.1_hiera_tiny.pt
    │   ├── sam2.1_hiera_small.pt
    │   └── ...
    ├── gdino_checkpoints/         # GroundingDINO weights (local demo)
    ├── grounding_dino/            # GroundingDINO source code
    ├── sam2/                      # SAM2 source code
    ├── tree_detect_sahi.py        # Single image detection + SAHI
    ├── batch_video_trees.py       # Video batch processing + output assembly
    └── outputs/
        ├── sahi_tree_detection/   # Single image outputs
        │   ├── sahi_boxes.jpg
        │   ├── sahi_masks.jpg
        │   ├── comparison_standard_vs_sahi.jpg
        │   └── detections.json
        └── video_batch/           # Video batch outputs
            ├── frames/            # Raw extracted frames
            ├── annotated/         # Annotated frames with bounding boxes
            ├── detections/        # Per-frame JSON files → feeds Workstream B
            └── annotated_output.mp4  # Reassembled annotated video
```

---

## Switching to GPU Later

If you get access to an NVIDIA GPU or use Google Colab, the changes are minimal:

```bash
# 1. Install GPU PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Reinstall SAM2 with CUDA extensions
pip uninstall SAM-2 -y
pip install -e .

# 3. In your scripts, change the SAM2 device:
#    device="cpu"  →  device="cuda"
```

Everything else is identical — just 10–50x faster.

---

## Milestone Checklist

- [ ] **M1:** Virtual environment created, `import torch` works
- [ ] **M2:** Grounded-SAM-2 repo cloned, checkpoints downloaded
- [ ] **M3:** `grounded_sam2_hf_model_demo.py` runs on sample image
- [ ] **M4:** `tree_detect_sahi.py` runs, comparison shows SAHI finds more trees
- [ ] **M5:** `batch_video_trees.py` processes a sample video, per-frame JSONs + output video generated
- [ ] **M6:** Tested on actual tree farm imagery
- [ ] **M7:** Tuned thresholds and SAHI params for target tree species and camera setup

---

## Troubleshooting

**"No module named 'groundingdino'"**  
The local GroundingDINO C extension failed to build. This is fine — our scripts use the HuggingFace version via `transformers`. No local build needed.

**"Skipping the post-processing step due to the error above"**  
SAM2's CUDA extension didn't build. Expected on CPU. Masks will be very slightly less clean but the difference is negligible.

**"CUDA not available" / "No CUDA GPUs are available"**  
Expected on CPU machines. The scripts use `device="cpu"` explicitly. If this appears as a hard error (not just a warning), make sure you installed CPU-only PyTorch, not a CUDA build.

**supervision syntax errors (OverlapFilter, iou_threshold, etc.)**  
See the "CRITICAL: supervision InferenceSlicer Syntax" section above. Pin `supervision==0.21.0` if in doubt.

**Out of memory (RAM)**  
SAM2 + GroundingDINO together use ~4-6 GB of RAM. If you have less than 8 GB total: close other applications, use `sam2.1_hiera_tiny.pt`, and resize images to 1080p.

**First run is extremely slow**  
HuggingFace downloads model weights on first run (~1-2 GB). Subsequent runs use the cache at `~/.cache/huggingface/hub/`.
