"""
Sylva — Batch process a video file: extract frames, detect trees with SAHI.
Saves per-frame detection JSONs for downstream tracking pipeline (Workstream B).

WHAT THIS SCRIPT DOES:
1. Opens a video file
2. Extracts every Nth frame (skipping frames to save time)
3. Runs SAHI sliced detection on each extracted frame
4. Saves three things per frame:
   - The raw frame as a .jpg
   - An annotated frame with bounding boxes drawn on it
   - A JSON file listing every detected tree's bounding box and confidence
"""

# ─── Imports ────────────────────────────────────────────────────

import os                      # operating system utilities
import cv2                     # OpenCV — reads video files, extracts frames, writes images
import json                    # read/write JSON files
import numpy as np             # array math
import supervision as sv       # SAHI slicer + annotators
from pathlib import Path       # clean file path handling

# ─── Import detection functions from the SAHI script ────────────
# We reuse the model loading and detection function from tree_detect_sahi.py
# so we don't duplicate code. When Python imports this, it will load
# GroundingDINO and SAM2 into memory (you'll see the loading messages).
from tree_detect_sahi import (
    detect_trees_grounding_dino,   # the function that runs GroundingDINO on one image/slice
    run_sam2_on_detections,        # the function that adds SAM2 masks to detections
    SLICE_WH,                     # slice size from the config (640, 640)
)

# ─── Configuration ──────────────────────────────────────────────
# CHANGE THESE for your video

VIDEO_PATH = "notebooks/videos/tree_vid_test2.mp4"   # path to your input video — CHANGE THIS
OUTPUT_DIR = Path("outputs/video_batch")       # where all results get saved
FRAME_INTERVAL = 10                            # process every Nth frame
                                               # 10 means: from a 30fps video, you get ~3 detections per second
                                               # higher number = faster but less coverage
                                               # lower number = more frames but slower
ENABLE_SAM2_MASKS = True   # Set to False to skip segmentation (boxes only, much faster)

# ─── Create output directories ──────────────────────────────────
# Three subfolders: raw frames, annotated frames, and detection JSONs

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)          # create the main output folder
(OUTPUT_DIR / "frames").mkdir(exist_ok=True)            # raw extracted video frames
(OUTPUT_DIR / "annotated").mkdir(exist_ok=True)         # frames with bounding boxes drawn on them
(OUTPUT_DIR / "detections").mkdir(exist_ok=True)        # per-frame JSON detection results

# ─── Set up SAHI slicer ────────────────────────────────────────
# Same slicer as in the single-image script.
# It will be called on every extracted frame.

slicer = sv.InferenceSlicer(
    callback=detect_trees_grounding_dino,   # function to run on each slice
    slice_wh=SLICE_WH,                      # size of each slice (640, 640)
    overlap_wh=(128, 128),                  # 128px overlap between slices (20% of 640)
)

# Set up the box annotator for drawing bounding boxes on frames
box_annotator = sv.BoxAnnotator(thickness=2)

# ─── Open the video file ────────────────────────────────────────

cap = cv2.VideoCapture(VIDEO_PATH)   # open the video file for reading

# Get video metadata
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # total number of frames in the video
fps = cap.get(cv2.CAP_PROP_FPS)                          # frames per second of the video

print(f"Video: {VIDEO_PATH}")
print(f"Total frames: {total_frames}, FPS: {fps:.1f}")
print(f"Processing every {FRAME_INTERVAL}th frame ({total_frames // FRAME_INTERVAL} frames)")

# ─── Process the video frame by frame ───────────────────────────

frame_idx = 0      # tracks which frame number we're on in the video
processed = 0      # counts how many frames we've actually processed

while cap.isOpened():                  # keep going as long as the video is open
    ret, frame_bgr = cap.read()        # read the next frame. ret=True if successful, frame_bgr is the image
    if not ret:                        # if ret is False, we've reached the end of the video
        break

    # Only process every Nth frame (skip the rest)
    if frame_idx % FRAME_INTERVAL == 0:

        # Convert BGR (OpenCV default) to RGB (what GroundingDINO expects)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Calculate the timestamp of this frame in the video
        timestamp_sec = frame_idx / fps   # frame number / frames per second = seconds

        # Detect trees
        detections = slicer(frame_rgb)

        # Optionally run SAM2 for segmentation masks
        if ENABLE_SAM2_MASKS:
            detections = run_sam2_on_detections(frame_rgb, detections)

        # Save raw frame
        cv2.imwrite(str(OUTPUT_DIR / "frames" / f"frame_{frame_idx:06d}.jpg"), frame_bgr)

        # Save annotated frame
        annotated = frame_bgr.copy()
        if ENABLE_SAM2_MASKS and detections.mask is not None:
            mask_annotator = sv.MaskAnnotator(opacity=0.4)
            annotated = mask_annotator.annotate(annotated, detections)
        annotated = box_annotator.annotate(annotated, detections)
        cv2.imwrite(str(OUTPUT_DIR / "annotated" / f"frame_{frame_idx:06d}.jpg"), annotated)

        # ── Save the detection results as JSON ──
        # This is the key output — Workstream B (tracking) will read these files
        results = {
            "frame_idx": frame_idx,                          # which frame in the video
            "timestamp_sec": round(timestamp_sec, 3),        # when in the video (seconds)
            "num_trees": len(detections),                    # how many trees were detected
            "trees": [                                       # list of all detected trees
                {
                    "bbox_xyxy": detections.xyxy[i].tolist(),        # bounding box [x1, y1, x2, y2]
                    "confidence": float(detections.confidence[i]),   # detection confidence 0-1
                }
                for i in range(len(detections))              # one entry per detected tree
            ],
        }

        # Write the JSON file for this frame
        with open(OUTPUT_DIR / "detections" / f"frame_{frame_idx:06d}.json", "w") as f:
            json.dump(results, f, indent=2)   # indent=2 for human readability

        # Update counter and print progress
        processed += 1
        print(f"  Frame {frame_idx}/{total_frames} ({timestamp_sec:.1f}s) — {len(detections)} trees")

    frame_idx += 1   # move to the next frame regardless of whether we processed this one

# ─── Cleanup ────────────────────────────────────────────────────

cap.release()   # close the video file
print(f"\n✓ Done. Processed {processed} frames. Results in {OUTPUT_DIR}/")

# ── Reassemble annotated frames into output video ──
print("\n--- Assembling annotated frames into output video ---")
annotated_dir = OUTPUT_DIR / "annotated"
frame_files = sorted(annotated_dir.glob("frame_*.jpg"))

if len(frame_files) > 0:
    # Read first frame to get dimensions
    sample = cv2.imread(str(frame_files[0]))
    h, w = sample.shape[:2]

    # Output FPS = original FPS / FRAME_INTERVAL (since we skipped frames)
    output_fps = fps / FRAME_INTERVAL

    out_path = str(OUTPUT_DIR / "annotated_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, output_fps, (w, h))

    for ff in frame_files:
        writer.write(cv2.imread(str(ff)))

    writer.release()
    print(f"✓ Output video saved: {out_path} ({len(frame_files)} frames @ {output_fps:.1f} fps)")
else:
    print("⚠ No annotated frames found — skipping video assembly.")