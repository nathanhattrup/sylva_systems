"""
Sylva — Tree Detection with SAHI (Sliced Aided Hyper Inference)
Uses local GroundingDINO 1.0 + SAM2 with supervision's InferenceSlicer.
Works on CPU.

WHAT THIS SCRIPT DOES:
1. Loads GroundingDINO (finds trees) and SAM2 (outlines them)
2. Runs detection on the full image (for comparison)
3. Runs SAHI: slices the image into overlapping patches, detects trees
   in each patch, merges results — catches small/distant trees
4. Runs SAM2 on the SAHI detections to get pixel-perfect masks
5. Saves annotated images and a JSON file of all detections
"""

# ─── Imports ────────────────────────────────────────────────────

import os                      # operating system utilities (file paths, etc.)
import cv2                     # OpenCV — reads/writes images, color conversion
import json                    # read/write JSON files (our detection output format)
import torch                   # PyTorch — the ML framework everything runs on
import numpy as np             # NumPy — array/matrix math, used for image data
import supervision as sv       # Supervision library — provides SAHI slicer, annotators, detection objects
from pathlib import Path       # nice way to handle file paths
from PIL import Image          # Pillow — image library, GroundingDINO expects PIL images

# ─── Configuration ──────────────────────────────────────────────
# CHANGE THESE for your images/setup

IMG_PATH = "notebooks/images/tree_farm_test1.jpg"        # path to the image you want to process — change this!
TEXT_PROMPT = "tree."                          # what to detect. MUST be lowercase and end with a period
OUTPUT_DIR = Path("outputs/sahi_tree_detection")  # where results get saved
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)     # create the output folder if it doesn't exist

# Detection thresholds — control sensitivity
BOX_THRESHOLD = 0.30    # minimum confidence to keep a detection (0-1). lower = more detections but more false positives
TEXT_THRESHOLD = 0.25   # minimum score for how well the detection matches the text prompt "tree."

# SAHI slicing parameters — control how the image gets chopped up
SLICE_WH = (640, 640)          # width x height of each slice in pixels. 640x640 is good for 1080p images
# OVERLAP_RATIO_WH = (0.2, 0.2) # 20% overlap between slices. prevents trees on slice borders from being cut in half

# SAM2 config — which segmentation model to use
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"   # tiny = fastest on CPU. swap to small for better quality
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"     # config file that matches the tiny checkpoint

# ─── Load GroundingDINO ─────────────────────────────────────────
# This is the model that FINDS trees. Given an image and the text "tree.",
# it outputs bounding boxes around everything it thinks is a tree.
# We're using the HuggingFace version so no local compilation is needed.

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

GD_MODEL_ID = "IDEA-Research/grounding-dino-tiny"      # which model to download from HuggingFace
print(f"Loading GroundingDINO from HuggingFace: {GD_MODEL_ID}")

gd_processor = AutoProcessor.from_pretrained(GD_MODEL_ID)   # handles image preprocessing + text tokenization
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_MODEL_ID)  # the actual neural network
gd_model.eval()   # put model in evaluation mode (disables training-specific behaviors like dropout)

print("GroundingDINO loaded.")

# ─── Load SAM2 ──────────────────────────────────────────────────
# This is the model that OUTLINES trees. Given an image and a bounding box,
# it produces a pixel-perfect segmentation mask (the pink/green overlay you saw).

from sam2.build_sam import build_sam2                    # function to construct the SAM2 model
from sam2.sam2_image_predictor import SAM2ImagePredictor # wrapper that handles single-image prediction

print(f"Loading SAM2: {SAM2_CHECKPOINT}")
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cpu")  # build model, force CPU mode
sam2_predictor = SAM2ImagePredictor(sam2_model)   # wrap it in the predictor interface
print("SAM2 loaded.")


def detect_trees_grounding_dino(image_rgb: np.ndarray) -> sv.Detections:
    """
    Run GroundingDINO on a single image (or image slice).
    This function gets called once for the full image, and then once per SAHI slice.

    Input: image as a numpy array (height x width x 3, RGB color order)
    Output: supervision Detections object containing bounding boxes + confidence scores
    """

    # GroundingDINO expects a PIL Image, not a numpy array, so convert it
    pil_image = Image.fromarray(image_rgb)

    # Preprocess: the processor resizes the image, normalizes pixel values,
    # and tokenizes the text prompt "tree." into numbers the model understands.
    # return_tensors="pt" means give us PyTorch tensors (the format the model expects)
    inputs = gd_processor(images=pil_image, text=TEXT_PROMPT, return_tensors="pt")

    # Run the model. torch.no_grad() tells PyTorch we're just doing inference,
    # not training — saves memory and speeds things up
    with torch.no_grad():
        outputs = gd_model(**inputs)   # **inputs unpacks the dict into keyword arguments

    # Post-process: convert raw model output into usable bounding boxes.
    # The model outputs thousands of candidate boxes — this filters to only the ones
    # that meet our confidence thresholds.
    # target_sizes tells it the original image dimensions so boxes are in pixel coordinates
    results = gd_processor.post_process_grounded_object_detection(
        outputs,
        threshold=BOX_THRESHOLD,
        target_sizes=[pil_image.size[::-1]]   # (height, width) — PIL gives (w,h), model wants (h,w)
    )[0]   # [0] because we're processing one image, not a batch

    boxes = results["boxes"].cpu().numpy()       # bounding boxes as numpy array, shape (N, 4) in [x1, y1, x2, y2] format
    scores = results["scores"].cpu().numpy()     # confidence score for each box, shape (N,)
    labels = results["labels"]                   # list of text labels (will all be "tree" for us)

    # If nothing was detected, return an empty Detections object
    if len(boxes) == 0:
        return sv.Detections.empty()

    # Package everything into a supervision Detections object.
    # This is the standard format that supervision's annotators, trackers, and slicers expect.
    return sv.Detections(
        xyxy=boxes,                                        # bounding boxes
        confidence=scores,                                 # confidence scores
        class_id=np.zeros(len(boxes), dtype=int),          # class ID for each detection (all 0 = "tree")
    )


def run_sam2_on_detections(image_rgb: np.ndarray, detections: sv.Detections) -> sv.Detections:
    """
    Given an image and bounding box detections, run SAM2 to get segmentation masks.
    This turns each bounding box into a pixel-perfect outline of the tree.

    Input: image (numpy array) + detections (with bounding boxes)
    Output: same detections object but now with masks attached
    """

    # Nothing to do if no trees were found
    if len(detections) == 0:
        return detections

    # Tell SAM2 which image we're working with. It pre-computes an internal
    # representation of the image (an "embedding") that makes mask prediction fast.
    sam2_predictor.set_image(image_rgb)

    masks_list = []
    for box in detections.xyxy:
        # For each bounding box, ask SAM2: "what's the object inside this box?"
        # It returns a mask (2D array of True/False for each pixel)
        masks, scores, _ = sam2_predictor.predict(
            box=box,               # the bounding box to segment
            multimask_output=False, # just give us one mask per box, not three options
        )
        masks_list.append(masks[0])   # masks[0] is the single mask (2D boolean array)

    # Attach all masks to the detections object as a numpy array
    detections.mask = np.array(masks_list).astype(bool) # shape will be (N, height, width) where N is number of detections
    return detections


# ─── Main: Load Image ───────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nLoading image: {IMG_PATH}")
    image_bgr = cv2.imread(IMG_PATH)   # OpenCV loads images in BGR color order (blue, green, red)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {IMG_PATH}")

    # Convert to RGB because GroundingDINO and SAM2 expect RGB, not BGR
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]   # get image dimensions (height, width)
    print(f"Image size: {w}x{h}")


    # ─── Method 1: Standard inference (no SAHI) ─────────────────────
    # Run detection on the full image at once. This is what you did in Test 2.
    # Small/distant trees will likely be missed.

    print("\n--- Running standard (full-image) inference ---")
    detections_standard = detect_trees_grounding_dino(image_rgb)
    print(f"Standard detection: {len(detections_standard)} trees found")


    # ─── Method 2: SAHI sliced inference ────────────────────────────
    # This is the magic. InferenceSlicer will:
    #   1. Chop the image into overlapping 640x640 patches
    #   2. Call detect_trees_grounding_dino() on each patch
    #   3. Map all the per-patch detections back to full-image coordinates
    #   4. Merge overlapping boxes (so a tree on a patch border isn't counted twice)

    print(f"\n--- Running SAHI sliced inference (slices={SLICE_WH}, overlap=128px) ---")

    slicer = sv.InferenceSlicer(
        callback=detect_trees_grounding_dino,
        slice_wh=SLICE_WH,
        overlap_wh=(128, 128),       # 20% of 640 = 128 pixels overlap
    )

    # Run it! This returns all detections across all slices, merged and deduplicated
    detections_sahi = slicer(image_rgb)
    print(f"SAHI detection: {len(detections_sahi)} trees found")


    # ─── Run SAM2 on SAHI detections ────────────────────────────────
    # Now that we have all the bounding boxes from SAHI, run SAM2 to get
    # pixel-perfect masks for each detected tree.

    print(f"\n--- Running SAM2 for segmentation masks on {len(detections_sahi)} detections ---")
    detections_sahi = run_sam2_on_detections(image_rgb, detections_sahi)
    print("Segmentation complete.")


    # ─── Save outputs ────────────────────────────────────────────────

    # --- Output 1: Image with bounding boxes drawn on it ---
    box_annotator = sv.BoxAnnotator(thickness=2)            # draws rectangle outlines
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)  # draws text labels

    # Create a label for each detection showing "tree 0.85" (name + confidence)
    labels = [f"tree {conf:.2f}" for conf in detections_sahi.confidence]

    annotated_boxes = image_bgr.copy()   # work on a copy so we don't modify the original
    annotated_boxes = box_annotator.annotate(annotated_boxes, detections_sahi)       # draw boxes
    annotated_boxes = label_annotator.annotate(annotated_boxes, detections_sahi, labels=labels)  # draw labels
    cv2.imwrite(str(OUTPUT_DIR / "sahi_boxes.jpg"), annotated_boxes)   # save to disk

    # --- Output 2: Image with segmentation masks overlaid ---
    if detections_sahi.mask is not None:
        mask_annotator = sv.MaskAnnotator(opacity=0.4)      # semi-transparent colored overlay
        annotated_masks = image_bgr.copy()
        annotated_masks = mask_annotator.annotate(annotated_masks, detections_sahi)  # draw masks
        annotated_masks = box_annotator.annotate(annotated_masks, detections_sahi)   # draw boxes on top
        cv2.imwrite(str(OUTPUT_DIR / "sahi_masks.jpg"), annotated_masks)

    # --- Output 3: Side-by-side comparison (standard vs SAHI) ---
    # This is the money shot — you'll see standard found 2 trees, SAHI found 15+
    annotated_standard = image_bgr.copy()
    annotated_standard = box_annotator.annotate(annotated_standard, detections_standard)  # standard results
    comparison = np.hstack([annotated_standard, annotated_boxes])  # stick them side by side horizontally
    cv2.imwrite(str(OUTPUT_DIR / "comparison_standard_vs_sahi.jpg"), comparison)

    # --- Output 4: JSON file with all detections ---
    # This is what the downstream pipeline (tracking, health assessment) will read
    results_json = []
    for i in range(len(detections_sahi)):
        entry = {
            "tree_id": i,                                          # simple index for now, tracking will assign real IDs later
            "bbox_xyxy": detections_sahi.xyxy[i].tolist(),         # bounding box as [x1, y1, x2, y2] list
            "confidence": float(detections_sahi.confidence[i]),    # detection confidence score
        }
        results_json.append(entry)

    # Write the JSON file
    with open(OUTPUT_DIR / "detections.json", "w") as f:
        json.dump(results_json, f, indent=2)   # indent=2 makes it human-readable

    # ─── Summary ────────────────────────────────────────────────────
    print(f"\n✓ Results saved to {OUTPUT_DIR}/")
    print(f"  - sahi_boxes.jpg        (bounding box annotations)")
    print(f"  - sahi_masks.jpg        (segmentation mask annotations)")
    print(f"  - comparison_standard_vs_sahi.jpg (side-by-side)")
    print(f"  - detections.json       (machine-readable results)")
    print(f"\nStandard found {len(detections_standard)} trees, SAHI found {len(detections_sahi)} trees")
