"""
Sylva — DeepForest Tree Analysis Pipeline
==========================================
Takes a GeoTIFF orthomosaic, detects individual tree crowns using DeepForest,
and extracts per-tree metrics: canopy diameter, coloration (RGB stats, greenness
indices), geo-location, and confidence scores.

This is the proof-of-concept demo: overhead drone imagery → tree-level analytics.

Usage:
    python sylva_deepforest_pipeline.py /path/to/orthomosaic.tif

    # Or with options:
    python sylva_deepforest_pipeline.py /path/to/orthomosaic.tif \
        --patch-size 400 \
        --patch-overlap 0.25 \
        --score-thresh 0.3 \
        --tile-size 2048 \
        --max-tiles 0 \
        --output-dir outputs/sylva_analysis

Dependencies:
    pip install deepforest rasterio numpy opencv-python pandas matplotlib
"""

import sys
import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── Suppress noisy warnings from DeepForest / PyTorch ──────────────────────
warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION (can be overridden via CLI args)
# ═══════════════════════════════════════════════════════════════════════════

# DeepForest prediction parameters
# The pretrained model was trained on 400×400 px crops at ~10 cm GSD.
# patch_size and patch_overlap control DeepForest's internal sliding window.
DEFAULT_PATCH_SIZE = 500        # pixels — matches DeepForest training resolution
DEFAULT_PATCH_OVERLAP = 0.25    # 25% overlap between patches
DEFAULT_SCORE_THRESH = 0.4      # minimum detection confidence (0–1)

# Tiling parameters for large orthomosaics
# We break the full raster into manageable tiles, run DeepForest on each,
# and stitch detections back into global pixel / geo coordinates.
DEFAULT_TILE_SIZE = 2048        # pixels per tile side
DEFAULT_TILE_OVERLAP = 128      # pixel overlap between tiles (catches edge trees)

# Output directory
DEFAULT_OUTPUT_DIR = Path("outputs/sylva_analysis")


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: READ GSD (GROUND SAMPLING DISTANCE)
# ═══════════════════════════════════════════════════════════════════════════

def get_gsd_meters(src):
    """
    Compute the ground sampling distance in meters from a rasterio dataset.
    Returns (gsd_x_m, gsd_y_m) — the real-world size of one pixel in meters.
    If the CRS is geographic (lat/lon), we approximate using 111,320 m/degree.
    If there's no CRS at all, returns (None, None).
    """
    if src.crs is None:
        # No CRS — can't compute GSD
        return None, None
    elif src.crs.is_projected:
        # Projected CRS: resolution is already in meters (or feet, but usually meters)
        return abs(src.res[0]), abs(src.res[1])
    elif src.crs.is_geographic:
        # Geographic CRS: resolution is in degrees — rough conversion
        lat_center = (src.bounds.top + src.bounds.bottom) / 2.0
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(np.radians(lat_center))
        return abs(src.res[0]) * m_per_deg_lon, abs(src.res[1]) * m_per_deg_lat
    else:
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: PIXEL → GEO COORDINATE CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def pixel_to_geo(col, row, transform):
    """
    Convert pixel coordinates (col, row) to geographic coordinates (x, y)
    using the rasterio affine transform.
    """
    x = transform.c + col * transform.a + row * transform.b
    y = transform.f + col * transform.d + row * transform.e
    return x, y


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: COLOR / HEALTH METRICS PER TREE CROWN
# ═══════════════════════════════════════════════════════════════════════════

def compute_crown_color_metrics(tile_rgb, xmin, ymin, xmax, ymax):
    """
    Given an RGB tile and a bounding box (in tile-local pixel coords),
    compute color-based metrics for the tree crown.

    Returns a dict with:
        - mean_r, mean_g, mean_b: average channel values (0–255)
        - std_r, std_g, std_b: channel standard deviations
        - greenness_exg: Excess Green Index = 2*G - R - B (higher = greener)
        - vari: Visible Atmospherically Resistant Index = (G-R)/(G+R-B)
                Ranges roughly -1 to +1; higher = healthier green vegetation
        - brightness: mean luminance (0–255)
    """
    # Clamp box to tile bounds
    h, w = tile_rgb.shape[:2]
    x0 = max(0, int(xmin))
    y0 = max(0, int(ymin))
    x1 = min(w, int(xmax))
    y1 = min(h, int(ymax))

    # Extract the crown region
    crop = tile_rgb[y0:y1, x0:x1]

    if crop.size == 0:
        # Degenerate box — return NaNs
        return {k: float("nan") for k in [
            "mean_r", "mean_g", "mean_b",
            "std_r", "std_g", "std_b",
            "greenness_exg", "vari", "brightness"
        ]}

    # Convert to float for index calculations
    rf = crop[:, :, 0].astype(np.float64)
    gf = crop[:, :, 1].astype(np.float64)
    bf = crop[:, :, 2].astype(np.float64)

    # Basic channel stats
    mean_r, mean_g, mean_b = rf.mean(), gf.mean(), bf.mean()
    std_r, std_g, std_b = rf.std(), gf.std(), bf.std()

    # Excess Green Index (ExG) — simple vegetation index from RGB
    # ExG = 2*G - R - B (normalized to 0–255 scale, so range is roughly -255 to +510)
    exg = (2.0 * mean_g - mean_r - mean_b)

    # VARI — Visible Atmospherically Resistant Index
    # VARI = (G - R) / (G + R - B + epsilon)
    denom = mean_g + mean_r - mean_b
    if abs(denom) < 1e-6:
        vari = 0.0
    else:
        vari = (mean_g - mean_r) / denom

    # Overall brightness
    brightness = (mean_r + mean_g + mean_b) / 3.0

    return {
        "mean_r": round(mean_r, 1),
        "mean_g": round(mean_g, 1),
        "mean_b": round(mean_b, 1),
        "std_r": round(std_r, 1),
        "std_g": round(std_g, 1),
        "std_b": round(std_b, 1),
        "greenness_exg": round(exg, 2),
        "vari": round(vari, 4),
        "brightness": round(brightness, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# CORE: INSPECT THE ORTHOMOSAIC
# ═══════════════════════════════════════════════════════════════════════════

def inspect_orthomosaic(tif_path):
    """
    Print metadata about the GeoTIFF and return key properties.
    Reuses the inspection logic from inspect_ortho.py.
    """
    with rasterio.open(tif_path) as src:
        gsd_x, gsd_y = get_gsd_meters(src)

        print("=" * 60)
        print(f"ORTHOMOSAIC: {tif_path}")
        print(f"  Dimensions:  {src.width} x {src.height} pixels")
        print(f"  Bands:       {src.count} ({src.dtypes})")
        print(f"  CRS:         {src.crs}")
        print(f"  Bounds:      {src.bounds}")
        if gsd_x is not None:
            print(f"  GSD:         {gsd_x*100:.2f} x {gsd_y*100:.2f} cm/pixel")
        else:
            print(f"  GSD:         Unknown (no CRS)")
        file_mb = os.path.getsize(tif_path) / (1024 * 1024)
        print(f"  File size:   {file_mb:.1f} MB")
        print("=" * 60)

        return {
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs) if src.crs else None,
            "bounds": dict(zip(["left", "bottom", "right", "top"], src.bounds)),
            "gsd_x_m": gsd_x,
            "gsd_y_m": gsd_y,
            "transform": src.transform,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CORE: TILE THE ORTHOMOSAIC AND RUN DEEPFOREST
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(tif_path, patch_size, patch_overlap, score_thresh,
                 tile_size, tile_overlap, output_dir, max_tiles=0,
                 annotate=False):
    """
    Main pipeline:
      1. Load DeepForest pretrained model
      2. Iterate over tiles from the orthomosaic (windowed reads — low memory)
      3. For each tile, run DeepForest predict_tile
      4. Convert detections to global pixel coords + geo coords
      5. Compute per-tree crown diameter and color metrics
      6. Aggregate all results and save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load DeepForest model ──────────────────────────────────
    print("\n[1/5] Loading DeepForest pretrained model...")
    from deepforest import main as df_main

    # DeepForest 2.0: pretrained weights load automatically on init.
    # (use_release() was deprecated — no longer needed)
    model = df_main.deepforest()
    print("       Model loaded successfully.")

    # ── Step 2: Open the raster and prepare tiling ─────────────────────
    print("\n[2/5] Preparing tiles from orthomosaic...")
    src = rasterio.open(tif_path)
    gsd_x, gsd_y = get_gsd_meters(src)
    transform = src.transform

    # Use the average GSD for crown diameter calculations
    if gsd_x is not None and gsd_y is not None:
        gsd_avg_m = (gsd_x + gsd_y) / 2.0
        print(f"       GSD: {gsd_avg_m*100:.2f} cm/pixel")
    else:
        gsd_avg_m = None
        print("       WARNING: No CRS found — crown diameters will be in pixels only.")

    # Calculate tile grid
    step = tile_size - tile_overlap
    tiles = []
    for row_start in range(0, src.height, step):
        for col_start in range(0, src.width, step):
            win_h = min(tile_size, src.height - row_start)
            win_w = min(tile_size, src.width - col_start)
            # Skip tiny edge tiles (less than half the tile size)
            if win_h < tile_size // 2 or win_w < tile_size // 2:
                continue
            tiles.append((row_start, col_start, win_w, win_h))

    # Respect max_tiles limit if set
    if max_tiles > 0:
        tiles = tiles[:max_tiles]

    print(f"       Total tiles to process: {len(tiles)}")

    # ── Step 3: Process each tile ──────────────────────────────────────
    print("\n[3/5] Running DeepForest detection on each tile...")
    all_trees = []  # will hold one dict per detected tree
    tree_id_counter = 0

    # Store tile image data for composite generation when --annotate is set
    tile_images = []  # list of (row_start, col_start, tile_rgb, tile_annotated)

    for tile_idx, (row_start, col_start, win_w, win_h) in enumerate(tiles):
        # Read the tile (windowed read — doesn't load full raster)
        window = Window(col_start, row_start, win_w, win_h)
        data = src.read(window=window)  # shape: (bands, H, W)

        # Convert to RGB uint8 numpy array (H, W, 3) — what DeepForest expects
        if data.shape[0] >= 3:
            # Take first 3 bands as RGB
            tile_rgb = np.moveaxis(data[:3], 0, -1)  # (3, H, W) → (H, W, 3)
        elif data.shape[0] == 1:
            # Grayscale → fake RGB
            tile_rgb = np.moveaxis(np.repeat(data, 3, axis=0), 0, -1)
        else:
            tile_rgb = np.moveaxis(data, 0, -1)

        # Normalize to uint8 if needed (some GeoTIFFs are 16-bit or float)
        if tile_rgb.dtype != np.uint8:
            valid = tile_rgb[tile_rgb > 0]
            if valid.size > 0:
                p2, p98 = np.percentile(valid, [2, 98])
            else:
                p2, p98 = 0, 255
            tile_rgb = np.clip(
                (tile_rgb.astype(np.float64) - p2) / max(p98 - p2, 1) * 255,
                0, 255
            ).astype(np.uint8)

        # Skip mostly-empty tiles (nodata / black borders from the mosaic)
        if np.mean(tile_rgb) < 10:
            continue

        # DeepForest expects BGR (OpenCV convention) for raw_image input,
        # but predict_tile with a numpy array expects BGR via cv2 convention.
        # The `image` kwarg in predict_tile expects a numpy array in BGR.
        tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)

        # Save raw tile if --annotate is set (even before detection,
        # so we can see tiles with zero detections too)
        if annotate:
            raw_dir = output_dir / "raw_tiles"
            raw_dir.mkdir(parents=True, exist_ok=True)
            tile_name = f"tile_{tile_idx:03d}_r{row_start}_c{col_start}"
            cv2.imwrite(str(raw_dir / f"{tile_name}.jpg"),
                        tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Run DeepForest predict_tile on this tile
        try:
            detections_df = model.predict_tile(
                image=tile_bgr,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                iou_threshold=0.15,  # NMS threshold for overlapping boxes
            )
        except Exception as e:
            print(f"       WARNING: Tile {tile_idx} failed: {e}")
            # Store unannotated tile for composite even on failure
            if annotate:
                tile_images.append((row_start, col_start, tile_rgb, tile_rgb.copy()))
            continue

        # Filter by score threshold
        if detections_df is None or detections_df.empty:
            if annotate:
                tile_images.append((row_start, col_start, tile_rgb, tile_rgb.copy()))
            continue
        detections_df = detections_df[detections_df["score"] >= score_thresh]

        if detections_df.empty:
            if annotate:
                tile_images.append((row_start, col_start, tile_rgb, tile_rgb.copy()))
            continue

        # ── Save annotated tile image if --annotate flag is set ────────
        tile_annotated = tile_rgb.copy()
        if annotate:
            annotate_dir = output_dir / "annotated_tiles"
            annotate_dir.mkdir(parents=True, exist_ok=True)

            # Draw bounding boxes on a copy of the RGB tile
            for _, det_row in detections_df.iterrows():
                # Box coordinates (local to this tile)
                x0 = int(det_row["xmin"])
                y0 = int(det_row["ymin"])
                x1 = int(det_row["xmax"])
                y1 = int(det_row["ymax"])
                score_val = det_row["score"]

                # Green box
                color = (0, 255, 0)
                thickness = 2
                cv2.rectangle(tile_annotated, (x0, y0), (x1, y1),
                              color, thickness)

                # Score label above the box
                label_text = f"{score_val:.2f}"
                cv2.putText(tile_annotated, label_text, (x0, max(y0 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            tile_name = f"tile_{tile_idx:03d}_r{row_start}_c{col_start}"
            cv2.imwrite(str(annotate_dir / f"{tile_name}_annotated.jpg"),
                        cv2.cvtColor(tile_annotated, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Store for composite generation
            tile_images.append((row_start, col_start, tile_rgb, tile_annotated))

        # ── Step 4: For each detection, compute metrics ────────────────
        for _, det in detections_df.iterrows():
            # Local pixel coords within this tile
            local_xmin = det["xmin"]
            local_ymin = det["ymin"]
            local_xmax = det["xmax"]
            local_ymax = det["ymax"]

            # Global pixel coords in the full orthomosaic
            global_xmin = local_xmin + col_start
            global_ymin = local_ymin + row_start
            global_xmax = local_xmax + col_start
            global_ymax = local_ymax + row_start

            # Crown width and height in pixels
            crown_w_px = global_xmax - global_xmin
            crown_h_px = global_ymax - global_ymin

            # Crown diameter: average of width and height (trees aren't perfectly square)
            crown_diam_px = (crown_w_px + crown_h_px) / 2.0

            # Convert to meters if GSD is available
            if gsd_avg_m is not None:
                crown_diam_m = crown_diam_px * gsd_avg_m
                crown_w_m = crown_w_px * gsd_avg_m
                crown_h_m = crown_h_px * gsd_avg_m
            else:
                crown_diam_m = None
                crown_w_m = None
                crown_h_m = None

            # Geo-coordinates of the crown center
            center_col = (global_xmin + global_xmax) / 2.0
            center_row = (global_ymin + global_ymax) / 2.0
            geo_x, geo_y = pixel_to_geo(center_col, center_row, transform)

            # Color / health metrics from the RGB crop
            color_metrics = compute_crown_color_metrics(
                tile_rgb, local_xmin, local_ymin, local_xmax, local_ymax
            )

            # Assemble tree record
            tree_record = {
                "tree_id": tree_id_counter,
                "score": round(float(det["score"]), 4),
                # Pixel coordinates (global, in full orthomosaic)
                "px_xmin": round(float(global_xmin), 1),
                "px_ymin": round(float(global_ymin), 1),
                "px_xmax": round(float(global_xmax), 1),
                "px_ymax": round(float(global_ymax), 1),
                # Crown dimensions
                "crown_width_px": round(float(crown_w_px), 1),
                "crown_height_px": round(float(crown_h_px), 1),
                "crown_diameter_px": round(float(crown_diam_px), 1),
                "crown_width_m": round(crown_w_m, 3) if crown_w_m else None,
                "crown_height_m": round(crown_h_m, 3) if crown_h_m else None,
                "crown_diameter_m": round(crown_diam_m, 3) if crown_diam_m else None,
                # Geo-coordinates (CRS of the raster)
                "geo_x": round(geo_x, 6),
                "geo_y": round(geo_y, 6),
                # Color / vegetation metrics
                **color_metrics,
                # Source tile info (for debugging)
                "tile_idx": tile_idx,
                "tile_origin_col": col_start,
                "tile_origin_row": row_start,
            }

            all_trees.append(tree_record)
            tree_id_counter += 1

        # Progress update
        if (tile_idx + 1) % 5 == 0 or tile_idx == len(tiles) - 1:
            print(f"       Processed tile {tile_idx + 1}/{len(tiles)} "
                  f"— {tree_id_counter} trees detected so far")

    src.close()

    # ── Generate composite images if --annotate was used ───────────────
    if annotate and tile_images:
        print("\n       Stitching composite images...")

        # Figure out the bounding box of all processed tiles in pixel space
        max_row = max(r + t.shape[0] for r, c, t, _ in tile_images)
        max_col = max(c + t.shape[1] for r, c, t, _ in tile_images)

        # Downscale factor — keep composite manageable in memory
        # Target ~4000px on the longest side
        scale = min(1.0, 4000.0 / max(max_row, max_col))
        comp_h = int(max_row * scale)
        comp_w = int(max_col * scale)

        # Create blank canvases (black background)
        composite_raw = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)
        composite_ann = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)

        for row_start, col_start, tile_raw, tile_ann in tile_images:
            # Compute where this tile lands in the downscaled composite
            y0 = int(row_start * scale)
            x0 = int(col_start * scale)
            th = int(tile_raw.shape[0] * scale)
            tw = int(tile_raw.shape[1] * scale)

            # Resize tiles to match composite scale
            raw_small = cv2.resize(tile_raw, (tw, th))
            ann_small = cv2.resize(tile_ann, (tw, th))

            # Clamp to canvas bounds
            th = min(th, comp_h - y0)
            tw = min(tw, comp_w - x0)

            # Paste into composite (later tiles overwrite overlap regions)
            composite_raw[y0:y0+th, x0:x0+tw] = raw_small[:th, :tw]
            composite_ann[y0:y0+th, x0:x0+tw] = ann_small[:th, :tw]

        # Save composites
        comp_raw_path = output_dir / "composite_raw.jpg"
        comp_ann_path = output_dir / "composite_annotated.jpg"
        cv2.imwrite(str(comp_raw_path),
                    cv2.cvtColor(composite_raw, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(comp_ann_path),
                    cv2.cvtColor(composite_ann, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"       Composite raw:       {comp_raw_path}")
        print(f"       Composite annotated: {comp_ann_path}")
        print(f"       Composite size:      {comp_w}x{comp_h} px "
              f"(scale: {scale:.2f}x)")

    # ── Step 5: Save results ───────────────────────────────────────────
    print(f"\n[4/5] Saving results ({len(all_trees)} trees detected)...")

    if len(all_trees) == 0:
        print("       No trees detected! Try lowering --score-thresh or check your image.")
        return

    # Convert to DataFrame
    trees_df = pd.DataFrame(all_trees)

    # Save as CSV (primary output for data analysis)
    csv_path = output_dir / "tree_inventory.csv"
    trees_df.to_csv(csv_path, index=False)
    print(f"       CSV:  {csv_path}")

    # Save as JSON (for web/API consumption)
    json_path = output_dir / "tree_inventory.json"
    with open(json_path, "w") as f:
        json.dump(all_trees, f, indent=2)
    print(f"       JSON: {json_path}")

    # Save summary statistics
    summary = generate_summary(trees_df, tif_path, gsd_avg_m)
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"       Summary: {summary_path}")

    # Generate visualizations
    print("\n[5/5] Generating visualizations...")
    generate_visualizations(trees_df, tif_path, output_dir, gsd_avg_m)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Trees detected:    {len(all_trees)}")
    if gsd_avg_m:
        diams = trees_df["crown_diameter_m"].dropna()
        print(f"  Mean crown diam:   {diams.mean():.2f} m")
        print(f"  Min crown diam:    {diams.min():.2f} m")
        print(f"  Max crown diam:    {diams.max():.2f} m")
    print(f"  Mean confidence:   {trees_df['score'].mean():.3f}")
    print(f"  Mean VARI:         {trees_df['vari'].mean():.4f}")
    print(f"  Output directory:  {output_dir}")
    print(f"{'='*60}")

    return trees_df


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def generate_summary(trees_df, tif_path, gsd_avg_m):
    """
    Compute summary statistics for the detected tree inventory.
    Returns a dict suitable for JSON serialization.
    """
    summary = {
        "source_file": str(tif_path),
        "total_trees": len(trees_df),
        "detection_confidence": {
            "mean": round(trees_df["score"].mean(), 4),
            "median": round(trees_df["score"].median(), 4),
            "min": round(trees_df["score"].min(), 4),
            "max": round(trees_df["score"].max(), 4),
        },
        "crown_diameter_px": {
            "mean": round(trees_df["crown_diameter_px"].mean(), 1),
            "std": round(trees_df["crown_diameter_px"].std(), 1),
            "min": round(trees_df["crown_diameter_px"].min(), 1),
            "max": round(trees_df["crown_diameter_px"].max(), 1),
        },
        "vegetation_indices": {
            "exg_mean": round(trees_df["greenness_exg"].mean(), 2),
            "exg_std": round(trees_df["greenness_exg"].std(), 2),
            "vari_mean": round(trees_df["vari"].mean(), 4),
            "vari_std": round(trees_df["vari"].std(), 4),
        },
        "color_stats": {
            "mean_r": round(trees_df["mean_r"].mean(), 1),
            "mean_g": round(trees_df["mean_g"].mean(), 1),
            "mean_b": round(trees_df["mean_b"].mean(), 1),
        },
    }

    # Add metric-unit stats if GSD is available
    if gsd_avg_m is not None:
        diams = trees_df["crown_diameter_m"].dropna()
        summary["gsd_cm_per_pixel"] = round(gsd_avg_m * 100, 2)
        summary["crown_diameter_m"] = {
            "mean": round(diams.mean(), 3),
            "std": round(diams.std(), 3),
            "min": round(diams.min(), 3),
            "max": round(diams.max(), 3),
        }

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

def generate_visualizations(trees_df, tif_path, output_dir, gsd_avg_m):
    """
    Create a set of diagnostic plots:
      1. Crown diameter histogram
      2. Greenness (VARI) histogram
      3. Scatter: crown diameter vs VARI
      4. Tree location map (pixel coords, colored by VARI)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Sylva Tree Analysis — {len(trees_df)} trees detected", fontsize=14)

    # ── Plot 1: Crown diameter distribution ────────────────────────────
    ax = axes[0, 0]
    if gsd_avg_m is not None and "crown_diameter_m" in trees_df.columns:
        diams = trees_df["crown_diameter_m"].dropna()
        ax.hist(diams, bins=30, color="#2d6a4f", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Crown Diameter (m)")
        ax.set_title("Crown Diameter Distribution")
        ax.axvline(diams.mean(), color="#d62828", linestyle="--",
                    label=f"Mean: {diams.mean():.2f} m")
    else:
        diams = trees_df["crown_diameter_px"]
        ax.hist(diams, bins=30, color="#2d6a4f", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Crown Diameter (pixels)")
        ax.set_title("Crown Diameter Distribution")
        ax.axvline(diams.mean(), color="#d62828", linestyle="--",
                    label=f"Mean: {diams.mean():.1f} px")
    ax.set_ylabel("Count")
    ax.legend()

    # ── Plot 2: VARI distribution ──────────────────────────────────────
    ax = axes[0, 1]
    vari_vals = trees_df["vari"].dropna()
    ax.hist(vari_vals, bins=30, color="#588157", edgecolor="white", alpha=0.85)
    ax.set_xlabel("VARI (Visible Atmospherically Resistant Index)")
    ax.set_ylabel("Count")
    ax.set_title("Vegetation Health (VARI) Distribution")
    ax.axvline(vari_vals.mean(), color="#d62828", linestyle="--",
                label=f"Mean: {vari_vals.mean():.4f}")
    ax.legend()

    # ── Plot 3: Crown diameter vs VARI scatter ─────────────────────────
    ax = axes[1, 0]
    if gsd_avg_m is not None and "crown_diameter_m" in trees_df.columns:
        x_data = trees_df["crown_diameter_m"].dropna()
        x_label = "Crown Diameter (m)"
    else:
        x_data = trees_df["crown_diameter_px"]
        x_label = "Crown Diameter (pixels)"
    scatter = ax.scatter(
        x_data, trees_df["vari"].iloc[:len(x_data)],
        c=trees_df["score"].iloc[:len(x_data)],
        cmap="YlGn", s=8, alpha=0.6, edgecolors="none"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("VARI")
    ax.set_title("Crown Size vs Health (color = confidence)")
    plt.colorbar(scatter, ax=ax, label="Detection Score")

    # ── Plot 4: Spatial map of tree locations ──────────────────────────
    ax = axes[1, 1]
    # Plot trees at their pixel center, colored by VARI
    cx = (trees_df["px_xmin"] + trees_df["px_xmax"]) / 2
    cy = (trees_df["px_ymin"] + trees_df["px_ymax"]) / 2
    scatter2 = ax.scatter(
        cx, cy,
        c=trees_df["vari"], cmap="RdYlGn", s=6, alpha=0.7, edgecolors="none"
    )
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_title("Tree Locations (color = VARI)")
    ax.invert_yaxis()  # image convention: y increases downward
    ax.set_aspect("equal")
    plt.colorbar(scatter2, ax=ax, label="VARI")

    plt.tight_layout()
    fig_path = output_dir / "analysis_plots.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"       Plots: {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sylva — DeepForest Tree Analysis Pipeline"
    )
    parser.add_argument(
        "tif_path",
        help="Path to the orthomosaic GeoTIFF file"
    )
    parser.add_argument(
        "--patch-size", type=int, default=DEFAULT_PATCH_SIZE,
        help=f"DeepForest patch size in pixels (default: {DEFAULT_PATCH_SIZE})"
    )
    parser.add_argument(
        "--patch-overlap", type=float, default=DEFAULT_PATCH_OVERLAP,
        help=f"DeepForest patch overlap ratio (default: {DEFAULT_PATCH_OVERLAP})"
    )
    parser.add_argument(
        "--score-thresh", type=float, default=DEFAULT_SCORE_THRESH,
        help=f"Minimum detection confidence (default: {DEFAULT_SCORE_THRESH})"
    )
    parser.add_argument(
        "--tile-size", type=int, default=DEFAULT_TILE_SIZE,
        help=f"Tile size for processing large rasters (default: {DEFAULT_TILE_SIZE})"
    )
    parser.add_argument(
        "--tile-overlap", type=int, default=DEFAULT_TILE_OVERLAP,
        help=f"Tile overlap in pixels (default: {DEFAULT_TILE_OVERLAP})"
    )
    parser.add_argument(
        "--max-tiles", type=int, default=0,
        help="Max tiles to process (0 = all, useful for testing)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--annotate", action="store_true",
        help="Save annotated tile images with bounding boxes drawn on them"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.tif_path):
        print(f"Error: File not found: {args.tif_path}")
        sys.exit(1)

    # Print metadata
    meta = inspect_orthomosaic(args.tif_path)

    # Run the pipeline
    run_pipeline(
        tif_path=args.tif_path,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
        score_thresh=args.score_thresh,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        output_dir=args.output_dir,
        max_tiles=args.max_tiles,
        annotate=args.annotate,
    )


if __name__ == "__main__":
    main()
