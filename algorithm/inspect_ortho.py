"""
Sylva — GeoTIFF Orthomosaic Inspector & Tiler
Reads a large orthomosaic .tif, prints metadata, and extracts
tiles suitable for feeding into the tree detection pipeline.

Usage:
    python inspect_ortho.py /path/to/orthomosaic.tif

Dependencies:
    pip install rasterio numpy opencv-python matplotlib
"""

import sys
import os
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path

# ── Configuration ──
TILE_SIZE = 1024        # pixels per tile side (adjust based on your SAHI slice size)
TILE_OVERLAP = 128      # pixel overlap between tiles
OUTPUT_DIR = Path("outputs/ortho_tiles")


def inspect_geotiff(tif_path: str):
    """Print metadata about the GeoTIFF without loading pixel data."""
    with rasterio.open(tif_path) as src:
        print("=" * 60)
        print(f"FILE: {tif_path}")
        print(f"  Size:       {src.width} x {src.height} pixels")
        print(f"  Bands:      {src.count} ({src.dtypes})")
        print(f"  CRS:        {src.crs}")
        print(f"  Transform:  {src.transform}")
        print(f"  Bounds:     {src.bounds}")
        print(f"  Resolution: {src.res[0]:.4f} x {src.res[1]:.4f} (units of CRS)")
        
        # Estimate GSD if CRS is projected (meters)
        if src.crs and src.crs.is_projected:
            gsd_cm = src.res[0] * 100
            print(f"  GSD:        ~{gsd_cm:.2f} cm/pixel")
        elif src.crs and src.crs.is_geographic:
            # Rough estimate: 1 degree lat ≈ 111,320 meters
            gsd_m = src.res[1] * 111320
            print(f"  GSD:        ~{gsd_m*100:.2f} cm/pixel (approx from geographic CRS)")
        
        # File size
        file_size_mb = os.path.getsize(tif_path) / (1024 * 1024)
        print(f"  File size:  {file_size_mb:.1f} MB")
        
        # Check for internal tiling (overviews/pyramids)
        overviews = src.overviews(1)
        if overviews:
            print(f"  Overviews:  {overviews} (has pyramids — fast zoom)")
        else:
            print(f"  Overviews:  None (consider adding with `gdaladdo`)")
        
        # Nodata value
        print(f"  NoData:     {src.nodata}")
        print("=" * 60)
        
        return src.width, src.height, src.count, src.crs, src.bounds


def generate_overview_image(tif_path: str, max_dim: int = 2048):
    """
    Create a downsampled overview image for quick visualization.
    Doesn't load the full raster into memory.
    """
    import cv2
    
    with rasterio.open(tif_path) as src:
        # Calculate downsample factor
        scale = max_dim / max(src.width, src.height)
        out_w = int(src.width * scale)
        out_h = int(src.height * scale)
        
        print(f"\nGenerating overview ({out_w}x{out_h}) from {src.width}x{src.height}...")
        
        # Read at reduced resolution (rasterio handles the resampling)
        # Read ALL bands first, then slice — out_shape must match band count
        data = src.read(
            out_shape=(src.count, out_h, out_w),
            resampling=rasterio.enums.Resampling.bilinear
        )
        
        # Handle different band counts
        if data.shape[0] >= 3:
            # RGB or RGBA — take first 3 bands
            rgb = np.moveaxis(data[:3], 0, -1)  # (3, H, W) -> (H, W, 3)
        elif data.shape[0] == 1:
            # Grayscale
            rgb = np.moveaxis(np.repeat(data, 3, axis=0), 0, -1)
        else:
            rgb = np.moveaxis(data[:3], 0, -1)
        
        # Normalize to 0-255 if needed
        if rgb.dtype != np.uint8:
            # Handle potential 16-bit or float data
            p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if np.any(rgb > 0) else (0, 255)
            rgb = np.clip((rgb - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        
        overview_path = OUTPUT_DIR / "overview.jpg"
        cv2.imwrite(str(overview_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 
                     [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  Saved: {overview_path}")
        
        return rgb


def extract_tiles(tif_path: str, tile_size: int = TILE_SIZE, 
                  overlap: int = TILE_OVERLAP, max_tiles: int = None):
    """
    Extract tiles from the orthomosaic for feeding into detection pipeline.
    Uses windowed reading — never loads the full raster.
    
    Each tile is saved as a JPEG with a sidecar JSON containing its
    geo-coordinates (for mapping detections back to real-world locations).
    """
    tiles_dir = OUTPUT_DIR / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = OUTPUT_DIR / "tile_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    import cv2
    
    with rasterio.open(tif_path) as src:
        step = tile_size - overlap
        tile_count = 0
        
        for row_start in range(0, src.height, step):
            for col_start in range(0, src.width, step):
                # Define the window
                win_height = min(tile_size, src.height - row_start)
                win_width = min(tile_size, src.width - col_start)
                
                # Skip tiny edge tiles
                if win_height < tile_size // 2 or win_width < tile_size // 2:
                    continue
                
                window = Window(col_start, row_start, win_width, win_height)
                
                # Read tile data
                data = src.read(window=window)
                
                # Take RGB bands
                if data.shape[0] >= 3:
                    rgb = np.moveaxis(data[:3], 0, -1)
                elif data.shape[0] == 1:
                    rgb = np.moveaxis(np.repeat(data, 3, axis=0), 0, -1)
                else:
                    rgb = np.moveaxis(data, 0, -1)
                
                # Normalize
                if rgb.dtype != np.uint8:
                    p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if np.any(rgb > 0) else (0, 255)
                    rgb = np.clip((rgb - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
                
                # Skip mostly-empty tiles (nodata / black borders)
                if np.mean(rgb) < 10:
                    continue
                
                # Get geo-coordinates of tile corners
                tile_bounds = rasterio.windows.bounds(window, src.transform)
                
                # Save tile image
                tile_name = f"tile_{row_start:06d}_{col_start:06d}"
                cv2.imwrite(str(tiles_dir / f"{tile_name}.jpg"),
                           cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Save tile metadata (for geo-referencing detections later)
                meta = {
                    "tile_name": tile_name,
                    "pixel_origin": {"row": row_start, "col": col_start},
                    "pixel_size": {"width": win_width, "height": win_height},
                    "geo_bounds": {
                        "left": tile_bounds[0],
                        "bottom": tile_bounds[1],
                        "right": tile_bounds[2],
                        "top": tile_bounds[3],
                    },
                    "crs": str(src.crs),
                }
                with open(meta_dir / f"{tile_name}.json", "w") as f:
                    json.dump(meta, f, indent=2)
                
                tile_count += 1
                if tile_count % 50 == 0:
                    print(f"  Extracted {tile_count} tiles...")
                
                if max_tiles and tile_count >= max_tiles:
                    print(f"  Reached max_tiles limit ({max_tiles})")
                    return tile_count
        
        print(f"\n  Total tiles extracted: {tile_count}")
        print(f"  Tiles saved to: {tiles_dir}")
        print(f"  Metadata saved to: {meta_dir}")
        return tile_count


def extract_sample_region(tif_path: str, center_pct: tuple = (0.5, 0.5),
                          region_size: int = 2048):
    """
    Extract a sample region from the center (or specified location) of the
    orthomosaic for quick testing with your detection pipeline.
    """
    import cv2
    
    with rasterio.open(tif_path) as src:
        center_row = int(src.height * center_pct[1])
        center_col = int(src.width * center_pct[0])
        
        row_start = max(0, center_row - region_size // 2)
        col_start = max(0, center_col - region_size // 2)
        
        window = Window(col_start, row_start, 
                       min(region_size, src.width - col_start),
                       min(region_size, src.height - row_start))
        
        data = src.read(window=window)
        
        if data.shape[0] >= 3:
            rgb = np.moveaxis(data[:3], 0, -1)
        else:
            rgb = np.moveaxis(np.repeat(data[:1], 3, axis=0), 0, -1)
        
        if rgb.dtype != np.uint8:
            p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if np.any(rgb > 0) else (0, 255)
            rgb = np.clip((rgb - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        
        sample_path = OUTPUT_DIR / "sample_region.jpg"
        cv2.imwrite(str(sample_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Also save geo-info
        bounds = rasterio.windows.bounds(window, src.transform)
        meta = {
            "pixel_origin": {"row": row_start, "col": col_start},
            "pixel_size": {"width": int(window.width), "height": int(window.height)},
            "geo_bounds": {"left": bounds[0], "bottom": bounds[1], 
                          "right": bounds[2], "top": bounds[3]},
            "crs": str(src.crs),
        }
        with open(OUTPUT_DIR / "sample_region_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"\nSample region extracted ({int(window.width)}x{int(window.height)} px)")
        print(f"  Pixel origin: row={row_start}, col={col_start}")
        print(f"  Geo bounds: {bounds}")
        print(f"  Saved: {sample_path}")
        
        return rgb


# ── Main ──
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_ortho.py <path_to_orthomosaic.tif>")
        print("\nOptions (pass as second argument):")
        print("  inspect  — metadata only (default)")
        print("  overview — generate downsampled overview image")
        print("  sample   — extract a sample region for testing")
        print("  tile     — extract all tiles for pipeline processing")
        sys.exit(1)
    
    tif_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "inspect"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(tif_path):
        print(f"Error: File not found: {tif_path}")
        sys.exit(1)
    
    # Always inspect first
    inspect_geotiff(tif_path)
    
    if mode == "overview":
        generate_overview_image(tif_path)
    elif mode == "sample":
        extract_sample_region(tif_path)
    elif mode == "tile":
        extract_tiles(tif_path)
    elif mode != "inspect":
        print(f"Unknown mode: {mode}")
