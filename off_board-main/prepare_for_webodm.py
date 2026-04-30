#!/usr/bin/env python3
"""
Sylva — Prepare flight data for WebODM processing.

This script reads the Sylva CSV log and dataset.json, then:
  1. Generates a geo.txt file (WebODM geolocation format)
  2. Optionally writes EXIF GPS tags directly into the JPG images
     (WebODM prefers EXIF-tagged images, geo.txt is the fallback)

Usage:
  python prepare_for_webodm.py \
      --csv log.csv \
      --json dataset.json \
      --images-dir ./images \
      --output-dir ./webodm_upload

Requirements:
  pip install piexif Pillow
"""

import csv
import json
import math
import os
import argparse
from pathlib import Path


def radians_to_degrees(rad):
    """Convert radians to degrees."""
    return math.degrees(rad)


def decimal_to_dms_rational(decimal_deg):
    """
    Convert a decimal degree value to EXIF-compatible DMS rational format.
    Returns a tuple of three tuples: ((d,1), (m,1), (s_num, s_den))
    """
    # Work with absolute value; sign is handled by the ref tag (N/S, E/W)
    decimal_deg = abs(decimal_deg)
    degrees = int(decimal_deg)
    minutes_float = (decimal_deg - degrees) * 60
    minutes = int(minutes_float)
    # Use a denominator of 10000 for sub-arcsecond precision
    seconds_float = (minutes_float - minutes) * 60
    seconds_num = int(round(seconds_float * 10000))
    seconds_den = 10000
    return ((degrees, 1), (minutes, 1), (seconds_num, seconds_den))


def write_geo_txt(rows, output_path):
    """
    Write a WebODM-compatible geo.txt file.

    Format (per WebODM docs):
      Line 1: EPSG:<code>
      Subsequent lines: image_name lon lat alt yaw pitch roll
    
    Note: For EPSG:4326 (WGS84), geo_x = longitude, geo_y = latitude.
    """
    with open(output_path, "w") as f:
        # Header line: coordinate reference system
        f.write("EPSG:4326\n")

        for row in rows:
            # Extract the image filename from the original Pi path
            # e.g. /home/sylva0/on_board/logs/images/0000000000.jpg -> 0000000000.jpg
            img_filename = os.path.basename(row["img_path"])

            lat = float(row["lat_deg"])
            lon = float(row["lon_deg"])
            alt = float(row["alt_m"])

            # Convert orientation from radians to degrees for WebODM
            yaw_deg = radians_to_degrees(float(row["yaw_rad"]))
            pitch_deg = radians_to_degrees(float(row["pitch_rad"]))
            roll_deg = radians_to_degrees(float(row["roll_rad"]))

            # WebODM geo.txt format: filename lon lat alt yaw pitch roll
            # (for EPSG:4326, x=longitude, y=latitude)
            f.write(
                f"{img_filename} {lon:.7f} {lat:.7f} {alt:.3f} "
                f"{yaw_deg:.4f} {pitch_deg:.4f} {roll_deg:.4f}\n"
            )

    print(f"[OK] geo.txt written: {output_path}  ({len(rows)} entries)")


def write_exif_gps(rows, images_dir, output_dir):
    """
    Copy images to output_dir and inject EXIF GPS tags.
    This is the preferred method — WebODM reads GPS from EXIF automatically.
    """
    try:
        import piexif
        from PIL import Image
    except ImportError:
        print("[WARN] piexif or Pillow not installed. Skipping EXIF tagging.")
        print("       Install with: pip install piexif Pillow")
        return False

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    tagged_count = 0
    skipped_count = 0

    for row in rows:
        img_filename = os.path.basename(row["img_path"])
        src_path = images_dir / img_filename
        dst_path = output_dir / img_filename

        if not src_path.exists():
            print(f"  [SKIP] {img_filename} — not found in {images_dir}")
            skipped_count += 1
            continue

        lat = float(row["lat_deg"])
        lon = float(row["lon_deg"])
        alt = float(row["alt_m"])

        # Build EXIF GPS IFD
        # Latitude reference: N for positive, S for negative
        lat_ref = b"N" if lat >= 0 else b"S"
        lon_ref = b"E" if lon >= 0 else b"W"

        gps_ifd = {
            piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
            piexif.GPSIFD.GPSLatitudeRef: lat_ref,
            piexif.GPSIFD.GPSLatitude: decimal_to_dms_rational(lat),
            piexif.GPSIFD.GPSLongitudeRef: lon_ref,
            piexif.GPSIFD.GPSLongitude: decimal_to_dms_rational(lon),
            # Altitude: reference 0 = above sea level
            piexif.GPSIFD.GPSAltitudeRef: 0,
            piexif.GPSIFD.GPSAltitude: (int(round(alt * 1000)), 1000),
        }

        # Load existing EXIF or create fresh
        try:
            img = Image.open(str(src_path))
            exif_dict = piexif.load(img.info.get("exif", b""))
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

        # Inject GPS data
        exif_dict["GPS"] = gps_ifd

        # Dump EXIF bytes and save
        exif_bytes = piexif.dump(exif_dict)
        img.save(str(dst_path), exif=exif_bytes, quality=95)
        tagged_count += 1

    print(f"[OK] EXIF GPS tagged: {tagged_count} images -> {output_dir}")
    if skipped_count:
        print(f"     ({skipped_count} images not found in source directory)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Sylva flight data for WebODM orthomosaic generation."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to the Sylva flight log CSV (log.csv)"
    )
    parser.add_argument(
        "--json", default=None,
        help="Path to dataset.json (optional, used for camera info)"
    )
    parser.add_argument(
        "--images-dir", default=None,
        help="Directory containing the JPG images. If provided, EXIF GPS tags "
             "will be written to copies in --output-dir."
    )
    parser.add_argument(
        "--output-dir", default="./webodm_upload",
        help="Output directory for geo.txt and tagged images (default: ./webodm_upload)"
    )
    parser.add_argument(
        "--geo-only", action="store_true",
        help="Only generate geo.txt, skip EXIF tagging even if --images-dir is set."
    )
    args = parser.parse_args()

    # --- Read CSV ---
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Loaded {len(rows)} entries from {args.csv}")

    # Filter to only enabled frames (toggle_enabled == 1)
    enabled_rows = [r for r in rows if r.get("toggle_enabled", "1") == "1"]
    print(f"  {len(enabled_rows)} frames with toggle_enabled=1")

    # --- Read JSON (optional, for camera info display) ---
    if args.json:
        with open(args.json, "r") as f:
            dataset = json.load(f)
        cam = dataset.get("camera_model", {})
        print(f"  Camera: {cam.get('image_width')}x{cam.get('image_height')}, "
              f"fx={cam.get('fx')}, fy={cam.get('fy')}")

    # --- Create output directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate geo.txt ---
    geo_path = output_dir / "geo.txt"
    write_geo_txt(enabled_rows, geo_path)

    # --- EXIF tagging (if images directory provided) ---
    if args.images_dir and not args.geo_only:
        print(f"\nTagging images with EXIF GPS data...")
        write_exif_gps(enabled_rows, args.images_dir, output_dir)
    elif args.images_dir and args.geo_only:
        print("\n[INFO] --geo-only flag set, skipping EXIF tagging.")
    else:
        print("\n[INFO] No --images-dir provided. Only geo.txt generated.")
        print("       To also EXIF-tag images, re-run with --images-dir /path/to/images")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  OUTPUT READY: {output_dir}/")
    print(f"  - geo.txt          ({len(enabled_rows)} image positions)")
    print(f"{'='*60}")
    print(f"\nNext step: Upload the contents of {output_dir}/ to WebODM.")
    print(f"  If images are EXIF-tagged: upload images only (WebODM reads GPS from EXIF).")
    print(f"  If using geo.txt:          upload images + geo.txt together.")


if __name__ == "__main__":
    main()
