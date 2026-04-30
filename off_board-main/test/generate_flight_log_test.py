# pip install selenium
# pip install pyproj
# pip install pillow

import json
from pathlib import Path
from pyproj import Transformer
from io import BytesIO
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import time


# ============================================================
# USER SETTINGS - EDIT THESE
# ============================================================

DATASET_NAME = "flight_0"
OUT_ROOT = Path(".")

TOP_LEFT = (35.781669, -78.666190)      # (lat, lon)
BOTTOM_RIGHT = (35.779836, -78.663957)  # (lat, lon)

RIGHT_STEP = 20.0   # meters
DOWN_STEP = 20.0    # meters

WIDTH_OFFSET = 210 # [px]
HEIGHT_OFFSET = 210 # [px]

ALT_M = 0.0

CAMERA_MODEL = {
    "image_width": 4656,
    "image_height": 3496,
    "fx": 3200.0,
    "fy": 3200.0,
    "cx": 2328.0,
    "cy": 1748.0
}

IMAGE_EXT = ".png"   # change if your future screenshot method saves jpg instead


# ============================================================
# COORDINATE TRANSFORMS
# ============================================================

# WGS84 lat/lon <-> UTM zone 17N (good for Raleigh area)
latlon_to_xy = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
xy_to_latlon = Transformer.from_crs("EPSG:32617", "EPSG:4326", always_xy=True)
driver = webdriver.Chrome()

# ============================================================
# PLACEHOLDER FOR IMAGE CAPTURE
# ============================================================

def get_screenshot(lat, lon, out_path, zoom=9):
    url = f"https://www.google.com/maps/@{lat},{lon},{zoom}m/data=!3m1!1e3"
    driver.get(url)
    WebDriverWait(driver, 10).until(
    lambda d: d.execute_script("return document.readyState") == "complete"
    )
    time.sleep(5)
    png_bytes = driver.get_screenshot_as_png()

    # open with PIL
    img = Image.open(BytesIO(png_bytes))

    width, height = img.size
    width_center = width // 2
    height_center = height // 2


    # crop box = (left, top, right, bottom)
    cropped = img.crop((width_center-WIDTH_OFFSET, height_center-HEIGHT_OFFSET, width_center+WIDTH_OFFSET, height_center+HEIGHT_OFFSET))

    # save only the cropped image
    cropped.save(out_path)


# ============================================================
# GRID GENERATION
# ============================================================

def get_centers():
    """
    Returns a list of (lat, lon) center points spaced by RIGHT_STEP and
    DOWN_STEP across the rectangle defined by TOP_LEFT and BOTTOM_RIGHT.
    """
    lat_start, lon_start = TOP_LEFT
    lat_end, lon_end = BOTTOM_RIGHT

    current_x, current_y = latlon_to_xy.transform(lon_start, lat_start)
    end_x, end_y = latlon_to_xy.transform(lon_end, lat_end)

    left_x = min(current_x, end_x)
    right_x = max(current_x, end_x)
    top_y = max(current_y, end_y)
    bottom_y = min(current_y, end_y)

    centers = []
    y = top_y
    while y >= bottom_y:
        x = left_x
        while x <= right_x:
            lon, lat = xy_to_latlon.transform(x, y)
            centers.append((lat, lon))
            x += RIGHT_STEP
        y -= DOWN_STEP

    return centers


# ============================================================
# JSON LOGGING
# ============================================================

def build_dataset_json(frames):
    return {
        "dataset_name": DATASET_NAME,
        "crs": "EPSG:4326",
        "alt_reference": "AGL",
        "camera_model": CAMERA_MODEL,
        "frames": frames
    }


# ============================================================
# MAIN
# ============================================================

def main():
    dataset_dir = OUT_ROOT / DATASET_NAME
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    centers = get_centers()

    print(f"Generating dataset: {DATASET_NAME}")
    print(f"Top left:      {TOP_LEFT}")
    print(f"Bottom right:  {BOTTOM_RIGHT}")
    print(f"Right step:    {RIGHT_STEP} m")
    print(f"Down step:     {DOWN_STEP} m")
    print(f"Total centers: {len(centers)}")

    frames = []

    for i, (lat, lon) in enumerate(centers):
        filename = f"image_{i:04d}{IMAGE_EXT}"
        image_path_abs = images_dir / filename
        image_path_rel = f"{DATASET_NAME}/images/{filename}"

        try:
            get_screenshot(lat, lon, image_path_abs)
            print(f"[OK] Saved {image_path_rel}")
        except NotImplementedError:
            print(f"[SKIP] get_screenshot not implemented yet for frame {i}")
        except Exception as e:
            print(f"[WARN] Failed to save {image_path_rel}: {e}")
            continue

        frame = {
            "id": i,
            "timestamp": float(i),
            "image_path": image_path_rel,
            "lat": lat,
            "lon": lon,
            "alt_m": ALT_M,
            "roll_deg": 0.0,
            "pitch_deg": -90.0,
            "yaw_deg": 0.0
        }
        frames.append(frame)

    dataset_json = build_dataset_json(frames)

    json_path = dataset_dir / "dataset.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)

    print("\nDone.")
    print(f"Images directory: {images_dir}")
    print(f"JSON saved to:    {json_path}")
    print(f"Frames logged:    {len(frames)}")


if __name__ == "__main__":
    main()