# Sylva — WebODM Setup & GeoTIFF Generation Guide

**Purpose:** Turn your 230 flight photos + GPS log into a georeferenced orthomosaic (GeoTIFF) using WebODM.

**What you have:**
- 230 JPG images (4656×3496, 16MP Arducam IMX519)
- `log.csv` with per-image GPS coordinates (lat, lon, alt) and orientation (roll, pitch, yaw)
- `dataset.json` with camera intrinsics (fx=3200, fy=3200, cx=2328, cy=1748)
- Flight area: ~45m × 65m near Raleigh, NC (EPSG:4326, ~35.7686°N, 78.6625°W)
- Altitude: ~10m AGL (very low — expect high resolution, ~0.3 cm/pixel)

**What you'll get:**
- `odm_orthophoto.tif` — a single stitched, georeferenced GeoTIFF of your entire flight area
- Point cloud, DSM (Digital Surface Model), and 3D textured mesh as bonuses

---

## Part 1: Install WebODM

WebODM runs inside Docker containers. You need Docker installed first.

### 1A. Install Docker

```bash
# Update package list
sudo apt update

# Install Docker prerequisites
sudo apt install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine + Compose plugin
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add your user to the docker group (avoids needing sudo every time)
sudo usermod -aG docker $USER

# IMPORTANT: Log out and back in for group change to take effect,
# or run: newgrp docker
```

Verify Docker is working:

```bash
docker run hello-world
```

### 1B. Install WebODM

```bash
# Clone WebODM
cd ~
git clone https://github.com/OpenDroneMap/WebODM.git --config core.autocrlf=input
cd WebODM

# Start WebODM (first run downloads ~2-3 GB of container images)
./webodm.sh start
```

WebODM will print something like `WebODM is running at http://localhost:8000`. Open that URL in your browser.

On first launch you'll create an admin account (pick any username/password you want — this is local only).

**Resource tips:** WebODM + NodeODM together use significant RAM during processing. For 230 images at 16MP, you'll want at least 16 GB of RAM available. If you're tight on memory, see the processing options in Part 3 for memory-saving flags.

To stop WebODM later: `./webodm.sh stop`
To restart: `./webodm.sh start`

---

## Part 2: Prepare Your Data

Your images came off the Raspberry Pi with GPS data stored in the CSV, not embedded in the EXIF headers. WebODM needs to know where each photo was taken. There are two approaches (the script `prepare_for_webodm.py` handles both):

### Option A: EXIF Tagging (Recommended)

This writes GPS coordinates directly into each JPG's EXIF metadata. WebODM automatically reads EXIF GPS — no extra configuration needed.

```bash
# Install dependencies (in your sylva venv)
cd ~/sylva
source venv/bin/activate
pip install piexif Pillow

# Run the prep script
# --images-dir should point to where your 230 JPGs live
python prepare_for_webodm.py \
    --csv log.csv \
    --json dataset.json \
    --images-dir /path/to/your/flight_images/ \
    --output-dir ./webodm_upload
```

This creates `./webodm_upload/` containing EXIF-tagged copies of all 230 images. The originals are not modified.

### Option B: geo.txt File (Fallback)

If you don't want to modify images or can't install piexif, generate a `geo.txt` file instead:

```bash
python prepare_for_webodm.py \
    --csv log.csv \
    --json dataset.json \
    --output-dir ./webodm_upload \
    --geo-only
```

The generated `geo.txt` looks like this:

```
EPSG:4326
0000000000.jpg -78.6627156 35.7686363 10.235 63.5455 1.0722 -1.4255
0000000001.jpg -78.6627040 35.7686422 10.201 63.9769 2.3569 0.5719
...
```

Format: `filename longitude latitude altitude yaw pitch roll` (angles in degrees).

**Important:** When using geo.txt, you must upload it alongside your images in WebODM. WebODM auto-detects a file named `geo.txt` in the upload.

### Fixing the Image Paths

Your CSV has Raspberry Pi paths like `/home/sylva0/on_board/logs/images/0000000000.jpg`. The script automatically extracts just the filename (`0000000000.jpg`). Make sure your actual JPG files on your local machine are named the same way. If they're in a different naming scheme, you'll need to rename them to match.

---

## Part 3: Process in WebODM

### 3A. Create a Project

1. Open WebODM at `http://localhost:8000`
2. Click **"Add Project"**
3. Name it something like `Sylva Flight 1`
4. Click **"Create Project"**

### 3B. Upload Images

1. Inside your new project, click **"Select Images and GCP"**
2. Navigate to your `./webodm_upload/` directory
3. Select **all 230 JPG files**
4. If using geo.txt (Option B): also select the `geo.txt` file in the same upload
5. Click **"Review"**

### 3C. Set Processing Options

Before clicking "Start Processing", click **"Edit"** next to the task options. These are the recommended settings for your flight:

**For a standard orthomosaic run:**

| Option | Value | Why |
|--------|-------|-----|
| `dsm` | ✓ (checked) | Generates a Digital Surface Model alongside the ortho |
| `orthophoto-resolution` | 1 | 1 cm/pixel output resolution (your GSD supports this) |
| `feature-quality` | `high` | Better feature matching for dense canopy |
| `min-num-features` | 10000 | More features = better alignment under tree canopy |
| `matcher-type` | `flann` | Fast and accurate for large datasets |
| `depthmap-resolution` | 1000 | Good detail without blowing up memory |

**If you're low on RAM (< 16 GB):**

| Option | Value |
|--------|-------|
| `feature-quality` | `medium` |
| `orthophoto-resolution` | 2 |
| `depthmap-resolution` | 640 |
| `max-concurrency` | 2 |
| `split` | 100 |
| `split-overlap` | 50 |

**If you want maximum quality (32+ GB RAM, patient):**

| Option | Value |
|--------|-------|
| `feature-quality` | `ultra` |
| `orthophoto-resolution` | 0.5 |
| `pc-quality` | `high` |
| `mesh-octree-depth` | 12 |

### 3D. Start Processing

Click **"Start Processing"**. For 230 images at 16MP on a typical machine:
- Low settings: ~20-40 minutes
- Standard settings: ~1-2 hours
- Ultra settings: ~3-5 hours

You can monitor progress in the WebODM dashboard. The stages are: dataset → split-merge (if applicable) → opensfm (feature matching + alignment) → openmvs (dense reconstruction) → odm_filterpoints → odm_meshing → odm_texturing → odm_georeferencing → odm_orthophoto → odm_dem.

### 3E. Download the GeoTIFF

Once processing completes:

1. Click **"View Map"** to see the 2D orthomosaic preview
2. Click **"Download Assets"** on the task card
3. Select **"Orthophoto (GeoTIFF)"** — this is your `odm_orthophoto.tif`

Other useful downloads:
- **All Assets** — gets everything (ortho, DSM, point cloud, 3D model, report)
- **Surface Model (GeoTIFF)** — the DSM, useful for height analysis

The GeoTIFF will be in UTM projection (WebODM auto-selects the correct UTM zone — for Raleigh, NC that's EPSG:32617, UTM Zone 17N).

---

## Part 4: Verify in QGIS

Open the GeoTIFF in QGIS to confirm it's georeferenced correctly:

```bash
# If you don't have QGIS installed:
sudo apt install qgis
```

1. Open QGIS
2. Drag `odm_orthophoto.tif` onto the map canvas
3. Right-click the layer → **Properties → Information** tab
4. Confirm the CRS is something like `EPSG:32617` (UTM Zone 17N)
5. Add a basemap (Plugins → QuickMapServices → OSM Standard) to verify alignment

You can also verify from the command line:

```bash
# Check GeoTIFF metadata
gdalinfo odm_orthophoto.tif | head -30
```

This should show the coordinate system, pixel size (GSD), and bounding box in UTM coordinates.

---

## Part 5: Feed Into DeepForest Pipeline

Once you have the GeoTIFF, it plugs directly into your existing Sylva DeepForest pipeline:

```bash
cd ~/sylva
source venv/bin/activate

python sylva_deepforest_pipeline.py \
    --ortho odm_orthophoto.tif \
    --max-tiles 5 \
    --annotate \
    --score-thresh 0.3
```

The pipeline will tile the orthomosaic, detect tree crowns, and output georeferenced per-tree data (crown diameter, VARI, ExG, RGB stats) — exactly as it did with the coworker's reference orthomosaic.

---

## Troubleshooting

**"Not enough overlap" / poor alignment:**
Your flight is at ~10m AGL which is very low. If the overlap between consecutive frames is less than ~60%, WebODM may struggle to find matching features. Check by looking at consecutive images — do they share significant visual overlap? If not, try increasing `min-num-features` to 15000 or 20000.

**Out of memory during processing:**
Use the low-RAM settings from Part 3C. You can also try `--split 100 --split-overlap 50` which processes the dataset in chunks.

**Processing fails at the texturing stage:**
Try adding `--use-3dmesh` or reducing `--mesh-octree-depth` to 10. Texturing is the most memory-intensive stage.

**GeoTIFF has black borders or warped edges:**
This is normal — the orthomosaic covers the convex hull of your flight path, and areas outside the actual imagery are filled with black (nodata). You can crop it in QGIS or with GDAL:
```bash
# Crop to a bounding box (adjust coordinates to your area)
gdal_translate -projwin <ulx> <uly> <lrx> <lry> odm_orthophoto.tif cropped_ortho.tif
```

**geo.txt not being detected:**
The file must be named exactly `geo.txt` (lowercase) and uploaded alongside the images in the same task upload. WebODM auto-detects it by filename.

**Images are very large and upload is slow:**
You can resize before uploading, but this will reduce your output resolution. For 16MP images, WebODM handles them fine — the bottleneck is processing, not upload.

---

## Quick Reference: File Flow

```
Raspberry Pi (flight)
  └── 230x JPG images + log.csv + dataset.json
        │
        ▼
  prepare_for_webodm.py
        │
        ├── geo.txt (or EXIF-tagged images)
        │
        ▼
  WebODM (Docker, localhost:8000)
        │
        ├── odm_orthophoto.tif  ← YOUR GEOTIFF
        ├── odm_dem/dsm.tif     ← Digital Surface Model
        ├── odm_report/         ← Processing report
        └── textured_model/     ← 3D mesh
              │
              ▼
  sylva_deepforest_pipeline.py
        │
        └── Per-tree detections (CSV, JSON, annotated tiles)
```
