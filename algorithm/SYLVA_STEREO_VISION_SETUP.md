# Sylva — Stereo Vision Setup Guide: Dual Pi Zero 2W + Arducam IMX519

**Target setup:** 2× Raspberry Pi Zero 2W, each with 1× Arducam 16MP IMX519, mounted on drone for overhead stereo depth estimation of tree canopies.

---

## Architecture Decision: Why Two Separate Pi Zeros (Not a Stereo HAT)

You might be tempted to use Arducam's Stereo Camera HAT or Camarray to connect both cameras to a single Pi. **Don't do this for your use case.** Here's why:

| Approach | Problem |
|----------|---------|
| Arducam Stereo HAT on single Pi | Only one camera active at a time (alternating capture). Not true simultaneous stereo. |
| Arducam Camarray (quad HAT) on single Pi | Requires Pi 3/4/5 (not Zero 2W). Splits resolution — each camera gets 1/4 the output. |
| Single Pi Zero 2W + multiplexer | Pi Zero only has 1 CSI port with 2 MIPI data lanes. Insufficient bandwidth for two 16MP sensors simultaneously. |
| **2× Pi Zero 2W, one camera each** | **Each Pi captures full 16MP independently. Sync via GPIO or network trigger. This is your best path.** |

The dual-Pi approach is well-proven — MIT researchers synchronized 16 Pi Zeros for multi-view capture using GPIO triggering. Your setup is the same concept with just two units.

---

## Hardware Setup

### Bill of Materials (Stereo Rig)

- 2× Raspberry Pi Zero 2W
- 2× Arducam IMX519 16MP camera module (with 22-pin FPC cable for Pi Zero)
- 1× Rigid mounting bracket (3D-printed or aluminum) to fix cameras at known baseline
- 2× MicroSD cards (32GB+ recommended)
- Jumper wire for GPIO sync trigger (1 wire + ground between the two Pis)
- USB power for both Pis (can share drone's power bus via BEC)

### Baseline Distance: How Far Apart?

This is the most critical design parameter. The relationship between baseline (b), focal length (f), and depth accuracy follows:

```
depth_error ≈ (Z² × disparity_error) / (b × f_pixels)
```

Where Z is the distance to the scene (your flight altitude above canopy).

**Rule of thumb:** baseline-to-depth ratio should be ~1:10 to 1:20 for reasonable depth accuracy.

| Flight altitude (AGL above canopy) | Recommended baseline | Expected depth precision |
|-------------------------------------|---------------------|-------------------------|
| 15 m (~50 ft) | 0.75 – 1.5 m | ±0.5 – 1.0 m |
| 25 m (~80 ft) | 1.25 – 2.5 m | ±0.8 – 1.5 m |
| 50 m (~160 ft) | 2.5 – 5.0 m | ±2.0 – 4.0 m |
| 76 m (~250 ft, your coworker's alt) | 3.8 – 7.5 m | ±3.0 – 6.0 m |

**For tree height estimation (6-7m trees):** You need depth precision of roughly ±0.5m to distinguish canopy tops from ground. At 25m altitude, a baseline of ~1.5m gives you this. At 50m, you'd need ~3m baseline, which is physically challenging on a single drone.

**Practical recommendation for your drone:** Start with a **30-50 cm baseline** for initial prototyping (easy to mount), then increase to **1.0-1.5 m** for production. Lower flight altitude compensates for smaller baseline.

### Mounting the Cameras

The cameras must be **rigidly fixed** relative to each other. Any flex or vibration between them during flight invalidates your calibration.

**Mounting requirements:**
- Both cameras point straight down (nadir), parallel to each other
- Cameras separated horizontally along one axis (left-right relative to flight direction is standard)
- No rotation between cameras (both image sensors in the same plane)
- Vibration dampening between drone frame and camera mount (but cameras must be rigid relative to *each other*)

A simple approach: mount both Pis and cameras on a single rigid aluminum or carbon fiber plate, then vibration-isolate that plate from the drone.

---

## Software Setup (On Each Pi Zero 2W)

You've already got the Arducam IMX519 driver working on one Pi. Repeat the same setup on the second Pi.

### Prerequisites (per Pi)

```bash
# Install Arducam IMX519 driver (you've done this before)
wget -O install_pivariety_pkgs.sh \
  https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh
chmod +x install_pivariety_pkgs.sh
./install_pivariety_pkgs.sh -p libcamera_dev
./install_pivariety_pkgs.sh -p libcamera_apps

# Edit config.txt to enable IMX519
# For Pi Zero 2W on Bookworm:
sudo nano /boot/firmware/config.txt
# Add: dtoverlay=imx519

sudo reboot

# Verify camera works
libcamera-hello --list-cameras
libcamera-still -o test.jpg
```

### Install OpenCV and Dependencies

```bash
# On each Pi Zero 2W
sudo apt update
sudo apt install python3-opencv python3-numpy python3-pip

# Verify
python3 -c "import cv2; print(cv2.__version__)"
```

Note: Building OpenCV from source on Pi Zero 2W takes hours. Use the apt package unless you need a specific version.

---

## Step 1: Synchronizing Capture Between Two Pis

The cameras must capture frames at the same instant. Even a few milliseconds of offset causes depth errors when the drone is moving. There are three approaches, from simplest to most precise:

### Option A: Network Trigger (Simplest, ~10-50ms sync)

One Pi acts as the "leader," the other as "follower." The leader sends a UDP packet to trigger capture on both.

This is good enough for: still captures, slow flight, initial prototyping.

**leader_capture.py** (runs on Pi #1):

```python
"""
Sylva Stereo — Leader capture node.
Triggers synchronized capture on both Pis via UDP broadcast.
"""
import socket
import time
import subprocess
import threading
import json
from datetime import datetime

FOLLOWER_IP = "192.168.1.102"  # Set to your follower Pi's IP
TRIGGER_PORT = 5555
CAPTURE_DIR = "/home/pi/stereo_captures/left"
FRAME_INTERVAL = 2.0  # seconds between captures

def send_trigger(sock, frame_id):
    """Send capture trigger to follower."""
    msg = json.dumps({"cmd": "capture", "frame_id": frame_id, "timestamp": time.time()})
    sock.sendto(msg.encode(), (FOLLOWER_IP, TRIGGER_PORT))

def capture_image(frame_id):
    """Capture image using libcamera-still."""
    filename = f"{CAPTURE_DIR}/frame_{frame_id:06d}.jpg"
    subprocess.run([
        "libcamera-still",
        "-o", filename,
        "--width", "4656",
        "--height", "3496",
        "--nopreview",
        "--immediate",
        "-t", "1"  # minimal timeout
    ], capture_output=True)
    return filename

if __name__ == "__main__":
    import os
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    frame_id = 0

    print("Stereo Leader — starting capture loop")
    while True:
        # Send trigger to follower FIRST, then capture locally
        # This partially compensates for network latency
        send_trigger(sock, frame_id)
        capture_image(frame_id)

        print(f"Frame {frame_id} captured at {datetime.now().isoformat()}")
        frame_id += 1
        time.sleep(FRAME_INTERVAL)
```

**follower_capture.py** (runs on Pi #2):

```python
"""
Sylva Stereo — Follower capture node.
Listens for UDP triggers from leader and captures immediately.
"""
import socket
import subprocess
import json
import os

TRIGGER_PORT = 5555
CAPTURE_DIR = "/home/pi/stereo_captures/right"

def capture_image(frame_id):
    filename = f"{CAPTURE_DIR}/frame_{frame_id:06d}.jpg"
    subprocess.run([
        "libcamera-still",
        "-o", filename,
        "--width", "4656",
        "--height", "3496",
        "--nopreview",
        "--immediate",
        "-t", "1"
    ], capture_output=True)
    return filename

if __name__ == "__main__":
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", TRIGGER_PORT))

    print("Stereo Follower — waiting for triggers")
    while True:
        data, addr = sock.recvfrom(1024)
        msg = json.loads(data.decode())

        if msg["cmd"] == "capture":
            capture_image(msg["frame_id"])
            print(f"Frame {msg['frame_id']} captured (trigger delay: "
                  f"{(time.time() - msg['timestamp'])*1000:.1f}ms)")
```

### Option B: GPIO Hardware Trigger (~1ms sync) ⭐ RECOMMENDED

Connect a GPIO pin from the leader Pi to the follower Pi. A rising edge triggers capture on both simultaneously.

**Wiring:**
```
Leader Pi GPIO 17  ──────────  Follower Pi GPIO 17
Leader Pi GND      ──────────  Follower Pi GND
```

**gpio_sync_capture.py** (runs on BOTH Pis, with --role flag):

```python
"""
Sylva Stereo — GPIO-synchronized capture.
Run on both Pis:
  Leader:   python3 gpio_sync_capture.py --role leader --side left
  Follower: python3 gpio_sync_capture.py --role follower --side right
"""
import argparse
import os
import time
import subprocess

try:
    import RPi.GPIO as GPIO
except ImportError:
    # For testing off-Pi
    print("WARNING: RPi.GPIO not available, running in simulation mode")
    GPIO = None

SYNC_PIN = 17
CAPTURE_DIR_BASE = "/home/pi/stereo_captures"

def setup_gpio(role):
    if GPIO is None:
        return
    GPIO.setmode(GPIO.BCM)
    if role == "leader":
        GPIO.setup(SYNC_PIN, GPIO.OUT, initial=GPIO.LOW)
    else:
        GPIO.setup(SYNC_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def capture_still(side, frame_id):
    """Capture a single still image."""
    capture_dir = f"{CAPTURE_DIR_BASE}/{side}"
    os.makedirs(capture_dir, exist_ok=True)
    filename = f"{capture_dir}/frame_{frame_id:06d}.jpg"
    subprocess.run([
        "libcamera-still",
        "-o", filename,
        "--width", "4656",
        "--height", "3496",
        "--nopreview",
        "--immediate",
        "-t", "1"
    ], capture_output=True)
    return filename

def capture_video_segment(side, segment_id, duration_sec=10):
    """Capture a video segment (alternative to stills for photogrammetry)."""
    capture_dir = f"{CAPTURE_DIR_BASE}/{side}"
    os.makedirs(capture_dir, exist_ok=True)
    filename = f"{capture_dir}/segment_{segment_id:04d}.h264"
    subprocess.run([
        "libcamera-vid",
        "-o", filename,
        "--width", "1920",
        "--height", "1080",
        "--framerate", "30",
        "--nopreview",
        "-t", str(duration_sec * 1000)
    ], capture_output=True)
    return filename

def run_leader(side, interval=2.0, num_frames=100):
    """Leader: pulse GPIO to trigger both cameras."""
    setup_gpio("leader")
    frame_id = 0
    print(f"Leader ({side}) — capturing {num_frames} frames at {interval}s intervals")

    for frame_id in range(num_frames):
        # Rising edge triggers follower
        if GPIO:
            GPIO.output(SYNC_PIN, GPIO.HIGH)

        # Capture locally
        fname = capture_still(side, frame_id)

        # Reset trigger
        if GPIO:
            time.sleep(0.01)  # 10ms pulse
            GPIO.output(SYNC_PIN, GPIO.LOW)

        print(f"[{side}] Frame {frame_id}: {fname}")
        time.sleep(interval)

    if GPIO:
        GPIO.cleanup()

def run_follower(side):
    """Follower: wait for GPIO trigger, then capture."""
    setup_gpio("follower")
    frame_id = 0
    print(f"Follower ({side}) — waiting for GPIO triggers on pin {SYNC_PIN}")

    try:
        while True:
            if GPIO:
                # Block until rising edge
                GPIO.wait_for_edge(SYNC_PIN, GPIO.RISING)

            fname = capture_still(side, frame_id)
            print(f"[{side}] Frame {frame_id}: {fname}")
            frame_id += 1
    except KeyboardInterrupt:
        pass
    finally:
        if GPIO:
            GPIO.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sylva stereo sync capture")
    parser.add_argument("--role", choices=["leader", "follower"], required=True)
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between captures (leader only)")
    parser.add_argument("--num-frames", type=int, default=100)
    args = parser.parse_args()

    if args.role == "leader":
        run_leader(args.side, args.interval, args.num_frames)
    else:
        run_follower(args.side)
```

### Option C: Video Mode + Post-Hoc Sync (For Continuous Capture)

For continuous overhead flight (like your coworker), you may want both cameras recording video simultaneously rather than taking individual stills. In this case:

1. Both Pis record video with `libcamera-vid` simultaneously
2. Use timestamps embedded in each frame to align them in post-processing
3. A GPIO "start" signal kicks off both recordings at the same time

This is the most practical for your flight scenario. Stills are better for calibration and photogrammetry; video is better for continuous flight coverage.

---

## Step 2: Stereo Calibration

Before you can compute depth, you must calibrate both cameras individually (intrinsics) and then calibrate the stereo pair (extrinsics). This is a one-time process that you redo only if you change the physical mounting.

### Print a Calibration Pattern

- Use a **9×6 checkerboard** (inner corners), with squares of known physical size
- **Critical:** Print it on rigid, flat material (foam board, aluminum composite). Paper taped to cardboard warps and kills calibration accuracy. Spend $10-20 on a proper printed board.
- Square size: 25-30mm works well. Measure the actual printed size precisely with calipers.

### Capture Calibration Images

Take 15-20 synchronized stereo pairs of the checkerboard from different angles and distances. Both cameras must see the full checkerboard in every pair.

**Tips for good calibration images:**
- Show the board in all areas of the image (corners, edges, center)
- Vary the distance (0.5m to 2m)
- Tilt the board at different angles (but keep all corners visible)
- Good lighting, no reflections on the board surface
- Board must be fully visible in BOTH cameras for each pair

### Calibration Script

Save this as `stereo_calibrate.py` on your dev machine (not the Pis — run this on your Linux workstation with the captured images):

```python
"""
Sylva Stereo — Camera Calibration
Calibrates intrinsics for each camera, then stereo extrinsics.

Usage:
  1. Capture synced checkerboard image pairs (left/frame_000000.jpg, right/frame_000000.jpg)
  2. Run: python3 stereo_calibrate.py --left-dir captures/left --right-dir captures/right

Outputs calibration parameters to stereo_calibration.npz
"""
import cv2
import numpy as np
import glob
import argparse
import json
from pathlib import Path

# Checkerboard configuration — CHANGE THESE to match your board
CHECKERBOARD = (9, 6)       # inner corners (columns, rows)
SQUARE_SIZE_MM = 25.0       # physical size of each square in mm

# Termination criteria for corner sub-pixel refinement
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def find_checkerboard_points(image_dir, checkerboard=CHECKERBOARD):
    """Find checkerboard corners in all images in a directory."""
    images = sorted(glob.glob(f"{image_dir}/*.jpg"))
    if not images:
        images = sorted(glob.glob(f"{image_dir}/*.png"))

    print(f"Found {len(images)} images in {image_dir}")

    objpoints = []  # 3D points in world coordinates
    imgpoints = []  # 2D points in image plane
    used_images = []
    img_size = None

    # Prepare object points: (0,0,0), (25,0,0), (50,0,0), ... in mm
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = gray.shape[::-1]  # (width, height)

        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

        if ret:
            # Sub-pixel refinement
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            objpoints.append(objp)
            imgpoints.append(corners)
            used_images.append(img_path)
            print(f"  ✓ {Path(img_path).name} — found {checkerboard[0]}×{checkerboard[1]} corners")
        else:
            print(f"  ✗ {Path(img_path).name} — checkerboard not detected")

    print(f"  Used {len(used_images)}/{len(images)} images")
    return objpoints, imgpoints, img_size, used_images


def calibrate_single_camera(image_dir, name="camera"):
    """Calibrate a single camera's intrinsic parameters."""
    print(f"\n{'='*60}")
    print(f"Calibrating {name}")
    print(f"{'='*60}")

    objpoints, imgpoints, img_size, used = find_checkerboard_points(image_dir)

    if len(objpoints) < 10:
        print(f"WARNING: Only {len(objpoints)} valid images. Need at least 10 for good calibration.")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    print(f"\n  RMSE: {ret:.4f}")
    print(f"  Camera matrix:\n{mtx}")
    print(f"  Distortion coefficients: {dist.ravel()}")

    if ret > 0.5:
        print("  WARNING: RMSE > 0.5 — calibration may be poor. Retake images.")
    elif ret < 0.25:
        print("  Excellent calibration quality.")

    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, img_size


def calibrate_stereo(left_dir, right_dir, output_file="stereo_calibration.npz"):
    """Full stereo calibration pipeline."""

    # Step 1: Calibrate each camera individually
    ret_l, mtx_l, dist_l, _, _, objpoints_l, imgpoints_l, img_size = \
        calibrate_single_camera(left_dir, "Left Camera")

    ret_r, mtx_r, dist_r, _, _, objpoints_r, imgpoints_r, _ = \
        calibrate_single_camera(right_dir, "Right Camera")

    # Step 2: Find matching pairs (both cameras detected checkerboard)
    # Assuming filenames match between left and right directories
    left_images = sorted(glob.glob(f"{left_dir}/*.jpg") + glob.glob(f"{left_dir}/*.png"))
    right_images = sorted(glob.glob(f"{right_dir}/*.jpg") + glob.glob(f"{right_dir}/*.png"))

    objpoints_stereo = []
    imgpoints_left = []
    imgpoints_right = []

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    print(f"\n{'='*60}")
    print("Stereo Calibration — finding matched pairs")
    print(f"{'='*60}")

    for l_path, r_path in zip(left_images, right_images):
        img_l = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)

        ret_l, corners_l = cv2.findChessboardCorners(img_l, CHECKERBOARD, None)
        ret_r, corners_r = cv2.findChessboardCorners(img_r, CHECKERBOARD, None)

        if ret_l and ret_r:
            corners_l = cv2.cornerSubPix(img_l, corners_l, (11, 11), (-1, -1), CRITERIA)
            corners_r = cv2.cornerSubPix(img_r, corners_r, (11, 11), (-1, -1), CRITERIA)

            objpoints_stereo.append(objp)
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)
            print(f"  ✓ Pair: {Path(l_path).name} + {Path(r_path).name}")
        else:
            print(f"  ✗ Skipped: {Path(l_path).name} (L:{ret_l}, R:{ret_r})")

    print(f"\n  Matched pairs: {len(objpoints_stereo)}")

    if len(objpoints_stereo) < 10:
        print("WARNING: < 10 matched pairs. Stereo calibration may be unreliable.")

    # Step 3: Stereo calibration (refines both intrinsics + computes extrinsics)
    flags = cv2.CALIB_FIX_INTRINSIC  # Use the individual calibrations, just find R and T

    ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints_stereo,
        imgpoints_left,
        imgpoints_right,
        mtx_l, dist_l,
        mtx_r, dist_r,
        img_size,
        criteria=CRITERIA,
        flags=flags
    )

    print(f"\n{'='*60}")
    print("Stereo Calibration Results")
    print(f"{'='*60}")
    print(f"  Stereo RMSE: {ret_stereo:.4f}")
    print(f"  Rotation matrix R:\n{R}")
    print(f"  Translation vector T (mm): {T.ravel()}")

    # The first element of T is the baseline (should be negative, and close to your measured value)
    baseline_mm = abs(T[0, 0])
    print(f"\n  Computed baseline: {baseline_mm:.1f} mm ({baseline_mm/10:.1f} cm)")
    print(f"  (Verify this matches your physical measurement!)")

    # Step 4: Stereo rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r,
        img_size, R, T,
        alpha=0  # 0 = crop to valid pixels only; 1 = keep all pixels
    )

    # Save everything
    np.savez(output_file,
        # Individual camera params
        mtx_l=mtx_l, dist_l=dist_l,
        mtx_r=mtx_r, dist_r=dist_r,
        # Stereo extrinsics
        R=R, T=T, E=E, F=F,
        # Rectification params
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        roi1=roi1, roi2=roi2,
        # Metadata
        img_size=np.array(img_size),
        baseline_mm=np.array([baseline_mm]),
        rmse_stereo=np.array([ret_stereo])
    )

    print(f"\n  Calibration saved to: {output_file}")
    print(f"  Load with: data = np.load('{output_file}')")

    return ret_stereo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sylva stereo camera calibration")
    parser.add_argument("--left-dir", required=True, help="Directory of left camera images")
    parser.add_argument("--right-dir", required=True, help="Directory of right camera images")
    parser.add_argument("--output", default="stereo_calibration.npz", help="Output file")
    parser.add_argument("--square-size", type=float, default=25.0, help="Checkerboard square size in mm")
    args = parser.parse_args()

    SQUARE_SIZE_MM = args.square_size
    calibrate_stereo(args.left_dir, args.right_dir, args.output)
```

---

## Step 3: Computing Depth Maps

Once calibrated, you can compute depth from any synchronized stereo pair.

Save as `stereo_depth.py`:

```python
"""
Sylva Stereo — Depth Map Computation
Takes a calibrated stereo pair and produces a depth map.

Usage:
  python3 stereo_depth.py --calib stereo_calibration.npz \
      --left captures/left/frame_000000.jpg \
      --right captures/right/frame_000000.jpg
"""
import cv2
import numpy as np
import argparse
from pathlib import Path


def load_calibration(calib_file):
    """Load stereo calibration parameters."""
    data = np.load(calib_file)
    return {
        'mtx_l': data['mtx_l'], 'dist_l': data['dist_l'],
        'mtx_r': data['mtx_r'], 'dist_r': data['dist_r'],
        'R1': data['R1'], 'R2': data['R2'],
        'P1': data['P1'], 'P2': data['P2'],
        'Q': data['Q'],
        'img_size': tuple(data['img_size']),
        'baseline_mm': float(data['baseline_mm'][0]),
    }


def rectify_pair(img_l, img_r, calib):
    """Undistort and rectify a stereo image pair."""
    h, w = img_l.shape[:2]

    # Compute rectification maps (could cache these for performance)
    map_l1, map_l2 = cv2.initUndistortRectifyMap(
        calib['mtx_l'], calib['dist_l'], calib['R1'], calib['P1'],
        calib['img_size'], cv2.CV_32FC1
    )
    map_r1, map_r2 = cv2.initUndistortRectifyMap(
        calib['mtx_r'], calib['dist_r'], calib['R2'], calib['P2'],
        calib['img_size'], cv2.CV_32FC1
    )

    rect_l = cv2.remap(img_l, map_l1, map_l2, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map_r1, map_r2, cv2.INTER_LINEAR)

    return rect_l, rect_r


def compute_depth_map(rect_l, rect_r, calib, method="sgbm"):
    """Compute disparity and convert to depth in meters."""

    # Convert to grayscale for stereo matching
    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY) if len(rect_l.shape) == 3 else rect_l
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY) if len(rect_r.shape) == 3 else rect_r

    if method == "sgbm":
        # SemiGlobal Block Matching — better quality, slower
        # Tune these parameters for your scene
        min_disp = 0
        num_disp = 128      # must be divisible by 16; increase for closer objects
        block_size = 5      # odd number, 3-11; smaller = more detail, more noise

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,    # smoothness penalty (lower = more detail)
            P2=32 * 3 * block_size**2,   # smoothness penalty (higher = smoother)
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # best quality
        )
    else:
        # StereoBM — faster, lower quality (good for prototyping)
        stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)

    # Compute disparity (in fixed-point: divide by 16 for actual disparity)
    disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

    # Optional: WLS filter for cleaner disparity (significantly improves results)
    try:
        right_matcher = cv2.ximgproc.createRightMatcher(stereo)
        disparity_r = right_matcher.compute(gray_r, gray_l).astype(np.float32) / 16.0

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
        wls_filter.setLambda(8000)
        wls_filter.setSigmaColor(1.5)
        disparity = wls_filter.filter(disparity, gray_l, disparity_map_right=disparity_r)
        print("  Applied WLS disparity filter (opencv-contrib available)")
    except AttributeError:
        print("  WLS filter not available (install opencv-contrib for better results)")

    # Convert disparity to depth using Q matrix from calibration
    # Q matrix reprojects disparity to 3D: [X, Y, Z, W] = Q * [x, y, disparity, 1]
    # depth_m = baseline_mm / disparity * focal_length_px  (simplified)
    points_3d = cv2.reprojectImageTo3D(disparity, calib['Q'])
    depth_map = points_3d[:, :, 2]  # Z coordinate = depth

    # Convert from mm to meters (since calibration was in mm)
    depth_map = depth_map / 1000.0

    # Mask invalid disparities
    mask = disparity > 0
    depth_map[~mask] = 0

    return disparity, depth_map, mask


def visualize_depth(depth_map, mask, output_path, max_depth_m=100):
    """Create a color-coded depth visualization."""
    # Clip to reasonable range
    vis = depth_map.copy()
    vis[~mask] = max_depth_m
    vis = np.clip(vis, 0, max_depth_m)

    # Normalize to 0-255
    vis_norm = ((vis / max_depth_m) * 255).astype(np.uint8)

    # Apply colormap (TURBO is great for depth)
    vis_color = cv2.applyColorMap(vis_norm, cv2.COLORMAP_TURBO)
    vis_color[~mask] = [0, 0, 0]  # Black for invalid regions

    cv2.imwrite(str(output_path), vis_color)
    return vis_color


def estimate_tree_heights(depth_map, mask, flight_altitude_m):
    """
    Rough tree height estimation from depth map.
    Tree height ≈ flight_altitude - depth_to_canopy_top

    This is a simplified version — the full pipeline would use
    your tree detection bounding boxes from Workstream A to
    isolate individual trees.
    """
    valid_depths = depth_map[mask & (depth_map > 0) & (depth_map < flight_altitude_m * 1.5)]

    if len(valid_depths) == 0:
        print("  No valid depth measurements")
        return None

    # Ground level ≈ maximum depth (furthest from camera = ground between trees)
    ground_depth = np.percentile(valid_depths, 95)

    # Canopy top ≈ minimum depth (closest to camera = tree tops)
    canopy_depth = np.percentile(valid_depths, 5)

    estimated_height = ground_depth - canopy_depth

    print(f"\n  Depth statistics:")
    print(f"    Min depth (canopy top): {canopy_depth:.1f} m")
    print(f"    Max depth (ground):     {ground_depth:.1f} m")
    print(f"    Estimated canopy height: {estimated_height:.1f} m")
    print(f"    Flight altitude:         {flight_altitude_m:.1f} m")

    return estimated_height


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sylva stereo depth computation")
    parser.add_argument("--calib", required=True, help="Calibration .npz file")
    parser.add_argument("--left", required=True, help="Left image path")
    parser.add_argument("--right", required=True, help="Right image path")
    parser.add_argument("--output-dir", default="outputs/stereo_depth")
    parser.add_argument("--altitude", type=float, default=25.0, help="Flight altitude above ground (m)")
    parser.add_argument("--method", choices=["sgbm", "bm"], default="sgbm")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration
    print("Loading calibration...")
    calib = load_calibration(args.calib)
    print(f"  Baseline: {calib['baseline_mm']:.1f} mm")
    print(f"  Image size: {calib['img_size']}")

    # Load images
    img_l = cv2.imread(args.left)
    img_r = cv2.imread(args.right)
    print(f"  Left:  {args.left} ({img_l.shape[1]}×{img_l.shape[0]})")
    print(f"  Right: {args.right} ({img_r.shape[1]}×{img_r.shape[0]})")

    # Rectify
    print("\nRectifying stereo pair...")
    rect_l, rect_r = rectify_pair(img_l, img_r, calib)

    # Save rectified pair (useful for debugging — epipolar lines should be horizontal)
    cv2.imwrite(str(output_dir / "rectified_left.jpg"), rect_l)
    cv2.imwrite(str(output_dir / "rectified_right.jpg"), rect_r)

    # Draw horizontal lines to verify rectification
    debug = np.hstack([rect_l, rect_r])
    for y in range(0, debug.shape[0], 50):
        cv2.line(debug, (0, y), (debug.shape[1], y), (0, 255, 0), 1)
    cv2.imwrite(str(output_dir / "rectification_check.jpg"), debug)
    print("  Saved rectification check (green lines should align features)")

    # Compute depth
    print(f"\nComputing depth map (method: {args.method})...")
    disparity, depth_map, mask = compute_depth_map(rect_l, rect_r, calib, args.method)

    # Visualize
    visualize_depth(depth_map, mask, output_dir / "depth_colormap.jpg")
    print(f"  Saved depth visualization")

    # Save raw depth as numpy array (for pipeline integration)
    np.save(str(output_dir / "depth_map.npy"), depth_map)
    print(f"  Saved raw depth map (depth_map.npy)")

    # Estimate tree heights
    estimate_tree_heights(depth_map, mask, args.altitude)

    print(f"\n✓ All outputs saved to {output_dir}/")
```

---

## Step 4: Integration with Sylva Detection Pipeline

The key insight: **depth data augments your existing Workstream A detection pipeline.** You don't replace GroundingDINO + SAM2 — you add a depth channel.

### Pipeline Integration Flow

```
Left Camera ─┐
             ├─── Stereo Depth ──→ depth_map (per-pixel meters)
Right Camera ─┘                          │
      │                                  │
      └── Left image ──→ SAHI + GroundingDINO ──→ tree bounding boxes
                                │                        │
                                └────────────────────────┘
                                         │
                                  For each detected tree:
                                    1. Get bbox from detection
                                    2. Extract depth values within bbox
                                    3. tree_height = ground_depth - median(depth_in_bbox)
                                    4. Store in per-tree JSON
```

### Per-Tree Output (extends your existing detection JSON)

```json
{
  "tree_id": 42,
  "tracker_id": 7,
  "bbox_xyxy": [1200, 800, 1350, 950],
  "confidence": 0.87,
  "depth_median_m": 18.3,
  "depth_min_m": 17.8,
  "depth_max_m": 19.1,
  "estimated_height_m": 6.2,
  "geo_location": {
    "lat": 35.7796,
    "lon": -78.6382,
    "elevation_m": 102.4
  }
}
```

---

## Tuning Guide for Aerial Stereo

### Stereo Matching Parameters

| Parameter | What it does | Aerial trees guidance |
|-----------|-------------|----------------------|
| `numDisparities` | Range of disparity search (must be ÷ 16) | Start with 128. If trees are very close (low alt), increase to 256. |
| `blockSize` | Window size for matching | 5-7 for textured canopy (good). 11-15 if canopy is smooth/uniform. |
| `P1` / `P2` (SGBM) | Smoothness penalties | Lower P1/P2 for more detail. Higher for smoother (less noisy) depth. |
| `uniquenessRatio` | Reject ambiguous matches | 10 is standard. Increase to 15 if you get noisy depth on uniform canopy. |
| `speckleWindowSize` | Remove small noise blobs | 100-200 for clean results. |

### Common Issues

**Problem: Depth map is mostly black/invalid**
→ Baseline may be too small for your altitude. Increase baseline or fly lower.
→ Check rectification: horizontal lines in rectification_check.jpg should align corresponding features across both images.

**Problem: Depth map is noisy on uniform canopy**
→ Stereo matching needs texture. Uniform green canopy is hard. Increase blockSize. The WLS filter helps significantly.
→ Consider flying when there's partial shadow (more texture contrast).

**Problem: Computed baseline doesn't match physical measurement**
→ Your checkerboard square size is wrong, or the board isn't flat. Remeasure carefully and recalibrate.

---

## Milestone Checklist

- [ ] **M1:** Both Pi Zero 2Ws imaging independently with IMX519
- [ ] **M2:** GPIO sync wiring done, simultaneous capture verified (compare timestamps)
- [ ] **M3:** Calibration checkerboard printed on rigid material
- [ ] **M4:** 15+ stereo calibration pairs captured, `stereo_calibrate.py` runs with RMSE < 0.3
- [ ] **M5:** Computed baseline matches physical measurement (within 5%)
- [ ] **M6:** Depth map generated from test stereo pair, rectification lines look correct
- [ ] **M7:** Depth map generated from actual tree canopy imagery
- [ ] **M8:** Tree height estimation integrated with Workstream A detection output
- [ ] **M9:** Depth data feeds into per-tree JSON (Workstream C ready)

---

## What's Next

Once stereo depth works, it feeds directly into your pipeline:

1. **Workstream A** detects + segments trees from the left camera image
2. **Stereo depth** provides height for each detected tree
3. **Workstream B (ByteTrack)** tracks trees across frames
4. **Workstream C (GPS geolocation)** maps each tracked tree to real-world coordinates, now with height data
5. **WebODM** can stitch the left camera frames into an orthomosaic for the map overlay on your website

The stereo depth essentially gives you what your coworker gets from LiDAR — a canopy height model (CHM) — but from passive imaging instead of active sensing. It won't be as precise as LiDAR, but it's dramatically cheaper and gives you enough resolution to flag trees that are shorter than their neighbors (potential health issues) or have irregular canopy shape.
