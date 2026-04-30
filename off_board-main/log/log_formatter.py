import pandas as pd
import json

FLIGHT_NAME = "flight_2"
CAMERA_MODEL = {
    "image_width": 1980,
    "image_height": 1080,
    "fx": 3200.0,
    "fy": 3200.0,
    "cx": 2328.0,
    "cy": 1748.0
}

def build_dataset_json(frames):
    return {
        "dataset_name": FLIGHT_NAME,
        "crs": "EPSG:4326",
        "alt_reference": "AGL",
        "camera_model": CAMERA_MODEL,
        "frames": frames
    }

def main(log_path, output_path):
    df = pd.read_csv(log_path)
    frames = []
    for _, row in df.iterrows():
        frame = {
            "timestamp": row["img_timestamp"],
            "image_path": row["img_path"][-21:],
            "lat": row["lat_deg"],
            "lon": row["lon_deg"],
            "alt": row["alt_m"],
            "roll": row["roll_rad"],
            "pitch": row["pitch_rad"],
            "yaw": row["yaw_rad"]
        }
        frames.append(frame)

    dataset_json = build_dataset_json(frames)
    with open(output_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

if __name__ == "__main__":
    log_path = f"flights/{FLIGHT_NAME}/log.csv"
    output_path = f"flights/{FLIGHT_NAME}/dataset.json"
    main(log_path, output_path)