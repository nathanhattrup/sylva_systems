# sylva_systems
Sylva Systems GitHub Repo

Here's the full Sylva pipeline end to end:
1. Drone Flight (Hardware)
The drone flies a predetermined path through the tree farm, recording video from front and back cameras. It logs GPS telemetry (lat/lon/altitude/heading) synced to timestamps alongside the video. All data is stored on an onboard hard drive — no live processing happens on the drone.
2. Workstream A — Tree Detection & Segmentation (where you are now)
Post-flight, the video gets pulled off the drone and fed into the detection pipeline. GroundingDINO + SAHI finds every tree in every frame and draws bounding boxes. SAM2 then converts those boxes into pixel-perfect segmentation masks. Output: per-frame JSON files with bounding box coordinates and confidence scores, plus annotated frames/video.
3. Workstream B — Multi-Object Tracking (ByteTrack)
The per-frame detections from Workstream A are just isolated snapshots — they don't know that "tree #3 in frame 100" is the same tree as "tree #5 in frame 101." ByteTrack solves this by tracking trees across consecutive frames, assigning each tree a persistent ID that follows it through the video. Output: the same detection data but now with consistent tree IDs across all frames.
4. Workstream C — Geolocation (GPS Projection)
Takes each tracked tree's pixel coordinates and maps them to real-world GPS coordinates using the drone's telemetry data (position, altitude, camera angle). Since the same tree appears in multiple frames with slightly different GPS readings, the system averages those readings into one precise location per tree. Output: a georeferenced map of every tree on the farm with its lat/lon.
5. Workstream D — Health Assessment (Custom ML)
The final layer. Uses the segmentation masks and geolocated tree data to assess each tree's health — detecting broken branches, disease, discoloration, canopy thinning, storm damage, etc. This is the part that requires custom model training (likely on Google Colab with a GPU) since there's no off-the-shelf model for "is this loblolly pine sick." Output: per-tree health reports tied to GPS locations.
The end product is a farmer plugging in a hard drive after a flight and getting back a map of their entire farm with every tree located and flagged with any health issues — replacing days of manual ATV inspection.