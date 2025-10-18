# Drone AI Face Follower

Lightweight face-tracking demo that makes a DJI Tello drone (or a local webcam) follow a specific person. The app generates a reference embedding from photos stored in `assets/target_images`, detects the face in live video, and annotates the stream in real time.

## Requirements
- Python 3.9 or newer
- Packages: `opencv-python`, `face_recognition`, `numpy`, `djitellopy`
- DJI Tello drone connected to the host machine (only for `drone` mode)

Install the dependencies with pip:

```bash
pip install opencv-python face_recognition numpy djitellopy
```

## Prepare the Target Face
1. Save one or more clear photos of the person in `assets/target_images`.  
   The script averages the embeddings of all valid images in this folder.
2. Remove or rename any non-face images; they are skipped but slow down startup.

## Run the Tracker
Use the module entry point and choose the video source:

```bash
python -m src.main --target "Alice" --video-source webcam
python -m src.main --target "Alice" --video-source drone
```

- `--target` is the label displayed when the face is recognized.
- `--video-source` accepts `webcam` for the built-in camera or `drone` for the Tello feed.

Press `q` to close the video preview. When running with the drone, ensure the Tello stream is live before starting and land the drone manually if needed.

