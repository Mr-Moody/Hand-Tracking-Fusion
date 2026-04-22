# Hand Tracking Fusion

Multi-camera 3D hand tracking using MediaPipe for 2D/3D landmark detection and an Extended Kalman Filter (EKF) to fuse observations across cameras via a partial observation matrix.

## How it works

Each camera runs MediaPipe `HandLandmarker` independently, producing 21 landmarks in metric 3D space (wrist-centred). The EKF maintains a joint state vector of 21 × 6 values (position + velocity per joint). Each frame:

1. **Predict** — propagates the state forward using a constant-velocity motion model.
2. **Update (per camera)** — each camera's visible landmarks contribute a partial observation. The H matrix is built only for joints with a high visibility score, so occluded joints in one view are still estimated from the other camera and the motion prior.

## Repository layout

```
.
├── main.py                  # Entry point — argparse handler and main loop
├── models/
│   └── hand_landmarker.task # MediaPipe model file (download separately, see below)
└── src/
    ├── camera.py            # CameraSource — wraps cv2.VideoCapture
    ├── hand_detector.py     # HandDetector + HandDetection dataclass
    ├── joint_hand_ekf.py    # JointHandEKF + build_partial_H
    ├── fusion.py            # HandFusion — orchestrates predict/update across cameras
    └── visualiser.py        # HandVisualiser — drawing helpers
```

## Setup

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install mediapipe opencv-python numpy

# Download the MediaPipe hand landmark model
wget -P models/ https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Usage

```bash
# Single camera (laptop webcam)
python main.py --cam0 0

# Two cameras fused together
python main.py --cam0 0 --cam1 1

# Two cameras with both feeds shown side by side
python main.py --cam0 0 --cam1 1 --show-both

# Custom model path or frame rate
python main.py --cam0 0 --cam1 1 --model models/hand_landmarker.task --fps 30
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--cam0` | `0` | Primary camera index |
| `--cam1` | — | Secondary camera index (enables EKF fusion across two views) |
| `--model` | `models/hand_landmarker.task` | Path to the MediaPipe model file |
| `--fps` | `30` | Target frame rate passed to the EKF motion model |
| `--show-both` | off | Display both camera feeds side by side |
