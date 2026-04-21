import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pathlib import Path

DIRECTORY_PATH = Path(__file__).parent

# Download the model once with:
#   wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
MODEL_PATH =  DIRECTORY_PATH / "models" / "hand_landmarker.task"

# Hand connection pairs
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17), # Palm knuckles
]

LANDMARK_COLOR = (0, 220, 255)
CONNECTION_COLOR = (255, 255, 255)
BOX_COLOR = (0, 255, 0)


def draw_hand(frame, hand_landmarks):
    """Draw skeleton and landmark dots for one hand."""
    h, w, _ = frame.shape

    # Pixel coords for every landmark
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    # Connections
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], CONNECTION_COLOR, 2, cv2.LINE_AA)

    # Dots
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, LANDMARK_COLOR, -1)
        cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)   # thin black outline


# MediaPipe Tasks API setup
BaseOptions = mp_python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    raise RuntimeError("Could not open camera. Check that it is connected and not in use.")

print("Hand detection running - press Q to quit.")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.monotonic() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w, _ = frame.shape

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            draw_hand(frame, hand_landmarks)

            # Bounding box
            xs = [lm.x * w for lm in hand_landmarks]
            ys = [lm.y * h for lm in hand_landmarks]
            x1, y1 = int(min(xs)) - 10, int(min(ys)) - 10
            x2, y2 = int(max(xs)) + 10, int(max(ys)) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

            # Left/Right label
            if result.handedness:
                label = result.handedness[idx][0].display_name
                score = result.handedness[idx][0].score
                cv2.putText(
                    frame, f"{label}  {score:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, BOX_COLOR, 2, cv2.LINE_AA,
                )

        # HUD
        cv2.putText(
            frame, f"Hands detected: {len(result.hand_landmarks)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "Press Q to quit",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA,
        )

        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()