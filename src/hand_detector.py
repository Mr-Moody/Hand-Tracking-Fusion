from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


@dataclass
class HandDetection:
    landmarks_2d: np.ndarray  # (21, 2) pixel coords
    landmarks_3d: np.ndarray  # (21, 3) world coords, metres, wrist-centred
    visible_mask: np.ndarray  # (21,) bool


def _visibility(lm) -> float:
    v = getattr(lm, "visibility", None)
    return float(v) if v is not None else 1.0


class HandDetector():
    def __init__(self, model_path: str | Path, fps: float = 30.0, num_hands: int = 2):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> HandDetection | None:
        """Detect the first (nearest) hand in the frame (backward compatible).
        
        Returns the first detected hand, or None if no hands found.
        Use detect_all() to get all hands in the frame.
        """
        detections = self.detect_all(frame, timestamp_ms)
        return detections[0] if detections else None

    def detect_all(self, frame: np.ndarray, timestamp_ms: int) -> list[HandDetection]:
        """Detect all hands in the frame.
        
        Returns
        -------
        list[HandDetection]
            All detected hands in the frame, ordered as returned by MediaPipe.
            Empty list if no hands found.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return []

        detections = []
        for lm2d, lm3d in zip(result.hand_landmarks, result.hand_world_landmarks):
            landmarks_2d = np.array([[lm.x * w, lm.y * h] for lm in lm2d], dtype=np.float32)
            # Negate x to undo the horizontal flip applied by CameraSource; without
            # this the chirality of the 3-D hand is mirrored (right hand looks left).
            landmarks_3d = np.array([[-lm.x, lm.y, lm.z] for lm in lm3d], dtype=np.float32)

            vis_scores = np.array([_visibility(lm) for lm in lm2d])
            # Only threshold if visibility is actually populated; otherwise all visible
            if vis_scores.max() < 1.0:
                visible_mask = vis_scores > 0.5
            else:
                visible_mask = np.ones(21, dtype=bool)

            detections.append(HandDetection(
                landmarks_2d=landmarks_2d,
                landmarks_3d=landmarks_3d,
                visible_mask=visible_mask,
            ))
        
        return detections

    def close(self):
        self._landmarker.close()
