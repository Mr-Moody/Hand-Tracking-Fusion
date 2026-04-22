import time

import cv2
import numpy as np


class CameraSource():
    def __init__(self, index: int):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        self.index = index

    @property
    def is_open(self) -> bool:
        return self._cap.isOpened()

    def read(self) -> tuple[np.ndarray | None, int]:
        """Returns (frame, timestamp_ms). Frame is None on read failure."""
        ret, frame = self._cap.read()
        timestamp_ms = int(time.monotonic() * 1000)
        if not ret:
            return None, timestamp_ms
        return cv2.flip(frame, 1), timestamp_ms

    def release(self):
        self._cap.release()
