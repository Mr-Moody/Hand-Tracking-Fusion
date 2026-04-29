import threading
import time

import cv2
import numpy as np


class CameraSource():
    def __init__(self, index: int):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        # Minimise the driver-side buffer so the background thread drains it quickly.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.index = index

        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_ts: int = int(time.monotonic() * 1000)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            ts = int(time.monotonic() * 1000)
            if ret:
                with self._lock:
                    self._latest_frame = cv2.flip(frame, 1)
                    self._latest_ts = ts

    @property
    def is_open(self) -> bool:
        return self._cap.isOpened()

    def wait_for_first_frame(self, timeout: float = 5.0) -> bool:
        """Block until the background thread captures the first frame."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    return True
            time.sleep(0.05)
        return False

    def read(self) -> tuple[np.ndarray | None, int]:
        """Returns the most recently captured frame and the current time in ms.

        Timestamp reflects the time of this call, not the capture time, so
        repeated reads of the same frame still produce strictly increasing
        timestamps as required by MediaPipe's detect_for_video.
        """
        with self._lock:
            return self._latest_frame, int(time.monotonic() * 1000)

    def release(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()
