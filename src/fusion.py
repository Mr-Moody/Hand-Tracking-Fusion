from hand_detector import HandDetection
from joint_hand_ekf import JointHandEKF

import numpy as np


class HandFusion():
    def __init__(self, fps: float = 30.0):
        self._ekf = JointHandEKF(dt=1.0 / fps)

    @property
    def is_initialised(self) -> bool:
        return self._ekf.initialised

    def update(self, detections: list[HandDetection | None]) -> np.ndarray | None:
        """Fuse detections from multiple cameras.

        detections: one entry per camera, None if no hand detected.
        Returns: fused (21, 3) joint positions, or None before first detection.
        """
        visible = [d for d in detections if d is not None]

        if not visible:
            if self._ekf.initialised:
                self._ekf.predict()
            return self._ekf.positions if self._ekf.initialised else None

        if not self._ekf.initialised:
            self._ekf.init(visible[0].landmarks_3d)

        self._ekf.predict()
        for det in visible:
            self._ekf.update(det.landmarks_3d, det.visible_mask)

        return self._ekf.positions
