from __future__ import annotations

from hand_detector import HandDetection
from joint_hand_ekf import JointHandEKF
from palm_utils import (
    compute_palm_frame,
    to_palm_frame,
    from_palm_frame,
    average_rotations,
    smooth_rotation,
    compute_bone_lengths,
    enforce_bone_lengths,
    rotation_angle,
    palm_depth_quality,
)

import numpy as np


class HandFusion:
    """Multi-camera hand pose fusion with palm-frame EKF.

    Architecture
    ------------
    Rather than tracking raw world-space joint positions (where global hand
    rotation shows up as large, depth-heavy movements that MediaPipe estimates
    poorly), the EKF operates in the *palm frame*:

      palm_x  — across the palm (wrist → index MCP)
      palm_y  — along the fingers (derived from palm normal x palm_x)
      palm_z  — palm normal (out-of-palm)

    In this frame the finger *shape* (flexion / extension) is stable during
    global rotation, so the constant-position EKF prediction stays accurate.
    The palm orientation itself is tracked separately as a smoothed SO(3)
    rotation and applied at render time.

    If a stereo calibration file is provided and both cameras see the hand,
    true stereo triangulation replaces MediaPipe's monocular depth estimate,
    giving geometrically accurate 3-D positions.
    """

    def __init__(self, fps: float = 30.0, calibration: dict | None = None):
        self._ekf = JointHandEKF(dt=1.0 / fps)

        # Stereo triangulation (optional)
        self._proj: tuple | None = None
        if calibration is not None:
            from calibration.calibration import build_projection_matrices
            self._proj = build_projection_matrices(calibration)

        self._R_palm: np.ndarray | None = None   # smoothed palm orientation
        self._bone_lengths: np.ndarray | None = None  # set at first detection

    @property
    def is_initialised(self) -> bool:
        return self._ekf.initialised

    def update(self, detections: list[HandDetection | None]) -> np.ndarray | None:
        """Fuse detections from multiple cameras.

        Parameters
        ----------
        detections:
            One entry per camera, None if no hand detected.

        Returns
        -------
        (21, 3) fused joint positions in world space (wrist-centred),
        or None before the first detection.
        """
        visible = [d for d in detections if d is not None]

        if not visible:
            if self._ekf.initialised:
                self._ekf.freeze()
                return self._to_world(self._ekf.positions)
            return None

        # Obtain 3D landmark sets
        if (self._proj is not None
                and len(detections) >= 2
                and detections[0] is not None
                and detections[1] is not None):
            lm3d_list = [self._triangulate(detections[0], detections[1])]
            mask_list = [detections[0].visible_mask & detections[1].visible_mask]
        else:
            lm3d_list = [d.landmarks_3d for d in visible]
            mask_list = [d.visible_mask for d in visible]

        # Compute palm frames and average orientation across cameras
        R_list = [compute_palm_frame(pts) for pts in lm3d_list]
        R_current = average_rotations(R_list) if len(R_list) > 1 else R_list[0]

        # Transform observations into the palm frame
        palm_obs = [to_palm_frame(pts, R_current) for pts in lm3d_list]

        # Initialise EKF on first detection
        if not self._ekf.initialised:
            self._R_palm = R_current.copy()
            self._ekf.init(palm_obs[0])
            self._bone_lengths = compute_bone_lengths(palm_obs[0])
            return self._to_world(self._ekf.positions)

        # Detect large orientation jumps. Any inter-frame rotation > 90° is
        # physically impossible for a hand at normal speed — it means MediaPipe
        # has flipped its palm frame estimate.  Re-seeding the EKF is cleaner
        # than trying to smooth through it (which would produce the squash).
        angle = rotation_angle(self._R_palm, R_current)
        if angle > np.pi / 2:
            self._R_palm = R_current.copy()
            self._ekf.init(palm_obs[0])
            if self._bone_lengths is None:
                self._bone_lengths = compute_bone_lengths(palm_obs[0])
            return self._to_world(self._ekf.positions)

        # Adaptive rotation smoothing: track fast rotations more aggressively
        # so the palm frame doesn't lag behind a quickly flipping hand.
        adaptive_alpha = min(0.85, 0.15 + angle / (np.pi / 4) * 0.15)
        self._R_palm = smooth_rotation(self._R_palm, R_current, alpha=adaptive_alpha)

        self._ekf.predict()
        for pts_palm, mask in zip(palm_obs, mask_list):
            # Scale up depth noise when MediaPipe's z estimate is unreliable
            # (hand flat-on to the camera).  The EKF then relies on its motion
            # model for depth and only trusts x/y from the observation.
            quality = palm_depth_quality(pts_palm)
            depth_noise_scale = float(np.exp(3.0 * (1.0 - quality)))  # 1x→20x
            self._ekf.update(pts_palm, mask, depth_noise_scale=depth_noise_scale)

        # Enforce bone lengths, write corrections back into EKF state
        positions = self._ekf.positions
        if self._bone_lengths is not None:
            positions = enforce_bone_lengths(positions, self._bone_lengths)
            for i in range(21):
                self._ekf.x[i * 6: i * 6 + 3] = positions[i]

        return self._to_world(positions)


    def _to_world(self, pts_palm: np.ndarray) -> np.ndarray:
        """Rotate palm-frame positions back to world space."""
        if self._R_palm is None:
            return pts_palm
        return from_palm_frame(pts_palm, self._R_palm)

    def _triangulate(self, det0: HandDetection, det1: HandDetection) -> np.ndarray:
        from calibration.calibration import triangulate_landmarks
        P0, P1, K0, d0, K1, d1 = self._proj
        return triangulate_landmarks(
            det0.landmarks_2d, det1.landmarks_2d,
            P0, P1, K0, d0, K1, d1,
        )
