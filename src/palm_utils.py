import numpy as np

# MediaPipe hand landmark connectivity
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm arch
]

# Indices used to define the palm plane
_WRIST = 0
_INDEX_MCP = 5
_PINKY_MCP = 17


def compute_palm_frame(pts: np.ndarray) -> np.ndarray:
    """Compute a right-handed palm rotation matrix from (21,3) wrist-centred landmarks.

    Returns R (3x3) whose columns are [palm_x, palm_y, palm_z] in world space:
      palm_x  — wrist → index MCP (across the palm)
      palm_z  — palm normal (cross product of the two knuckle vectors)
      palm_y  — palm_z × palm_x (roughly wrist → fingertips)

    To convert world → palm frame:  pts_palm = pts @ R
    To convert palm → world frame:  pts_world = pts_palm @ R.T
    """
    v_idx = pts[_INDEX_MCP] - pts[_WRIST]
    v_pnk = pts[_PINKY_MCP] - pts[_WRIST]

    palm_z = np.cross(v_idx, v_pnk)
    norm_z = np.linalg.norm(palm_z)
    if norm_z < 1e-6:
        return np.eye(3)
    palm_z /= norm_z

    palm_x = v_idx / (np.linalg.norm(v_idx) + 1e-8)
    palm_y = np.cross(palm_z, palm_x)
    palm_y /= np.linalg.norm(palm_y) + 1e-8

    return np.column_stack([palm_x, palm_y, palm_z])


def to_palm_frame(pts: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Transform (21,3) wrist-centred world points into the palm frame."""
    return pts @ R  # equivalent to (R.T @ pts.T).T for orthonormal R


def from_palm_frame(pts_palm: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Transform (21,3) palm-frame points back to world frame."""
    return pts_palm @ R.T


def average_rotations(R_list: list[np.ndarray]) -> np.ndarray:
    """Geodesic mean of rotation matrices via SVD projection onto SO(3)."""
    R_sum = sum(R_list)
    U, _, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:  # reflection guard
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def smooth_rotation(R_prev: np.ndarray, R_new: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """Exponential moving average on SO(3) via linear blend + SVD re-projection."""
    # Prevent z-axis from flipping discontinuously between frames
    if np.dot(R_new[:, 2], R_prev[:, 2]) < 0:
        R_new = R_new.copy()
        R_new[:, 2] *= -1
        R_new[:, 0] *= -1  # flip x too to keep det = +1

    R_blend = (1.0 - alpha) * R_prev + alpha * R_new
    U, _, Vt = np.linalg.svd(R_blend)
    R_s = U @ Vt
    if np.linalg.det(R_s) < 0:
        U[:, -1] *= -1
        R_s = U @ Vt
    return R_s


def compute_bone_lengths(pts: np.ndarray) -> np.ndarray:
    """Return Euclidean length of each bone in HAND_CONNECTIONS order."""
    return np.array([
        np.linalg.norm(pts[child] - pts[parent])
        for parent, child in HAND_CONNECTIONS
    ])


def enforce_bone_lengths(pts: np.ndarray, ref_lengths: np.ndarray) -> np.ndarray:
    """Project pts onto the manifold defined by the reference bone lengths.

    One forward pass from proximal to distal joints.  Keeps the parent fixed
    and repositions the child along the existing bone direction.
    """
    pts = pts.copy()
    for i, (parent, child) in enumerate(HAND_CONNECTIONS):
        vec = pts[child] - pts[parent]
        actual = np.linalg.norm(vec)
        if actual < 1e-8:
            continue
        pts[child] = pts[parent] + vec * (ref_lengths[i] / actual)
    return pts
