import numpy as np

from palm_utils import compute_palm_frame, from_palm_frame, to_palm_frame

JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "thumb_joint_0": (0.0, 90.0),
    "thumb_joint_1": (-60.0, 90.0),
    "thumb_joint_2": (0.0, 90.0),
    "index_joint_0": (-15.0, 15.0),
    "index_joint_1": (0.0, 110.0),
    "index_joint_2": (0.0, 110.0),
    "middle_joint_0": (0.0, 110.0),
    "middle_joint_1": (0.0, 110.0),
    "ring_joint_0": (0.0, 110.0),
    "ring_joint_1": (0.0, 110.0),
    "little_joint_0": (0.0, 110.0),
    "little_joint_1": (0.0, 110.0),
}

MEDIAPIPE_FINGER_CHAINS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}

XHAND1_NODE_INDEX = {
    "wrist": 0,
    "thumb_base": 1,
    "thumb_mid": 2,
    "thumb_tip": 3,
    "index_base": 4,
    "index_mid": 5,
    "index_tip": 6,
    "middle_base": 7,
    "middle_mid": 8,
    "middle_tip": 9,
    "ring_base": 10,
    "ring_mid": 11,
    "ring_tip": 12,
    "little_base": 13,
    "little_mid": 14,
    "little_tip": 15,
}

XHAND1_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15),
    (4, 7), (7, 10), (10, 13),
]


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def _clamp_joint(name: str, value: float) -> float:
    lower, upper = JOINT_LIMITS_DEG[name]
    return float(np.clip(value, lower, upper))


def _rotate(vec: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = _normalize(axis)
    if np.linalg.norm(axis) < 1e-8 or abs(angle_deg) < 1e-8:
        return vec
    angle = np.deg2rad(angle_deg)
    return (
        vec * np.cos(angle)
        + np.cross(axis, vec) * np.sin(angle)
        + axis * np.dot(axis, vec) * (1.0 - np.cos(angle))
    )


def _bend_axis(direction: np.ndarray, palm_z: np.ndarray) -> np.ndarray:
    axis = np.cross(direction, palm_z)
    if np.linalg.norm(axis) < 1e-8:
        axis = np.array([0.0, 1.0, 0.0])
    return _normalize(axis)


def get_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two 3D vectors."""
    u1 = v1 / (np.linalg.norm(v1) + 1e-8)
    u2 = v2 / (np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))))


def calculate_joint_angles_12dof(landmarks: np.ndarray) -> dict[str, float]:
    """Convert (21,3) landmarks to 12-DOF joint angles with configured limits."""
    palm_normal = np.cross(landmarks[5] - landmarks[0], landmarks[17] - landmarks[0])
    thumb_cmc_mcp = landmarks[2] - landmarks[1]

    angles = {
        "thumb_joint_0": _clamp_joint(
            "thumb_joint_0", get_angle(thumb_cmc_mcp, palm_normal) - 90.0
        ),
        "thumb_joint_1": _clamp_joint(
            "thumb_joint_1", get_angle(landmarks[2] - landmarks[1], landmarks[3] - landmarks[2])
        ),
        "thumb_joint_2": _clamp_joint(
            "thumb_joint_2", get_angle(landmarks[3] - landmarks[2], landmarks[4] - landmarks[3])
        ),
        "index_joint_0": _clamp_joint(
            "index_joint_0", get_angle(landmarks[5] - landmarks[0], landmarks[9] - landmarks[0]) - 15.0
        ),
        "index_joint_1": _clamp_joint(
            "index_joint_1", get_angle(landmarks[5] - landmarks[0], landmarks[6] - landmarks[5])
        ),
        "index_joint_2": _clamp_joint(
            "index_joint_2", get_angle(landmarks[6] - landmarks[5], landmarks[7] - landmarks[6])
        ),
    }

    for name, idx in (
        ("middle", [9, 10, 11, 12]),
        ("ring", [13, 14, 15, 16]),
        ("little", [17, 18, 19, 20]),
    ):
        angles[f"{name}_joint_0"] = _clamp_joint(
            f"{name}_joint_0",
            get_angle(landmarks[idx[0]] - landmarks[0], landmarks[idx[1]] - landmarks[idx[0]]),
        )
        angles[f"{name}_joint_1"] = _clamp_joint(
            f"{name}_joint_1",
            get_angle(landmarks[idx[1]] - landmarks[idx[0]], landmarks[idx[2]] - landmarks[idx[1]]),
        )

    return angles


def reconstruct_12dof_hand(landmarks: np.ndarray) -> np.ndarray:
    """Approximate an XHAND1-style 3D hand skeleton from the 12-DOF joint-angle representation."""
    joint_angles = calculate_joint_angles_12dof(landmarks)
    R = compute_palm_frame(landmarks)
    pts_palm = to_palm_frame(landmarks, R)
    recon = np.zeros((len(XHAND1_NODE_INDEX), 3), dtype=pts_palm.dtype)
    recon[XHAND1_NODE_INDEX["wrist"]] = pts_palm[0]

    palm_z = np.array([0.0, 0.0, 1.0])

    for name, chain in MEDIAPIPE_FINGER_CHAINS.items():
        mcp, pip, dip, tip = chain
        base_key = f"{name}_base"
        mid_key = f"{name}_mid"
        tip_key = f"{name}_tip"
        base = pts_palm[mcp]
        lengths = (
            np.linalg.norm(pts_palm[pip] - pts_palm[mcp]),
            np.linalg.norm(pts_palm[dip] - pts_palm[pip]) + np.linalg.norm(pts_palm[tip] - pts_palm[dip]),
        )
        base_dir = _normalize(pts_palm[mcp] - pts_palm[0])
        if name == "index":
            base_dir = _rotate(base_dir, palm_z, joint_angles["index_joint_0"])

        dir1 = _rotate(
            base_dir,
            _bend_axis(base_dir, palm_z),
            -joint_angles[f"{name}_joint_0" if name != "index" else "index_joint_1"],
        )
        dir2 = _rotate(
            dir1,
            _bend_axis(dir1, palm_z),
            -joint_angles[f"{name}_joint_1" if name != "index" else "index_joint_2"],
        )

        recon[XHAND1_NODE_INDEX[base_key]] = base
        recon[XHAND1_NODE_INDEX[mid_key]] = base + lengths[0] * _normalize(dir1)
        recon[XHAND1_NODE_INDEX[tip_key]] = recon[XHAND1_NODE_INDEX[mid_key]] + lengths[1] * _normalize(dir2)

    thumb_base = pts_palm[1]
    thumb_lengths = (
        np.linalg.norm(pts_palm[2] - pts_palm[1]) + np.linalg.norm(pts_palm[3] - pts_palm[2]),
        np.linalg.norm(pts_palm[4] - pts_palm[3]),
    )
    thumb_dir = _normalize(pts_palm[2] - pts_palm[1])
    thumb_out_axis = _bend_axis(thumb_dir, palm_z)
    thumb_dir = _rotate(thumb_dir, thumb_out_axis, -joint_angles["thumb_joint_0"])
    thumb_dir1 = _rotate(thumb_dir, _bend_axis(thumb_dir, palm_z), -joint_angles["thumb_joint_1"])
    thumb_dir2 = _rotate(thumb_dir1, _bend_axis(thumb_dir1, palm_z), -joint_angles["thumb_joint_2"])

    recon[XHAND1_NODE_INDEX["thumb_base"]] = thumb_base
    recon[XHAND1_NODE_INDEX["thumb_mid"]] = thumb_base + thumb_lengths[0] * _normalize(thumb_dir1)
    recon[XHAND1_NODE_INDEX["thumb_tip"]] = recon[XHAND1_NODE_INDEX["thumb_mid"]] + thumb_lengths[1] * _normalize(thumb_dir2)

    return from_palm_frame(recon, R)
