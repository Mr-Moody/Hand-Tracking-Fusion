"""Stereo camera calibration and triangulation utilities.

Typical workflow:
  1. Run main.py --calibrate to capture checkerboard pairs and save calibration.
  2. On subsequent runs, calibration is loaded automatically and used for
     stereo triangulation instead of MediaPipe's monocular depth estimates.
"""

import json
from pathlib import Path

import cv2
import numpy as np

BOARD_SIZE = (9, 6)   # interior corner count (columns, rows)
SQUARE_M   = 0.025    # checkerboard square side in metres (25 mm)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _object_points() -> np.ndarray:
    pts = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    pts[:, :2] = np.mgrid[:BOARD_SIZE[0], :BOARD_SIZE[1]].T.reshape(-1, 2)
    return pts * SQUARE_M


_SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
_STEREO_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def run_stereo_calibration(cam0_idx: int, cam1_idx: int,
                           n_pairs: int = 20) -> dict:
    """Interactive stereo calibration.

    Show a checkerboard (BOARD_SIZE interior corners, SQUARE_M metre squares)
    to both cameras.  Press SPACE to capture a pair, Q to finish early.
    """
    cap0 = cv2.VideoCapture(cam0_idx)
    cap1 = cv2.VideoCapture(cam1_idx)

    obj_pt = _object_points()
    obj_pts: list = []
    img_pts0: list = []
    img_pts1: list = []
    img_size = None

    print(f"Checkerboard: {BOARD_SIZE[0]}×{BOARD_SIZE[1]} inner corners, "
          f"{SQUARE_M * 100:.0f} mm squares")
    print("SPACE = capture pair, Q = finish")

    while True:
        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()
        if not ok0 or not ok1:
            break

        g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        found0, c0 = cv2.findChessboardCorners(g0, BOARD_SIZE)
        found1, c1 = cv2.findChessboardCorners(g1, BOARD_SIZE)

        disp0, disp1 = f0.copy(), f1.copy()
        cv2.drawChessboardCorners(disp0, BOARD_SIZE, c0, found0)
        cv2.drawChessboardCorners(disp1, BOARD_SIZE, c1, found1)
        ready = found0 and found1
        colour = (0, 255, 0) if ready else (0, 0, 255)
        label = "READY" if ready else "searching…"
        cv2.putText(disp0, f"{label}   {len(obj_pts)}/{n_pairs}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
        cv2.imshow("Calibration — cam0", disp0)
        cv2.imshow("Calibration — cam1", disp1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and ready:
            img_pts0.append(cv2.cornerSubPix(g0, c0, (11, 11), (-1, -1), _SUBPIX_CRIT))
            img_pts1.append(cv2.cornerSubPix(g1, c1, (11, 11), (-1, -1), _SUBPIX_CRIT))
            obj_pts.append(obj_pt)
            img_size = g0.shape[::-1]
            print(f"  Captured {len(obj_pts)}/{n_pairs}")
            if len(obj_pts) >= n_pairs:
                break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

    if len(obj_pts) < 5:
        raise RuntimeError(
            f"Only {len(obj_pts)} pairs captured — need at least 5."
        )

    print("Running calibration…")
    _, K0, d0, _, _ = cv2.calibrateCamera(obj_pts, img_pts0, img_size, None, None)
    _, K1, d1, _, _ = cv2.calibrateCamera(obj_pts, img_pts1, img_size, None, None)

    rms, K0, d0, K1, d1, R, T, _E, _F = cv2.stereoCalibrate(
        obj_pts, img_pts0, img_pts1,
        K0, d0, K1, d1, img_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=_STEREO_CRIT,
    )
    print(f"Stereo RMS reprojection error: {rms:.3f} px")

    return {
        "K0": K0.tolist(), "d0": d0.tolist(),
        "K1": K1.tolist(), "d1": d1.tolist(),
        "R":  R.tolist(),  "T":  T.tolist(),
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_calibration(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Calibration saved → {path}")


def load_calibration(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=np.float64) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Projection matrices + triangulation
# ---------------------------------------------------------------------------

def build_projection_matrices(
    cal: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (P0, P1, K0, d0, K1, d1) ready for triangulation.

    P0 = K0 @ [I | 0]  (camera-0 is the world origin)
    P1 = K1 @ [R | T]  (camera-1 relative to camera-0)
    """
    K0, d0 = cal["K0"], cal["d0"]
    K1, d1 = cal["K1"], cal["d1"]
    R, T   = cal["R"],  cal["T"].reshape(3, 1)
    P0 = K0 @ np.hstack([np.eye(3),    np.zeros((3, 1))])
    P1 = K1 @ np.hstack([R,            T               ])
    return P0, P1, K0, d0, K1, d1


def triangulate_landmarks(
    pts0: np.ndarray, pts1: np.ndarray,
    P0: np.ndarray,   P1: np.ndarray,
    K0: np.ndarray,   d0: np.ndarray,
    K1: np.ndarray,   d1: np.ndarray,
) -> np.ndarray:
    """Triangulate (21,3) 3D points from two sets of (21,2) image points.

    Returns wrist-centred metric coordinates in camera-0 frame, matching the
    convention used by MediaPipe's hand_world_landmarks.
    """
    u0 = cv2.undistortPoints(pts0.reshape(-1, 1, 2), K0, d0, P=K0).reshape(-1, 2)
    u1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, d1, P=K1).reshape(-1, 2)
    pts4d  = cv2.triangulatePoints(P0, P1, u0.T, u1.T)
    pts3d  = (pts4d[:3] / pts4d[3]).T.astype(np.float32)
    pts3d -= pts3d[0]   # wrist-centre (landmark 0)
    return pts3d
