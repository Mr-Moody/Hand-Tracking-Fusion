import numpy as np


def build_partial_H(vis_indices: np.ndarray, N: int = 21) -> np.ndarray:
    """Observation matrix for a subset of joints.

    Each visible joint contributes a 3-row block selecting its [x, y, z]
    from the 6-element (pos + vel) per-joint state.

    Returns: (3*k, 6*N) where k = len(vis_indices)
    """
    k = len(vis_indices)
    H = np.zeros((3 * k, 6 * N))
    
    for row, joint in enumerate(vis_indices):
        H[row * 3 : row * 3 + 3, joint * 6 : joint * 6 + 3] = np.eye(3)

    return H


class JointHandEKF():
    def __init__(self, N: int = 21, dt: float = 1 / 30):
        self.N = N
        self.n = N * 6
        self.F = np.kron(np.eye(N), self._f_single(dt))
        self.Q = np.kron(np.eye(N), np.diag([1.0] * 3 + [10.0] * 3))
        self.P = np.kron(np.eye(N), np.diag([50.0] * 3 + [20.0] * 3))
        self.R = np.kron(np.eye(N), np.diag([5.0] * 3))
        self.x = np.zeros(self.n)
        self.initialised = False

    def _f_single(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt

        return F

    @property
    def positions(self) -> np.ndarray:
        """Current estimated joint positions as (N, 3)."""
        return self.x.reshape(self.N, 6)[:, :3].copy()

    def init(self, pts3d: np.ndarray):
        """Initialise state from a (21, 3) point cloud."""
        for i in range(self.N):
            self.x[i * 6 : i * 6 + 3] = pts3d[i]

        self.initialised = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, pts3d: np.ndarray, visible_mask: np.ndarray):
        """Update with a partial observation.

        pts3d: (21, 3) — world-space positions from one camera
        visible_mask: (21,) bool — which joints are reliably observed
        """
        vis = np.where(visible_mask)[0]
        if len(vis) == 0:
            return

        H = build_partial_H(vis, self.N)
        z = pts3d[vis].flatten()
        R = np.kron(np.eye(len(vis)), np.diag([5.0] * 3))

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ H) @ self.P
