import matplotlib.pyplot as plt
import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

_BG = "#0d0d1a"
_JOINT_COLOR = "#00dfff"
_BONE_COLOR = "#ffffff"


class Hand3DPlot:
    def __init__(self, axis_limit: float = 0.12):
        self._lim = axis_limit
        plt.ion()
        self._fig = plt.figure("Fused 3D Hand", figsize=(5, 5))
        self._fig.patch.set_facecolor(_BG)
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.set_facecolor(_BG)
        self._configure_axes()
        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        # Pre-allocate artists so we can update data in-place each frame
        # rather than clearing and re-adding (much faster)
        self._bone_lines = [
            self._ax.plot([], [], [], color=_BONE_COLOR, lw=1.5)[0]
            for _ in HAND_CONNECTIONS
        ]
        self._joints = self._ax.scatter([], [], [], c=_JOINT_COLOR, s=25, depthshade=True)

    def _configure_axes(self) -> None:
        ax = self._ax
        lim = self._lim
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("X (m)", color="white", labelpad=4)
        ax.set_ylabel("Y (m)", color="white", labelpad=4)
        ax.set_zlabel("Z (m)", color="white", labelpad=4)
        ax.tick_params(colors="white", labelsize=7)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#333355")
        ax.grid(True, color="#222244", linewidth=0.5)

    def update(self, pts3d: np.ndarray | None) -> None:
        if pts3d is not None:
            xs, ys, zs = pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]

            for line, (a, b) in zip(self._bone_lines, HAND_CONNECTIONS):
                line.set_data([xs[a], xs[b]], [ys[a], ys[b]])
                line.set_3d_properties([zs[a], zs[b]])

            self._joints._offsets3d = (xs, ys, zs)
        else:
            for line in self._bone_lines:
                line.set_data([], [])
                line.set_3d_properties([])
            self._joints._offsets3d = ([], [], [])

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        plt.close(self._fig)
