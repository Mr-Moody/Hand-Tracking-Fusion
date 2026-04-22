import cv2
import numpy as np

from hand_detector import HandDetection

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm knuckles
]

_CYAN = (0, 220, 255)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_GREY = (100, 100, 100)


class HandVisualiser:
    @staticmethod
    def draw_detection(frame: np.ndarray, detection: HandDetection, color: tuple = _CYAN) -> None:
        pts = [(int(x), int(y)) for x, y in detection.landmarks_2d]

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], _WHITE, 2, cv2.LINE_AA)

        for i, (x, y) in enumerate(pts):
            dot_color = color if detection.visible_mask[i] else _GREY
            cv2.circle(frame, (x, y), 5, dot_color, -1)
            cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)

    @staticmethod
    def draw_bounding_box(frame: np.ndarray, detection: HandDetection) -> None:
        xs = detection.landmarks_2d[:, 0]
        ys = detection.landmarks_2d[:, 1]

        x1 = int(xs.min()) - 10
        y1 = int(ys.min()) - 10
        x2 = int(xs.max()) + 10
        y2 = int(ys.max()) + 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), _GREEN, 2)

    @staticmethod
    def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
        for i, line in enumerate(lines):
            cv2.putText(
                frame, line, (10, 30 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _WHITE, 2, cv2.LINE_AA,
            )
            
        cv2.putText(
            frame, "Press Q to quit",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
        )
