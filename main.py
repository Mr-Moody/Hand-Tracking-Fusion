import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))

from camera import CameraSource
from fusion import HandFusion
from hand_detector import HandDetector
from visualiser import HandVisualiser

DEFAULT_MODEL = Path(__file__).parent / "models" / "hand_landmarker.task"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-camera 3D hand tracking with EKF fusion")
    p.add_argument("--cam0", type=int, default=0, metavar="INDEX",
                   help="Primary camera index (default: 0)")
    p.add_argument("--cam1", type=int, default=None, metavar="INDEX",
                   help="Secondary camera index for multi-view fusion (optional)")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, metavar="PATH",
                   help="Path to hand_landmarker.task model")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Target capture/EKF frame rate (default: 30)")
    p.add_argument("--show-both", action="store_true",
                   help="Show both camera feeds side by side when --cam1 is set")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.model.exists():
        print(
            f"Model not found at {args.model}\n"
            "Download with:\n"
            "  wget -P models/ https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )
        sys.exit(1)

    cam0 = CameraSource(args.cam0)
    cam1 = CameraSource(args.cam1) if args.cam1 is not None else None

    det0 = HandDetector(args.model, fps=args.fps)
    det1 = HandDetector(args.model, fps=args.fps) if cam1 is not None else None

    fusion = HandFusion(fps=args.fps)
    vis = HandVisualiser()

    print(f"Camera 0: index {args.cam0}")
    if cam1:
        print(f"Camera 1: index {args.cam1}")
    print("Running — press Q to quit.")

    try:
        while True:
            frame0, ts0 = cam0.read()
            if frame0 is None:
                print("Camera 0 read failed.")
                break

            detection0 = det0.detect(frame0, ts0)

            detection1 = None
            frame1 = None
            if cam1 is not None:
                frame1, ts1 = cam1.read()
                if frame1 is not None:
                    detection1 = det1.detect(frame1, ts1)

            fused_pts = fusion.update([detection0, detection1] if cam1 else [detection0])

            # Annotate primary feed
            if detection0 is not None:
                vis.draw_detection(frame0, detection0)
                vis.draw_bounding_box(frame0, detection0)

            hud = [f"Cam0: {'detected' if detection0 else 'no hand'}"]
            if cam1 is not None:
                hud.append(f"Cam1: {'detected' if detection1 else 'no hand'}")
            if fused_pts is not None:
                w = fused_pts[0]
                hud.append(f"Wrist 3D: ({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}) m")
            vis.draw_hud(frame0, hud)

            # Build display frame
            if args.show_both and frame1 is not None:
                if detection1 is not None:
                    vis.draw_detection(frame1, detection1)
                    vis.draw_bounding_box(frame1, detection1)
                h0 = frame0.shape[0]
                h1, w1 = frame1.shape[:2]
                if h0 != h1:
                    frame1 = cv2.resize(frame1, (int(w1 * h0 / h1), h0))
                display = cv2.hconcat([frame0, frame1])
            else:
                display = frame0

            cv2.imshow("Hand Tracking Fusion", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam0.release()
        if cam1 is not None:
            cam1.release()
        det0.close()
        if det1 is not None:
            det1.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
