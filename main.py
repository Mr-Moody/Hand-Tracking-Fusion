import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))

from camera import CameraSource
from fusion import HandFusion
from hand_detector import HandDetector
from hand_filter import HandFilter
from plot3d import Hand3DPlot
from visualiser import HandVisualiser

DEFAULT_MODEL   = Path(__file__).parent / "models" / "hand_landmarker.task"
DEFAULT_CAL     = Path(__file__).parent / "calibration" / "stereo_cal.json"


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
    p.add_argument("--calibrate", action="store_true",
                   help="Run interactive stereo calibration before starting "
                        "(requires --cam1; saves to calibration/stereo_cal.json)")
    p.add_argument("--cal-path", type=Path, default=DEFAULT_CAL, metavar="PATH",
                   help=f"Stereo calibration file (default: {DEFAULT_CAL})")
    return p.parse_args()


def run_calibration(args: argparse.Namespace) -> None:
    from calibration.calibration import run_stereo_calibration, save_calibration
    if args.cam1 is None:
        print("--calibrate requires --cam1 to be set.")
        sys.exit(1)
    try:
        data = run_stereo_calibration(args.cam0, args.cam1)
        save_calibration(args.cal_path, data)
    except Exception as exc:
        print(f"[CAL] Calibration failed: {exc}")
        raise


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

    if args.calibrate:
        run_calibration(args)

    # Load stereo calibration if available
    calibration = None
    if args.cam1 is not None:
        from calibration.calibration import load_calibration
        calibration = load_calibration(args.cal_path)
        if calibration is not None:
            print(f"Stereo calibration loaded from {args.cal_path} — using triangulation")
        else:
            print("No stereo calibration found — using MediaPipe monocular 3D "
                  f"(run with --calibrate to improve rotation tracking)")

    cam0 = CameraSource(args.cam0)
    cam1 = CameraSource(args.cam1) if args.cam1 is not None else None

    det0 = HandDetector(args.model, fps=args.fps)
    det1 = HandDetector(args.model, fps=args.fps) if cam1 is not None else None

    fusion = HandFusion(fps=args.fps, calibration=calibration)
    hand_filter = HandFilter(max_hand_distance=0.3, memory_frames=30)
    vis = HandVisualiser()
    plot3d = Hand3DPlot()

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

            # Detect all hands in camera 0
            detections0 = det0.detect_all(frame0, ts0)

            # Detect all hands in camera 1 (if available)
            detections1 = []
            frame1 = None
            if cam1 is not None:
                frame1, ts1 = cam1.read()
                if frame1 is not None:
                    detections1 = det1.detect_all(frame1, ts1)

            # Filter to select the best hand across all cameras
            filtered_detection = hand_filter.filter_detections_from_all_cameras(
                [detections0, detections1] if cam1 else [detections0]
            )

            # Fuse the filtered detection (convert to expected format)
            fused_detections = [filtered_detection, None]
            if cam1 is not None and filtered_detection is not None:
                # For fusion: try to use stereo if available
                # Pass only the filtered detection from the first camera
                fused_detections = [filtered_detection, None]
            
            fused_pts = fusion.update(fused_detections if cam1 else [filtered_detection])

            if filtered_detection is not None:
                vis.draw_detection(frame0, filtered_detection)
                vis.draw_bounding_box(frame0, filtered_detection)

            hud = [f"Cam0: {len(detections0)} hand(s), tracking: {'yes' if filtered_detection else 'no'}"]
            if cam1 is not None:
                hud.append(f"Cam1: {len(detections1)} hand(s)")
            if hand_filter.tracked_hand_age is not None:
                hud.append(f"Hand age: {hand_filter.tracked_hand_age} frames")
            if fused_pts is not None:
                w = fused_pts[0]
                hud.append(f"Wrist 3D: ({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}) m")
            vis.draw_hud(frame0, hud)

            if args.show_both and frame1 is not None:
                if detections1:
                    # Show first detection from camera 1
                    vis.draw_detection(frame1, detections1[0])
                    vis.draw_bounding_box(frame1, detections1[0])
                h0 = frame0.shape[0]
                h1, w1 = frame1.shape[:2]
                if h0 != h1:
                    frame1 = cv2.resize(frame1, (int(w1 * h0 / h1), h0))
                display = cv2.hconcat([frame0, frame1])
            else:
                display = frame0

            plot3d.update(fused_pts)

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
        plot3d.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
