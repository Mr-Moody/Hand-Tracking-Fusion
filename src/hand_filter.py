"""Hand filter to prevent tracking from jumping between multiple hands.

Tracks the nearest hand across frames and prevents sudden jumps to other hands
by enforcing a distance threshold. When no hand is visible, the filter remembers
the last tracked hand position and resumes tracking if the same hand reappears.
"""

from dataclasses import dataclass
import numpy as np
from hand_detector import HandDetection


@dataclass
class TrackedHand:
    """State of a single tracked hand."""
    detection: HandDetection
    wrist_pos: np.ndarray  # (3,) world-space wrist position
    age: int  # frames since first detection


class HandFilter:
    """Filter to select which hand to track from multiple detections.
    
    Features:
    - Tracks the nearest hand (by wrist distance)
    - Prevents tracking jumps by enforcing a max distance threshold
    - Maintains temporal stability by preferring the previously tracked hand
      when multiple hands are within threshold
    """
    
    def __init__(self, max_hand_distance: float = 0.3, memory_frames: int = 30):
        """
        Parameters
        ----------
        max_hand_distance : float
            Maximum distance (metres) a hand can move between frames before
            being considered a different hand. Default 0.3m (30cm).
        memory_frames : int
            Number of frames to remember the last hand position if no hand
            is visible (for tracking resume). Default 30 frames.
        """
        self.max_hand_distance = max_hand_distance
        self.memory_frames = memory_frames
        
        self._tracked_hand: TrackedHand | None = None
        self._frames_since_lost = 0
    
    def filter_detection(self, detections: list[HandDetection | None]) -> HandDetection | None:
        """Select the best hand from detections across multiple cameras.
        
        This method now works with multiple hands from each camera.
        For a system with cameras 0 and 1, call it as:
            filter_detection([hand_from_cam0, hand_from_cam1])
        
        where each hand is either a HandDetection or None.
        
        If multiple hands are detected in a single camera, call detect_all()
        and handle them by collecting the best hand across all cameras first.
        
        Parameters
        ----------
        detections : list[HandDetection | None]
            One entry per camera, None if no hand detected in that camera.
        
        Returns
        -------
        HandDetection or None
            The filtered hand detection, or None if no suitable hand found.
        """
        # Collect all valid detections with their wrist positions
        candidates = []
        for detection in detections:
            if detection is not None:
                wrist_pos = detection.landmarks_3d[0]  # landmark 0 is wrist
                candidates.append((detection, wrist_pos))
        
        if not candidates:
            # No hands detected
            if self._tracked_hand is not None:
                self._frames_since_lost += 1
                if self._frames_since_lost > self.memory_frames:
                    # Forget the hand after memory expires
                    self._tracked_hand = None
            return None
        
        # If we have a tracked hand, prefer continuing to track it
        if self._tracked_hand is not None:
            best_detection, best_wrist = self._select_nearest_hand(
                candidates, 
                self._tracked_hand.wrist_pos
            )
            # Check if the nearest hand is within acceptable distance
            distance = np.linalg.norm(best_wrist - self._tracked_hand.wrist_pos)
            if distance <= self.max_hand_distance:
                # Continue tracking the same hand
                self._tracked_hand = TrackedHand(
                    detection=best_detection,
                    wrist_pos=best_wrist,
                    age=self._tracked_hand.age + 1
                )
                self._frames_since_lost = 0
                return best_detection
            else:
                # Hand moved too far; it's a different hand — ignore
                # Keep tracking the previous hand if it's still in memory
                if self._frames_since_lost < self.memory_frames:
                    self._frames_since_lost += 1
                    return None
                else:
                    # Memory expired; track the new closest hand
                    best_detection, best_wrist = candidates[0]
                    self._tracked_hand = TrackedHand(
                        detection=best_detection,
                        wrist_pos=best_wrist,
                        age=1
                    )
                    self._frames_since_lost = 0
                    return best_detection
        else:
            # No previously tracked hand; start tracking the first (nearest) detection
            best_detection, best_wrist = candidates[0]
            self._tracked_hand = TrackedHand(
                detection=best_detection,
                wrist_pos=best_wrist,
                age=1
            )
            self._frames_since_lost = 0
            return best_detection

    def filter_detections_from_all_cameras(
        self, 
        detections_per_camera: list[list[HandDetection]]
    ) -> HandDetection | None:
        """Select the best hand from all detected hands across multiple cameras.
        
        Parameters
        ----------
        detections_per_camera : list[list[HandDetection]]
            For each camera, a list of all detected hands (may be empty).
        
        Returns
        -------
        HandDetection or None
            The filtered hand detection, or None if no suitable hand found.
        """
        # Flatten detections across all cameras
        all_detections = []
        for camera_detections in detections_per_camera:
            all_detections.extend(camera_detections)
        
        if not all_detections:
            # No hands detected
            if self._tracked_hand is not None:
                self._frames_since_lost += 1
                if self._frames_since_lost > self.memory_frames:
                    # Forget the hand after memory expires
                    self._tracked_hand = None
            return None
        
        # Create candidates list
        candidates = []
        for detection in all_detections:
            wrist_pos = detection.landmarks_3d[0]  # landmark 0 is wrist
            candidates.append((detection, wrist_pos))
        
        # If we have a tracked hand, prefer continuing to track it
        if self._tracked_hand is not None:
            best_detection, best_wrist = self._select_nearest_hand(
                candidates, 
                self._tracked_hand.wrist_pos
            )
            # Check if the nearest hand is within acceptable distance
            distance = np.linalg.norm(best_wrist - self._tracked_hand.wrist_pos)
            if distance <= self.max_hand_distance:
                # Continue tracking the same hand
                self._tracked_hand = TrackedHand(
                    detection=best_detection,
                    wrist_pos=best_wrist,
                    age=self._tracked_hand.age + 1
                )
                self._frames_since_lost = 0
                return best_detection
            else:
                # Hand moved too far; it's a different hand — ignore
                # Keep tracking the previous hand if it's still in memory
                if self._frames_since_lost < self.memory_frames:
                    self._frames_since_lost += 1
                    return None
                else:
                    # Memory expired; track the new closest hand
                    best_detection, best_wrist = candidates[0]
                    self._tracked_hand = TrackedHand(
                        detection=best_detection,
                        wrist_pos=best_wrist,
                        age=1
                    )
                    self._frames_since_lost = 0
                    return best_detection
        else:
            # No previously tracked hand; start tracking the closest detection
            best_detection, best_wrist = self._select_nearest_hand(
                candidates,
                np.array([0.0, 0.0, 0.5])  # Default reference: center, 50cm away
            )
            self._tracked_hand = TrackedHand(
                detection=best_detection,
                wrist_pos=best_wrist,
                age=1
            )
            self._frames_since_lost = 0
            return best_detection
    
    @staticmethod
    def _select_nearest_hand(
        candidates: list[tuple[HandDetection, np.ndarray]],
        reference_pos: np.ndarray
    ) -> tuple[HandDetection, np.ndarray]:
        """Find the hand closest to the reference position.
        
        Parameters
        ----------
        candidates : list of (HandDetection, wrist_pos)
            Available hand detections.
        reference_pos : ndarray
            Reference 3D position to compare distances from.
        
        Returns
        -------
        (HandDetection, wrist_pos)
            The closest hand and its wrist position.
        """
        distances = [np.linalg.norm(wrist - reference_pos) for _, wrist in candidates]
        nearest_idx = np.argmin(distances)
        return candidates[nearest_idx]
    
    @property
    def tracked_hand_age(self) -> int | None:
        """Number of frames the current hand has been tracked, or None."""
        return self._tracked_hand.age if self._tracked_hand is not None else None
    
    @property
    def frames_since_lost(self) -> int:
        """Number of frames since the tracked hand was last visible."""
        return self._frames_since_lost
