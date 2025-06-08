import time
from typing import Optional, Tuple, List
import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, max_hands: int = 1, processing_size: int = 256):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.6,  # Lower confidence for speed
            min_tracking_confidence=0.5,
            model_complexity=0  # Lightest model
        )
        self.results = None
        self.trail: List[Tuple[Tuple[int, int], float]] = []
        self.processing_size = processing_size

    def process(self, frame: np.ndarray) -> None:
        # Downscale significantly
        small_frame = cv2.resize(frame, (self.processing_size, 
                                       int(frame.shape[0] * self.processing_size / frame.shape[1])))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

    def get_fingertip(self, frame_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int]]:
        if not self.results or not self.results.multi_hand_landmarks:
            return None

        hand_landmarks = self.results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark[8]  # Index fingertip
        
        # Scale coordinates
        orig_h, orig_w, _ = frame_shape
        x = int(lm.x * orig_w)
        y = int(lm.y * orig_h)
        
        # Update trail
        t = time.time()
        self.trail.append(((x, y), t))
        # Keep only recent points (0.2 seconds)
        self.trail = [(p, ts) for p, ts in self.trail if t - ts < 0.2]
        return x, y

    def is_slicing(self, threshold: float = 150.0) -> bool:
        if len(self.trail) < 2:
            return False
            
        # Check distance between first and last point
        (x0, y0), _ = self.trail[0]
        (x1, y1), _ = self.trail[-1]
        dist = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
        return dist > threshold