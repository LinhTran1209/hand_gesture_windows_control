from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2 as cv
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles


@dataclass(slots=True)
class HandLandmarkPoint:
    x: float
    y: float
    z: float


@dataclass(slots=True)
class HandDetectionResult:
    landmarks: List[HandLandmarkPoint]
    handedness: str
    score: float


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.mp_hands = mp_hands
        self.drawing = drawing
        self.drawing_styles = drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, List[HandDetectionResult]]:
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        annotated_frame = frame_bgr.copy()
        detections: List[HandDetectionResult] = []

        if results.multi_hand_landmarks:
            handedness_list = results.multi_handedness or []

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness_label = "Unknown"
                handedness_score = 0.0

                if idx < len(handedness_list) and handedness_list[idx].classification:
                    handedness_label = handedness_list[idx].classification[0].label
                    handedness_score = float(
                        handedness_list[idx].classification[0].score
                    )

                points = [
                    HandLandmarkPoint(lm.x, lm.y, lm.z)
                    for lm in hand_landmarks.landmark
                ]

                detections.append(
                    HandDetectionResult(
                        landmarks=points,
                        handedness=handedness_label,
                        score=handedness_score,
                    )
                )

                self.drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style(),
                )

                self._draw_label(
                    annotated_frame,
                    hand_landmarks,
                    handedness_label,
                    handedness_score,
                )

        return annotated_frame, detections

    def _draw_label(
        self,
        frame: np.ndarray,
        hand_landmarks,
        handedness_label: str,
        handedness_score: float,
    ) -> None:
        height, width, _ = frame.shape
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]

        min_x = int(min(x_coords) * width)
        min_y = int(min(y_coords) * height)

        label = f"{handedness_label} ({handedness_score:.2f})"
        cv.putText(
            frame,
            label,
            (max(min_x, 10), max(min_y - 10, 30)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
