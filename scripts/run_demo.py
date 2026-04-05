from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2 as cv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.capture.webcam import WebcamCapture, WebcamConfig
from src.perception.hand_tracker import HandTracker


def main() -> None:
    webcam = WebcamCapture(
        WebcamConfig(
            camera_index=0,
            width=1280,
            height=720,
            fps=30,
            mirror=True,
        )
    )

    tracker = HandTracker(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    prev_time = time.time()

    try:
        webcam.open()
        print("[INFO] Webcam đã mở. Nhấn 'q' để thoát.")

        while True:
            ok, frame = webcam.read()
            if not ok or frame is None:
                print("[WARN] Không đọc được frame từ webcam.")
                continue

            annotated_frame, detections = tracker.process(frame)

            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            cv.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (20, 35),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv.LINE_AA,
            )

            cv.putText(
                annotated_frame,
                f"Hands: {len(detections)}",
                (20, 75),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv.LINE_AA,
            )

            cv.imshow("Hand Gesture Starter - Webcam + Landmarks", annotated_frame)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        tracker.close()
        webcam.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
