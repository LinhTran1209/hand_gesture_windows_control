from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(slots=True)
class WebcamConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    backend: Optional[int] = None


class WebcamCapture:
    """Lớp bao webcam với kiểm tra lỗi và cấu hình cơ bản."""

    def __init__(self, config: Optional[WebcamConfig] = None) -> None:
        self.config = config or WebcamConfig()
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Mở webcam theo cấu hình đã khai báo."""
        if self.cap is not None and self.cap.isOpened():
            return

        if self.config.backend is None:
            self.cap = cv2.VideoCapture(self.config.camera_index)
        else:
            self.cap = cv2.VideoCapture(self.config.camera_index, self.config.backend)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Không thể mở webcam với camera_index={self.config.camera_index}."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Đọc một frame từ webcam."""
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Webcam chưa được mở. Hãy gọi open() trước.")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None

        if self.config.mirror:
            frame = cv2.flip(frame, 1)
        return True, frame

    def get_resolution(self) -> Tuple[int, int]:
        """Trả về kích thước hiện tại của webcam."""
        if self.cap is None or not self.cap.isOpened():
            return self.config.width, self.config.height
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def release(self) -> None:
        """Giải phóng webcam."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "WebcamCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[override]
        self.release()
