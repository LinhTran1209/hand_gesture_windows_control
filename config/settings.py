from dataclasses import dataclass


@dataclass(slots=True)
class CameraSettings:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    mirror: bool = True


@dataclass(slots=True)
class HandTrackingSettings:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_presence_confidence: float = 0.5
