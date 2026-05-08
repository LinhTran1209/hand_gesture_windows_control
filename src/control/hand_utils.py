from collections import Counter
from typing import Any
import numpy as np

from .config import EPS, PRIMARY_SWITCH_MAX_DISTANCE
from .state import ControlState


def get_landmarks(det: Any):
    if det is None:
        return None
    if hasattr(det, "landmarks"):
        return det.landmarks
    if isinstance(det, dict):
        return det.get("landmarks")
    return None


def get_handedness(det: Any) -> str:
    if det is None:
        return "Unknown"
    value = getattr(det, "handedness", None)
    if value is None and isinstance(det, dict):
        value = det.get("handedness")
    if value is None:
        return "Unknown"
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return str(value[0])
    return str(value)


def landmark_xy(landmarks: Any, idx: int) -> tuple[float, float] | None:
    if landmarks is None or len(landmarks) <= idx:
        return None
    lm = landmarks[idx]
    x = getattr(lm, "x", None)
    y = getattr(lm, "y", None)
    if x is None or y is None:
        return None
    return float(x), float(y)


def hand_center_xy(landmarks: Any) -> tuple[float, float] | None:
    if landmarks is None or len(landmarks) == 0:
        return None
    xs, ys = [], []
    for lm in landmarks:
        x = getattr(lm, "x", None)
        y = getattr(lm, "y", None)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    if not xs or not ys:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def choose_primary_and_secondary(
    detections: list[Any],
    state: ControlState,
) -> tuple[Any | None, Any | None]:
    if not detections:
        state.primary_center = None
        return None, None

    centers: list[tuple[int, tuple[float, float]]] = []
    for i, det in enumerate(detections[:2]):
        center = hand_center_xy(get_landmarks(det))
        if center is not None:
            centers.append((i, center))

    if not centers:
        state.primary_center = None
        secondary = detections[1] if len(detections) > 1 else None
        return detections[0], secondary

    if state.primary_center is None:
        primary_idx = centers[0][0]
    else:
        px, py = state.primary_center
        primary_idx, best_center = min(
            centers,
            key=lambda item: float(np.hypot(item[1][0] - px, item[1][1] - py)),
        )
        dist = float(np.hypot(best_center[0] - px, best_center[1] - py))
        if dist > PRIMARY_SWITCH_MAX_DISTANCE and len(centers) > 1:
            primary_idx = centers[0][0]

    primary_det = detections[primary_idx]
    primary_center = hand_center_xy(get_landmarks(primary_det))
    if primary_center is not None:
        state.primary_center = primary_center

    secondary_det = None
    for i, det in enumerate(detections[:2]):
        if i != primary_idx:
            secondary_det = det
            break
    return primary_det, secondary_det


def two_finger_pose_features(landmarks: Any) -> tuple[float, float, float, float, float] | None:
    index_tip = landmark_xy(landmarks, 8)
    middle_tip = landmark_xy(landmarks, 12)
    index_base = landmark_xy(landmarks, 5)
    middle_base = landmark_xy(landmarks, 9)

    if any(v is None for v in (index_tip, middle_tip, index_base, middle_base)):
        return None

    tip_x = (index_tip[0] + middle_tip[0]) / 2.0
    tip_y = (index_tip[1] + middle_tip[1]) / 2.0
    base_x = (index_base[0] + middle_base[0]) / 2.0
    base_y = (index_base[1] + middle_base[1]) / 2.0
    vx = tip_x - base_x
    vy = tip_y - base_y

    tip_gap = float(np.hypot(index_tip[0] - middle_tip[0], index_tip[1] - middle_tip[1]))
    index_len = float(np.hypot(index_tip[0] - index_base[0], index_tip[1] - index_base[1]))
    middle_len = float(np.hypot(middle_tip[0] - middle_base[0], middle_tip[1] - middle_base[1]))

    finger_len = max((index_len + middle_len) / 2.0, EPS)
    spread_ratio = tip_gap / finger_len

    return tip_x, tip_y, vx, vy, spread_ratio


def avg_pose_frames(
    frames: list[tuple[float, float, float, float, float, str]],
) -> tuple[float, float, float, float, float]:
    n = max(1, len(frames))
    return (
        sum(p[0] for p in frames) / n,
        sum(p[1] for p in frames) / n,
        sum(p[2] for p in frames) / n,
        sum(p[3] for p in frames) / n,
        sum(p[4] for p in frames) / n,
    )


def most_common_handedness(
    frames: list[tuple[float, float, float, float, float, str]],
) -> str:
    labels = [p[5] for p in frames]
    if not labels:
        return "Unknown"
    return Counter(labels).most_common(1)[0][0]
