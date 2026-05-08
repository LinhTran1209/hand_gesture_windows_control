import time
from collections import deque
from typing import Any
import numpy as np

from .config import (
    DOWN_SPREAD_RATIO_MAX,
    DYNAMIC_COOLDOWN_SEC,
    DYNAMIC_GATE_LABEL,
    NAVIGATION_ACTION_COOLDOWN_SEC,
    ORIENT_HORIZONTAL_RATIO,
    ORIENT_VERTICAL_RATIO,
    POSE_MAX_MOTION,
    POSE_MIN_FRAMES,
    UP_SPREAD_RATIO_MIN,
)
from .hand_utils import avg_pose_frames, most_common_handedness, two_finger_pose_features
from .state import ControlState

PoseFrame = tuple[float, float, float, float, float, str]


def classify_two_finger_example_pose(points: deque[PoseFrame]) -> str | None:
    if len(points) < POSE_MIN_FRAMES:
        return None

    frames = list(points)[-POSE_MIN_FRAMES:]
    split = max(1, len(frames) // 2)

    sx, sy, _svx, _svy, _sspread = avg_pose_frames(frames[:split])
    ex, ey, _evx, _evy, _espread = avg_pose_frames(frames[split:])

    motion = float(np.hypot(ex - sx, ey - sy))
    if motion > POSE_MAX_MOTION:
        return None

    _avg_x, _avg_y, avg_vx, avg_vy, avg_spread = avg_pose_frames(frames)
    handedness = most_common_handedness(frames)

    abs_vx = abs(avg_vx)
    abs_vy = abs(avg_vy)

    is_vertical = abs_vy >= abs_vx * ORIENT_VERTICAL_RATIO
    is_horizontal = abs_vx >= abs_vy * ORIENT_HORIZONTAL_RATIO

    if is_vertical:
        if avg_spread <= DOWN_SPREAD_RATIO_MAX:
            return "up"
        if avg_spread >= UP_SPREAD_RATIO_MIN:
            return "down"
        return None

    if is_horizontal:
        if handedness.lower() == "right":
            return "back"
        if handedness.lower() == "left":
            return "next"
        if avg_vx > 0:
            return "back"
        if avg_vx < 0:
            return "next"
    return None


def detect_two_finger_gesture(
    state: ControlState,
    stable_static_pred: str,
    handedness: str,
    landmarks: Any,
    pose_buffer: deque[PoseFrame],
) -> tuple[str, str | None]:
    now = time.perf_counter()

    if not state.active:
        state.dynamic_collecting = False
        pose_buffer.clear()
        return "inactive", None

    if stable_static_pred != DYNAMIC_GATE_LABEL:
        state.dynamic_collecting = False
        pose_buffer.clear()
        return f"waiting two_fingers ({stable_static_pred})", None

    if now - state.last_navigation_ts < NAVIGATION_ACTION_COOLDOWN_SEC:
        state.dynamic_collecting = False
        pose_buffer.clear()
        remain = NAVIGATION_ACTION_COOLDOWN_SEC - (now - state.last_navigation_ts)
        return f"navigation cooldown {remain:.1f}s", None

    pose = two_finger_pose_features(landmarks)
    if pose is None:
        state.dynamic_collecting = False
        pose_buffer.clear()
        return "invalid two_finger pose", None

    x, y, vx, vy, spread_ratio = pose

    if not state.dynamic_collecting:
        state.dynamic_collecting = True
        pose_buffer.clear()

    pose_buffer.append((x, y, vx, vy, spread_ratio, handedness))

    if now - state.last_dynamic_ts < DYNAMIC_COOLDOWN_SEC:
        return f"cooldown {len(pose_buffer)}/{pose_buffer.maxlen}", None

    gesture = classify_two_finger_example_pose(pose_buffer)
    if gesture is None:
        return f"collecting {len(pose_buffer)}/{pose_buffer.maxlen}", None

    return gesture, gesture


def mark_dynamic_action_executed(
    state: ControlState,
    gesture: str,
    pose_buffer: deque[PoseFrame],
) -> None:
    now = time.perf_counter()
    state.last_dynamic_ts = now
    if gesture in {"back", "next"}:
        state.last_navigation_ts = now
    state.dynamic_collecting = False
    pose_buffer.clear()


def reset_dynamic_runtime(dynamic_history: deque[str], pose_buffer: deque[PoseFrame]) -> None:
    dynamic_history.clear()
    pose_buffer.clear()
