import time
from typing import Any, Callable
import numpy as np

from .config import (
    ACTIVE_LABEL,
    CLICK_COOLDOWN_SEC,
    CLICK_LABEL,
    EPS,
    LOCK_LABEL,
    MODE_SWITCH_COOLDOWN_SEC,
    MOVE_DEADZONE,
    MOVE_LABEL,
    MOVE_ONLY_WHEN_ACTIVE,
    MOUSE_SMOOTHING,
    PINCH_RELEASE_THRESHOLD,
    PINCH_TOUCH_THRESHOLD,
    POINTER_MARGIN_X,
    POINTER_MARGIN_Y,
    POINT_CURSOR_OFFSET_X,
    POINT_CURSOR_OFFSET_Y,
)
from .hand_utils import landmark_xy
from .state import ControlState


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def normalized_point_to_screen(nx: float, ny: float, screen_w: int, screen_h: int) -> tuple[int, int]:
    x = (nx - POINTER_MARGIN_X) / max(EPS, 1.0 - 2 * POINTER_MARGIN_X)
    y = (ny - POINTER_MARGIN_Y) / max(EPS, 1.0 - 2 * POINTER_MARGIN_Y)
    x = clamp01(x)
    y = clamp01(y)
    return int(x * (screen_w - 1)), int(y * (screen_h - 1))


def smooth_mouse_target(state: ControlState, tx: int, ty: int) -> tuple[int, int]:
    if state.mouse_x is None or state.mouse_y is None:
        state.mouse_x = float(tx)
        state.mouse_y = float(ty)
    else:
        alpha = MOUSE_SMOOTHING
        state.mouse_x = (1.0 - alpha) * state.mouse_x + alpha * tx
        state.mouse_y = (1.0 - alpha) * state.mouse_y + alpha * ty
    return int(state.mouse_x), int(state.mouse_y)


def maybe_move_mouse(state: ControlState, stable_static_pred: str, landmarks: Any, pyautogui_module) -> None:
    if MOVE_ONLY_WHEN_ACTIVE and not state.active:
        return
    if stable_static_pred != MOVE_LABEL:
        return

    pt = landmark_xy(landmarks, 8)
    if pt is None:
        return

    nx = clamp01(pt[0] + POINT_CURSOR_OFFSET_X)
    ny = clamp01(pt[1] + POINT_CURSOR_OFFSET_Y)

    screen_w, screen_h = pyautogui_module.size()
    target_x, target_y = normalized_point_to_screen(nx, ny, screen_w, screen_h)

    mx, my = smooth_mouse_target(state, target_x, target_y)
    current_x, current_y = pyautogui_module.position()

    dx = abs(mx - current_x) / max(screen_w, 1)
    dy = abs(my - current_y) / max(screen_h, 1)

    if dx < MOVE_DEADZONE and dy < MOVE_DEADZONE:
        return

    pyautogui_module.moveTo(mx, my, duration=0)
    state.last_action = "Move mouse"


def is_pinch_touching(landmarks: Any) -> tuple[bool, float]:
    thumb_tip = landmark_xy(landmarks, 4)
    index_tip = landmark_xy(landmarks, 8)
    if thumb_tip is None or index_tip is None:
        return False, 999.0

    distance = float(np.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1]))
    return distance <= PINCH_TOUCH_THRESHOLD, distance


def maybe_trigger_pinch_action(
    state: ControlState,
    stable_static_pred: str,
    landmarks: Any,
    source: str,
    action_callback: Callable[[], tuple[bool, str]],
) -> None:
    flag_name = "secondary_pinch_touching" if source == "secondary" else "primary_pinch_touching"

    if not state.active:
        state.primary_pinch_touching = False
        state.secondary_pinch_touching = False
        return

    if stable_static_pred != CLICK_LABEL or landmarks is None:
        setattr(state, flag_name, False)
        return

    touching, distance = is_pinch_touching(landmarks)
    was_touching = bool(getattr(state, flag_name))

    if was_touching and distance >= PINCH_RELEASE_THRESHOLD:
        was_touching = False
        setattr(state, flag_name, False)

    if not touching or was_touching:
        return

    now = time.perf_counter()
    if now - state.last_click_ts < CLICK_COOLDOWN_SEC:
        return

    ok, msg = action_callback()
    state.last_action = msg

    if ok:
        state.last_click_ts = now
        setattr(state, flag_name, True)


def maybe_toggle_active(state: ControlState, stable_static_pred: str) -> None:
    now = time.perf_counter()
    if now - state.last_mode_ts < MODE_SWITCH_COOLDOWN_SEC:
        return

    if stable_static_pred == ACTIVE_LABEL and not state.active:
        state.active = True
        state.last_mode_ts = now
        state.last_action = "ACTIVE"

    elif stable_static_pred == LOCK_LABEL and state.active:
        state.active = False
        state.last_mode_ts = now
        state.last_action = "LOCK"
        state.dynamic_collecting = False
