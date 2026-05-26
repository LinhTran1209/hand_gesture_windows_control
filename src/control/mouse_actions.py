import math
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

# Time-based smoothing: tương đương MOUSE_SMOOTHING=0.2 ở 60 FPS,
# nhưng không bị lag thêm khi FPS tụt do preview/render.
_MOUSE_TIME_CONSTANT: float = -1.0 / (60.0 * math.log(1.0 - MOUSE_SMOOTHING))
_screen_size_cache: tuple[int, int] | None = None


def _get_screen_size(pyautogui_module) -> tuple[int, int]:
    global _screen_size_cache
    if _screen_size_cache is None:
        _screen_size_cache = pyautogui_module.size()
    return _screen_size_cache


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def normalized_point_to_screen(
    nx: float, ny: float, screen_w: int, screen_h: int
) -> tuple[int, int]:
    x = (nx - POINTER_MARGIN_X) / max(EPS, 1.0 - 2 * POINTER_MARGIN_X)
    y = (ny - POINTER_MARGIN_Y) / max(EPS, 1.0 - 2 * POINTER_MARGIN_Y)
    x = clamp01(x)
    y = clamp01(y)
    return int(x * (screen_w - 1)), int(y * (screen_h - 1))


def smooth_mouse_target(state: ControlState, tx: int, ty: int) -> tuple[int, int]:
    """EMA smoothing theo thời gian thật, không phụ thuộc FPS."""
    now = time.perf_counter()

    if state.mouse_x is None or state.mouse_y is None:
        state.mouse_x = float(tx)
        state.mouse_y = float(ty)
        state.last_mouse_ts = now
        return tx, ty

    dt = max(1e-4, min(now - state.last_mouse_ts, 0.5))
    alpha = 1.0 - math.exp(-dt / _MOUSE_TIME_CONSTANT)

    state.mouse_x = (1.0 - alpha) * state.mouse_x + alpha * float(tx)
    state.mouse_y = (1.0 - alpha) * state.mouse_y + alpha * float(ty)
    state.last_mouse_ts = now

    return int(state.mouse_x), int(state.mouse_y)


def maybe_move_mouse(
    state: ControlState,
    stable_static_pred: str,
    landmarks: Any,
    pyautogui_module,
) -> None:
    if MOVE_ONLY_WHEN_ACTIVE and not state.active:
        return
    if stable_static_pred != MOVE_LABEL:
        return

    pt = landmark_xy(landmarks, 8)
    if pt is None:
        return

    nx = clamp01(pt[0] + POINT_CURSOR_OFFSET_X)
    ny = clamp01(pt[1] + POINT_CURSOR_OFFSET_Y)

    screen_w, screen_h = _get_screen_size(pyautogui_module)
    target_x, target_y = normalized_point_to_screen(nx, ny, screen_w, screen_h)

    mx, my = smooth_mouse_target(state, target_x, target_y)
    current_x, current_y = pyautogui_module.position()

    dx = abs(mx - current_x) / max(screen_w, 1)
    dy = abs(my - current_y) / max(screen_h, 1)

    if dx < MOVE_DEADZONE and dy < MOVE_DEADZONE:
        return

    pyautogui_module.moveTo(mx, my, duration=0)
    if state.mouse_left_down:
        state.last_action = "Drag / select"
    else:
        state.last_action = "Move mouse"


def is_pinch_touching(landmarks: Any) -> tuple[bool, float]:
    thumb_tip = landmark_xy(landmarks, 4)
    index_tip = landmark_xy(landmarks, 8)
    if thumb_tip is None or index_tip is None:
        return False, 999.0

    distance = float(np.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1]))
    return distance <= PINCH_TOUCH_THRESHOLD, distance


def release_mouse_left_if_down(
    state: ControlState,
    pyautogui_module,
    reason: str = "Release left mouse",
) -> bool:
    """Nhả chuột trái nếu đang bị giữ bởi thao tác drag/select."""
    if not state.mouse_left_down:
        return False

    try:
        pyautogui_module.mouseUp(button="left")
        state.last_action = reason
    except Exception as exc:
        state.last_action = f"MouseUp error: {type(exc).__name__}"
    finally:
        state.mouse_left_down = False
        state.drag_source = None
        state.secondary_pinch_touching = False

    return True


def maybe_handle_secondary_pinch_drag(
    state: ControlState,
    primary_static_pred: str,
    secondary_static_pred: str,
    secondary_landmarks: Any,
    use_pointer_move: bool,
    pyautogui_module,
) -> bool:
    """
    Dùng 2 tay như chuột thật:
    - Tay chính point: di chuyển con trỏ.
    - Tay phụ pinch: giữ chuột trái.
    - Giữ pinch + di chuyển point: kéo thả / tô đen chữ.
    - Thả pinch: nhả chuột trái.

    Trả về True nếu pinch tay phụ đã được xử lý bởi cơ chế drag/select,
    để app không gọi thêm click thường cho cùng một pinch.
    """
    can_drag = state.active and use_pointer_move and primary_static_pred == MOVE_LABEL

    if (
        not can_drag
        or secondary_landmarks is None
        or secondary_static_pred != CLICK_LABEL
    ):
        release_mouse_left_if_down(state, pyautogui_module, "Release left mouse")
        return False

    touching, distance = is_pinch_touching(secondary_landmarks)

    if state.mouse_left_down:
        if distance >= PINCH_RELEASE_THRESHOLD:
            release_mouse_left_if_down(state, pyautogui_module, "Release left mouse")
        else:
            state.secondary_pinch_touching = True
            state.last_action = "Hold left mouse"
        return True

    if not touching:
        state.secondary_pinch_touching = False
        return False

    try:
        pyautogui_module.mouseDown(button="left")
        state.mouse_left_down = True
        state.drag_source = "secondary_pinch"
        state.secondary_pinch_touching = True
        state.last_action = "Hold left mouse"
        return True
    except Exception as exc:
        state.last_action = f"MouseDown error: {type(exc).__name__}"
        state.mouse_left_down = False
        state.drag_source = None
        state.secondary_pinch_touching = False
        return True


def maybe_trigger_pinch_action(
    state: ControlState,
    stable_static_pred: str,
    landmarks: Any,
    source: str,
    action_callback: Callable[[], tuple[bool, str]],
) -> None:
    flag_name = (
        "secondary_pinch_touching"
        if source == "secondary"
        else "primary_pinch_touching"
    )

    if not state.active:
        state.primary_pinch_touching = False
        state.secondary_pinch_touching = False
        return

    # Nếu đang giữ chuột trái để kéo/thả thì không phát sinh click thường nữa.
    if state.mouse_left_down:
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
