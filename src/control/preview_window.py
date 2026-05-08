import sys
import time
import cv2 as cv

from .config import (
    FIXED_PREVIEW_WINDOW,
    MIN_RENDER_WINDOW_SIZE,
    PREVIEW_BASE_HEIGHT,
    PREVIEW_BASE_WIDTH,
    PREVIEW_COMPACT_RENDER,
    PREVIEW_MARGIN_BOTTOM,
    PREVIEW_MARGIN_RIGHT,
    PREVIEW_PIN_TO_BOTTOM_RIGHT,
    PREVIEW_SCALE_DEFAULT,
    PREVIEW_SCALE_MAX,
    PREVIEW_SCALE_MIN,
    PREVIEW_SCALE_STEP,
    PREVIEW_TOPMOST,
    SKIP_PREVIEW_WHEN_MINIMIZED,
)
from .state import ControlState


def get_preview_size(state: ControlState | None = None) -> tuple[int, int]:
    scale = state.preview_scale if state is not None else PREVIEW_SCALE_DEFAULT
    width = int(PREVIEW_BASE_WIDTH * scale)
    height = int(PREVIEW_BASE_HEIGHT * scale)
    return max(320, width), max(180, height)


def configure_fixed_preview_window(window_name: str, state: ControlState | None = None, force: bool = False) -> None:
    if not FIXED_PREVIEW_WINDOW:
        return

    now = time.perf_counter()
    if not force and state is not None and now - state.last_preview_window_fix_ts < 0.50:
        return

    if state is not None:
        state.last_preview_window_fix_ts = now

    preview_w, preview_h = get_preview_size(state)

    try:
        if force:
            cv.resizeWindow(window_name, preview_w, preview_h)
        if PREVIEW_TOPMOST and hasattr(cv, "WND_PROP_TOPMOST"):
            cv.setWindowProperty(window_name, cv.WND_PROP_TOPMOST, 1)
    except Exception:
        return

    if sys.platform != "win32":
        return

    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return

        HWND_TOPMOST = -1
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOACTIVATE = 0x0010
        SWP_FRAMECHANGED = 0x0020

        if force and PREVIEW_PIN_TO_BOTTOM_RIGHT:
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)
            x = max(0, screen_w - preview_w - PREVIEW_MARGIN_RIGHT)
            y = max(0, screen_h - preview_h - PREVIEW_MARGIN_BOTTOM)

            user32.SetWindowPos(
                hwnd,
                HWND_TOPMOST if PREVIEW_TOPMOST else 0,
                x,
                y,
                preview_w,
                preview_h,
                SWP_NOACTIVATE | SWP_FRAMECHANGED,
            )
        else:
            user32.SetWindowPos(
                hwnd,
                HWND_TOPMOST if PREVIEW_TOPMOST else 0,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED,
            )
    except Exception:
        return


def make_preview_frame(frame, state: ControlState):
    if not PREVIEW_COMPACT_RENDER:
        return frame
    preview_w, preview_h = get_preview_size(state)
    return cv.resize(frame, (preview_w, preview_h), interpolation=cv.INTER_AREA)


def is_preview_renderable(window_name: str) -> bool:
    if not SKIP_PREVIEW_WHEN_MINIMIZED:
        return True

    try:
        visible = cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) >= 1
    except Exception:
        return False

    if not visible:
        return False

    try:
        _x, _y, w, h = cv.getWindowImageRect(window_name)
        if w < MIN_RENDER_WINDOW_SIZE or h < MIN_RENDER_WINDOW_SIZE:
            return False
    except Exception:
        pass
    return True


def change_preview_scale(state: ControlState, delta: float) -> None:
    state.preview_scale = max(PREVIEW_SCALE_MIN, min(PREVIEW_SCALE_MAX, state.preview_scale + delta))
    state.last_action = f"Preview zoom {state.preview_scale:.2f}x"


def zoom_step() -> float:
    return PREVIEW_SCALE_STEP
