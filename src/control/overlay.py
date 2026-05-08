from collections import deque
from typing import Any
import cv2 as cv
import numpy as np

from .config import DRAW_MOUSE_PREVIEW, GUIDE_PANEL_WIDTH
from .hand_utils import landmark_xy
from .state import ControlState

from pathlib import Path
from .config import DRAW_MOUSE_PREVIEW, GUIDE_PANEL_WIDTH, ICON_DIR

GuideItem = tuple[str, str]

ICON_PATHS = {
    "open_palm": ICON_DIR / "open_palm.png",
    "fist": ICON_DIR / "fist.png",
    "point": ICON_DIR / "point.png",
    "pinch": ICON_DIR / "pinch.png",
    "tf_up": ICON_DIR / "tf_up.png",
    "tf_down": ICON_DIR / "tf_down.png",
    "tf_back": ICON_DIR / "tf_back.png",
    "tf_next": ICON_DIR / "tf_next.png",
}


_loaded_icons: dict[str, np.ndarray] = {}


def load_icon(icon_key: str) -> np.ndarray | None:
    if icon_key in _loaded_icons:
        return _loaded_icons[icon_key]

    path = ICON_PATHS.get(icon_key)
    if path is None or not Path(path).exists():
        _loaded_icons[icon_key] = None
        return None

    icon = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    _loaded_icons[icon_key] = icon
    return icon


def paste_rgba_icon(
    dst: np.ndarray,
    icon: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    if icon is None:
        return

    resized = cv.resize(icon, (w, h), interpolation=cv.INTER_AREA)

    if resized.shape[2] == 4:
        bgr = resized[:, :, :3]
        alpha = resized[:, :, 3] / 255.0
    else:
        bgr = resized
        alpha = np.ones((h, w), dtype=np.float32)

    y1 = max(0, y)
    y2 = min(dst.shape[0], y + h)
    x1 = max(0, x)
    x2 = min(dst.shape[1], x + w)

    if y1 >= y2 or x1 >= x2:
        return

    icon_crop = bgr[0 : y2 - y1, 0 : x2 - x1]
    alpha_crop = alpha[0 : y2 - y1, 0 : x2 - x1]

    for c in range(3):
        dst[y1:y2, x1:x2, c] = (
            alpha_crop * icon_crop[:, :, c] + (1.0 - alpha_crop) * dst[y1:y2, x1:x2, c]
        )


def draw_landmarks(frame: np.ndarray, landmarks: Any) -> np.ndarray:
    canvas = frame.copy()
    if not landmarks:
        return canvas

    h, w = canvas.shape[:2]
    points = []

    for lm in landmarks:
        x = getattr(lm, "x", None)
        y = getattr(lm, "y", None)
        if x is None or y is None:
            continue
        points.append((int(float(x) * w), int(float(y) * h)))

    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    for a, b in connections:
        if a < len(points) and b < len(points):
            cv.line(canvas, points[a], points[b], (0, 255, 0), 2)

    for idx, (px, py) in enumerate(points):
        cv.circle(
            canvas, (px, py), 5 if idx in {0, 4, 8, 12, 16, 20} else 4, (0, 0, 255), -1
        )

    return canvas


def draw_mouse_preview(frame: np.ndarray, landmarks: Any) -> None:
    if not DRAW_MOUSE_PREVIEW or landmarks is None:
        return
    pt = landmark_xy(landmarks, 8)
    if pt is None:
        return
    h, w = frame.shape[:2]
    px, py = int(pt[0] * w), int(pt[1] * h)
    cv.circle(frame, (px, py), 12, (255, 0, 255), 2)


def draw_text_panel(
    frame: np.ndarray,
    fps: float,
    mode_name: str,
    label_text: str,
    state: ControlState,
    action_history: deque[str],
    show_panel: bool,
) -> None:
    if not show_panel:
        return
    lines = [
        f"FPS: {fps:.1f}",
        f"Mode: {mode_name}",
        f"Label: {label_text}",
        f"Action: {state.last_action}",
        f"Action history: {list(action_history)}",
    ]
    x, y = 20, 35
    for line in lines:
        cv.putText(
            frame,
            line,
            (x, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )
        y += 30


def build_preview_canvas(
    frame: np.ndarray,
    fps: float,
    mode_name: str,
    label_text: str,
    state: ControlState,
    action_history: deque[str],
    show_panel: bool,
    mode_key_hints: list[str],
    mode_guide_items: list[GuideItem],
) -> np.ndarray:
    main = frame.copy()
    draw_text_panel(main, fps, mode_name, label_text, state, action_history, show_panel)
    h, w = main.shape[:2]
    panel = np.full((h, GUIDE_PANEL_WIDTH, 3), 28, dtype=np.uint8)
    draw_guide_panel(panel, mode_name, mode_key_hints, mode_guide_items)
    return np.hstack([main, panel])


def draw_guide_panel(
    panel: np.ndarray,
    mode_name: str,
    mode_key_hints: list[str],
    mode_guide_items: list[GuideItem],
) -> None:
    h, w = panel.shape[:2]
    cv.rectangle(panel, (0, 0), (w - 1, h - 1), (70, 70, 70), 2)

    y = 28
    cv.putText(
        panel,
        "GUIDE",
        (18, y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    y += 32
    cv.putText(
        panel,
        f"Mode: {mode_name}",
        (18, y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv.LINE_AA,
    )
    y += 26

    for hint in mode_key_hints:
        cv.putText(
            panel,
            hint,
            (18, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.50,
            (190, 190, 190),
            1,
            cv.LINE_AA,
        )
        y += 20

    y += 10
    cv.putText(
        panel,
        "Global",
        (18, y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.58,
        (120, 255, 120),
        2,
        cv.LINE_AA,
    )
    y += 10

    global_items: list[GuideItem] = [
        ("open_palm", "Open palm -> ACTIVE"),
        ("fist", "Fist -> LOCK"),
    ]

    for item in global_items:
        y = draw_gesture_card(panel, 14, y + 8, w - 28, 86, item[0], item[1])
        y += 8

    y += 28
    cv.putText(
        panel,
        "Mode actions",
        (18, y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.58,
        (120, 255, 120),
        2,
        cv.LINE_AA,
    )
    y += 14

    for icon_key, text in mode_guide_items:
        y = draw_gesture_card(panel, 14, y + 6, w - 28, 62, icon_key, text)
        y += 6
        if y > h - 72:
            break


def draw_gesture_card(
    panel: np.ndarray, x: int, y: int, w: int, h: int, icon_key: str, text: str
) -> int:
    cv.rectangle(panel, (x, y), (x + w, y + h), (55, 55, 55), -1)
    cv.rectangle(panel, (x, y), (x + w, y + h), (95, 95, 95), 1)

    icon_box = (x + 8, y + 8, 72, h - 16)
    # icon_box = (x + 10, y + 8, 84, 84)
    draw_gesture_icon(panel, icon_box, icon_key)

    text_x = x + 92
    text_y = y + 23
    draw_multiline_text(panel, text, text_x, text_y, max_width=w - 102)
    return y + h


def draw_multiline_text(
    panel: np.ndarray, text: str, x: int, y: int, max_width: int
) -> None:
    words = text.split()
    if not words:
        return

    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        width = cv.getTextSize(candidate, cv.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    for i, line in enumerate(lines[:2]):
        cv.putText(
            panel,
            line,
            (x, y + i * 18),
            cv.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )


def draw_gesture_icon(
    panel: np.ndarray, box: tuple[int, int, int, int], icon_key: str
) -> None:
    x, y, w, h = box
    cv.rectangle(panel, (x, y), (x + w, y + h), (34, 34, 34), -1)
    cv.rectangle(panel, (x, y), (x + w, y + h), (80, 80, 80), 1)

    icon = load_icon(icon_key)
    if icon is not None:
        paste_rgba_icon(panel, icon, x + 4, y + 4, w - 8, h - 8)
        return

    # fallback nếu không có ảnh
    if icon_key == "open_palm":
        draw_open_palm_icon(panel, x, y, w, h)
    elif icon_key == "fist":
        draw_fist_icon(panel, x, y, w, h)
    elif icon_key == "point":
        draw_point_icon(panel, x, y, w, h)
    elif icon_key == "pinch":
        draw_pinch_icon(panel, x, y, w, h)
    elif icon_key == "tf_up":
        draw_two_finger_vertical_icon(panel, x, y, w, h, spread="close", arrow="up")
    elif icon_key == "tf_down":
        draw_two_finger_vertical_icon(panel, x, y, w, h, spread="open", arrow="down")
    elif icon_key == "tf_back":
        draw_two_finger_horizontal_icon(panel, x, y, w, h, direction="left")
    elif icon_key == "tf_next":
        draw_two_finger_horizontal_icon(panel, x, y, w, h, direction="right")


def draw_open_palm_icon(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    cx = x + w // 2
    base_y = y + int(h * 0.72)
    palm_top = y + int(h * 0.45)
    cv.rectangle(img, (cx - 14, palm_top), (cx + 14, base_y), (210, 210, 210), 2)
    finger_xs = [cx - 18, cx - 9, cx, cx + 9, cx + 18]
    finger_tops = [y + 8, y + 4, y + 2, y + 4, y + 8]
    for fx, ft in zip(finger_xs, finger_tops):
        cv.line(img, (fx, palm_top), (fx, ft), (210, 210, 210), 2)
    cv.line(img, (cx - 14, palm_top + 12), (cx - 28, palm_top + 22), (210, 210, 210), 2)


def draw_fist_icon(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    cx = x + w // 2
    top = y + int(h * 0.28)
    bottom = y + int(h * 0.78)
    cv.rectangle(img, (cx - 18, top + 10), (cx + 18, bottom), (210, 210, 210), 2)
    for i in range(4):
        px = cx - 16 + i * 10
        cv.rectangle(img, (px, top), (px + 8, top + 12), (210, 210, 210), 2)
    cv.line(img, (cx - 18, top + 26), (cx - 30, top + 34), (210, 210, 210), 2)


def draw_point_icon(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    cx = x + w // 2
    palm_top = y + int(h * 0.52)
    bottom = y + int(h * 0.80)
    cv.rectangle(img, (cx - 14, palm_top), (cx + 14, bottom), (210, 210, 210), 2)
    cv.line(img, (cx, palm_top), (cx, y + 4), (210, 210, 210), 2)
    cv.line(img, (cx - 8, palm_top), (cx - 14, palm_top - 12), (210, 210, 210), 2)
    cv.line(img, (cx + 8, palm_top), (cx + 14, palm_top - 10), (210, 210, 210), 2)
    cv.line(img, (cx - 14, palm_top + 12), (cx - 26, palm_top + 20), (210, 210, 210), 2)


def draw_pinch_icon(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    cx = x + w // 2
    cy = y + h // 2 + 2
    cv.circle(img, (cx - 10, cy - 8), 6, (210, 210, 210), 2)
    cv.circle(img, (cx + 2, cy - 8), 6, (210, 210, 210), 2)
    cv.line(img, (cx - 4, cy - 8), (cx - 2, cy - 8), (0, 255, 255), 2)
    cv.rectangle(img, (cx - 16, cy + 2), (cx + 10, cy + 18), (210, 210, 210), 2)
    cv.line(img, (cx - 16, cy + 10), (cx - 28, cy + 2), (210, 210, 210), 2)


def draw_two_finger_vertical_icon(
    img: np.ndarray, x: int, y: int, w: int, h: int, spread: str, arrow: str
) -> None:
    cx = x + w // 2
    bottom = y + int(h * 0.80)
    palm_top = y + int(h * 0.56)
    gap = 8 if spread == "close" else 16
    cv.rectangle(img, (cx - 14, palm_top), (cx + 14, bottom), (210, 210, 210), 2)
    cv.line(img, (cx - gap // 2, palm_top), (cx - gap // 2, y + 6), (210, 210, 210), 2)
    cv.line(img, (cx + gap // 2, palm_top), (cx + gap // 2, y + 4), (210, 210, 210), 2)
    cv.line(img, (cx - 14, palm_top + 12), (cx - 26, palm_top + 18), (210, 210, 210), 2)
    if arrow == "up":
        cv.arrowedLine(
            img, (x + 10, y + h - 12), (x + 10, y + 12), (0, 255, 255), 2, tipLength=0.3
        )
    else:
        cv.arrowedLine(
            img, (x + 10, y + 12), (x + 10, y + h - 12), (0, 255, 255), 2, tipLength=0.3
        )


def draw_two_finger_horizontal_icon(
    img: np.ndarray, x: int, y: int, w: int, h: int, direction: str
) -> None:
    cx = x + w // 2
    cy = y + h // 2
    cv.rectangle(img, (cx - 16, cy - 12), (cx + 12, cy + 12), (210, 210, 210), 2)
    cv.line(img, (cx - 2, cy - 4), (x + 8, cy - 4), (210, 210, 210), 2)
    cv.line(img, (cx - 2, cy + 4), (x + 8, cy + 4), (210, 210, 210), 2)
    cv.line(img, (cx + 8, cy + 10), (cx + 18, cy + 18), (210, 210, 210), 2)
    if direction == "left":
        cv.arrowedLine(
            img, (x + w - 10, cy), (x + 18, cy), (0, 255, 255), 2, tipLength=0.3
        )
    else:
        cv.arrowedLine(
            img, (x + 18, cy), (x + w - 10, cy), (0, 255, 255), 2, tipLength=0.3
        )
