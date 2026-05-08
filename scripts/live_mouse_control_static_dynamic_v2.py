from __future__ import annotations
from dataclasses import dataclass

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print(f"[INFO] PROJECT_ROOT set to: {PROJECT_ROOT}")

import inspect
import json
import time
import traceback
from collections import Counter, deque
from typing import Any

import cv2 as cv
import joblib
import numpy as np
import pyautogui

from src.capture.webcam import WebcamCapture, WebcamConfig
from src.perception.hand_tracker import HandTracker
from src.features.hand_landmark_features import landmarks_to_feature_vector

STATIC_MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "static_best_model.joblib"
STATIC_MODEL_META_PATH = (
    PROJECT_ROOT / "models" / "checkpoints" / "static_best_model_meta.json"
)
HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

# Static model labels expected:
# open_palm, fist, point, pinch, no_gesture, two_fingers
ACTIVE_LABEL = "open_palm"
LOCK_LABEL = "fist"
MOVE_LABEL = "point"
CLICK_LABEL = "pinch"
DYNAMIC_GATE_LABEL = "two_fingers"

# Static smoothing.
SMOOTHING_WINDOW = 7
SMOOTHING_MIN_VOTES = 5
NO_HAND_RESET_FRAMES = 8

# Mouse control.
MOUSE_SMOOTHING = 0.2
CLICK_COOLDOWN_SEC = 0.60
PINCH_TOUCH_THRESHOLD = 0.055
PINCH_RELEASE_THRESHOLD = 0.075
MODE_SWITCH_COOLDOWN_SEC = 0.80
MOVE_ONLY_WHEN_ACTIVE = True
MOVE_DEADZONE = 0.001
POINTER_MARGIN_X = 0.05
POINTER_MARGIN_Y = 0.05
POINT_CURSOR_OFFSET_X = 0.0
POINT_CURSOR_OFFSET_Y = 0.0
PRIMARY_SWITCH_MAX_DISTANCE = 0.35

# Rule-based two_fingers actions based on the 4 example poses.
# Hình 1/2 đã đảo lại theo yêu cầu mới:
# - vertical two_fingers, fingers close together -> scroll up
# - vertical two_fingers, fingers spread wider   -> scroll down
# - horizontal two_fingers, handedness Right     -> back
# - horizontal two_fingers, handedness Left      -> next
DYNAMIC_COOLDOWN_SEC = 0.65
NAVIGATION_ACTION_COOLDOWN_SEC = 3.0
POSE_BUFFER_SIZE = 12
POSE_MIN_FRAMES = 5
POSE_MAX_MOTION = 0.040
ORIENT_VERTICAL_RATIO = 1.22
ORIENT_HORIZONTAL_RATIO = 1.18
UP_SPREAD_RATIO_MIN = 0.22
DOWN_SPREAD_RATIO_MAX = 0.17
SCROLL_AMOUNT = 120

# Performance / preview.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60
PREVIEW_BASE_WIDTH = 1280
PREVIEW_BASE_HEIGHT = 720
PREVIEW_MARGIN_RIGHT = 18
PREVIEW_MARGIN_BOTTOM = 58
PREVIEW_WINDOW_NAME = "Hand Gesture Control"
DRAW_MOUSE_PREVIEW = False
DRAW_LANDMARKS_ON_PREVIEW = True
DRAW_DEBUG_PANEL = True
RENDER_EVERY_N_FRAMES = 1

# Always-on-top preview, but still allow maximize/minimize and mouse resizing.
# Use + / - keys to zoom the rendered preview.
FIXED_PREVIEW_WINDOW = True
PREVIEW_TOPMOST = True
PREVIEW_COMPACT_RENDER = True
SKIP_PREVIEW_WHEN_MINIMIZED = False
MIN_RENDER_WINDOW_SIZE = 64
PREVIEW_SCALE_DEFAULT = 1.0
PREVIEW_SCALE_MIN = 0.45
PREVIEW_SCALE_MAX = 1.80
PREVIEW_SCALE_STEP = 0.10
PREVIEW_PIN_TO_BOTTOM_RIGHT = True

VERBOSE_DYNAMIC_LOG = False
LOG_COOLDOWN_SEC = 0.35
ACTION_HISTORY_SIZE = 6
EPS = 1e-6

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


@dataclass
class ControlState:
    active: bool = False
    last_click_ts: float = 0.0
    primary_pinch_touching: bool = False
    secondary_pinch_touching: bool = False
    last_dynamic_ts: float = 0.0
    last_navigation_ts: float = 0.0
    last_mode_ts: float = 0.0
    last_action: str = "None"
    mouse_x: float | None = None
    mouse_y: float | None = None
    prev_stable_static_pred: str = "None"
    prev_stable_dynamic_pred: str = "None"
    dynamic_collecting: bool = False
    primary_center: tuple[float, float] | None = None
    preview_scale: float = PREVIEW_SCALE_DEFAULT
    last_preview_window_fix_ts: float = 0.0
    last_log_ts: float = 0.0


def log_dynamic(state: ControlState, message: str, force: bool = False) -> None:
    if not VERBOSE_DYNAMIC_LOG:
        return
    now = time.perf_counter()
    if force or (now - state.last_log_ts >= LOG_COOLDOWN_SEC):
        print(f"[DYNAMIC] {message}")
        state.last_log_ts = now


def get_preview_size(state: ControlState | None = None) -> tuple[int, int]:
    scale = state.preview_scale if state is not None else PREVIEW_SCALE_DEFAULT
    width = int(PREVIEW_BASE_WIDTH * scale)
    height = int(PREVIEW_BASE_HEIGHT * scale)
    return max(320, width), max(180, height)


def get_landmarks(det: Any):
    if det is None:
        return None
    if hasattr(det, "landmarks"):
        return det.landmarks
    if isinstance(det, dict):
        return det.get("landmarks")
    return None


def get_handedness(det: Any):
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


def majority_vote(history: deque[str], min_votes: int) -> tuple[str | None, int]:
    if not history:
        return None, 0
    label, count = Counter(history).most_common(1)[0]
    if count >= min_votes:
        return label, count
    return None, count


def safe_pyautogui(action_name: str, fn) -> tuple[bool, str]:
    try:
        fn()
        return True, action_name
    except pyautogui.FailSafeException:
        return False, f"{action_name} blocked by FAILSAFE"
    except Exception as exc:
        return False, f"{action_name} error: {type(exc).__name__}"


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


def create_tracker() -> Any:
    init_sig = inspect.signature(HandTracker.__init__)
    supported = init_sig.parameters
    candidate_kwargs = {
        "model_path": HAND_MODEL_PATH,
        "max_num_hands": 2,
        "min_detection_confidence": 0.65,
        "min_presence_confidence": 0.65,
        "min_tracking_confidence": 0.65,
    }
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported}
    print("[INFO] HandTracker.__init__ signature:", init_sig)
    print("[INFO] Tracker kwargs:", kwargs)
    return HandTracker(**kwargs)


def unpack_process_result(
    result: Any, original_frame: np.ndarray
) -> tuple[np.ndarray, list[Any]]:
    if result is None:
        return original_frame.copy(), []
    if isinstance(result, tuple):
        if len(result) >= 2:
            annotated_frame = (
                result[0] if result[0] is not None else original_frame.copy()
            )
            maybe = result[-1]
            if maybe is None:
                return annotated_frame, []
            if isinstance(maybe, list):
                return annotated_frame, maybe
            return annotated_frame, [maybe]
        if len(result) == 1:
            return (
                original_frame.copy(),
                unpack_process_result(result[0], original_frame)[1],
            )
    if isinstance(result, list):
        return original_frame.copy(), result
    return original_frame.copy(), [result]


def configure_fixed_preview_window(
    window_name: str, state: ControlState | None = None, force: bool = False
) -> None:
    if not FIXED_PREVIEW_WINDOW:
        return
    now = time.perf_counter()
    if (
        not force
        and state is not None
        and now - state.last_preview_window_fix_ts < 0.50
    ):
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


def make_preview_frame(frame: np.ndarray, state: ControlState) -> np.ndarray:
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


def run_tracker(
    tracker: Any, frame: np.ndarray, timestamp_ms: int
) -> tuple[np.ndarray, list[Any]]:
    process_sig = inspect.signature(tracker.process)
    if not getattr(run_tracker, "_printed", False):
        print("[INFO] HandTracker.process signature:", process_sig)
        run_tracker._printed = True
    try:
        params = process_sig.parameters
        if "timestamp_ms" in params:
            result = tracker.process(frame, timestamp_ms=timestamp_ms)
        elif len(params) >= 2:
            result = tracker.process(frame, timestamp_ms)
        else:
            result = tracker.process(frame)
    except TypeError:
        try:
            result = tracker.process(frame, timestamp_ms)
        except TypeError:
            result = tracker.process(frame)
    return unpack_process_result(result, frame)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


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
    xs: list[float] = []
    ys: list[float] = []
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
    """Keep the same primary hand when the second hand appears.

    MediaPipe can change detection order when a new hand enters the frame.
    If we always use detections[0], the cursor can jump to the new hand.
    This function chooses the hand closest to the previous primary center.
    """
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
        return detections[0], detections[1] if len(detections) > 1 else None

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
            # The nearest hand is still too far from the old primary.
            # Keep the first detected hand instead of jumping to the other side.
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


def normalized_point_to_screen(
    nx: float, ny: float, screen_w: int, screen_h: int
) -> tuple[int, int]:
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


def maybe_move_mouse(
    state: ControlState, stable_static_pred: str, landmarks: Any
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
    screen_w, screen_h = pyautogui.size()
    target_x, target_y = normalized_point_to_screen(nx, ny, screen_w, screen_h)
    mx, my = smooth_mouse_target(state, target_x, target_y)
    current_x, current_y = pyautogui.position()
    dx = abs(mx - current_x) / max(screen_w, 1)
    dy = abs(my - current_y) / max(screen_h, 1)
    if dx < MOVE_DEADZONE and dy < MOVE_DEADZONE:
        return
    _, msg = safe_pyautogui("Move mouse", lambda: pyautogui.moveTo(mx, my, duration=0))
    state.last_action = f"{msg} -> ({mx}, {my})"


def is_pinch_touching(landmarks: Any) -> tuple[bool, float]:
    """Check if thumb tip and index tip are really touching.

    The static model only says the hand looks like pinch. The click is fired
    only when landmark 4 (thumb tip) and landmark 8 (index tip) are close enough.
    """
    thumb_tip = landmark_xy(landmarks, 4)
    index_tip = landmark_xy(landmarks, 8)
    if thumb_tip is None or index_tip is None:
        return False, 999.0
    distance = float(np.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1]))
    return distance <= PINCH_TOUCH_THRESHOLD, distance


def maybe_left_click(
    state: ControlState,
    stable_static_pred: str,
    landmarks: Any,
    source: str = "primary",
) -> None:
    """Click only when the predicted pinch hand really touches thumb tip + index tip.

    source='primary'  : bàn tay chính, vẫn dùng được pinch 1 tay như cũ.
    source='secondary': bàn tay thứ 2, chỉ dùng để click, không điều khiển action khác.
    """
    flag_name = (
        "secondary_pinch_touching"
        if source == "secondary"
        else "primary_pinch_touching"
    )

    if not state.active:
        state.primary_pinch_touching = False
        state.secondary_pinch_touching = False
        return

    if stable_static_pred != CLICK_LABEL or landmarks is None:
        setattr(state, flag_name, False)
        return

    touching, distance = is_pinch_touching(landmarks)
    was_touching = bool(getattr(state, flag_name))

    # Release hysteresis: phải tách tay ra đủ xa thì lần chạm sau mới được click tiếp.
    if was_touching and distance >= PINCH_RELEASE_THRESHOLD:
        was_touching = False
        setattr(state, flag_name, False)

    # Chưa thật sự chạm thì không click, dù model đã nhận là pinch.
    if not touching:
        return

    # Đang giữ chạm thì không spam click liên tục.
    if was_touching:
        return

    now = time.perf_counter()
    if now - state.last_click_ts < CLICK_COOLDOWN_SEC:
        return

    action_name = "Left click" if source == "primary" else "Left click (second hand)"
    ok, msg = safe_pyautogui(action_name, lambda: pyautogui.click(button="left"))
    state.last_action = f"{msg} | pinch distance={distance:.3f}"
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


def load_model_predictor(
    model_path: Path, meta_path: Path, name: str
) -> tuple[Any, dict[str, Any]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {name} model: {model_path}")
    payload = joblib.load(model_path)
    meta: dict[str, Any] = {}
    if isinstance(payload, dict):
        meta.update(
            {
                "feature_columns": payload.get("feature_columns"),
                "model_name": payload.get("model_name"),
            }
        )
        label_encoder = payload.get("label_encoder")
        if label_encoder is not None and hasattr(label_encoder, "classes_"):
            meta["labels"] = list(label_encoder.classes_)
            meta["label_encoder"] = label_encoder
        model = None
        for key in (
            "pipeline",
            "model",
            "best_model",
            "estimator",
            "classifier",
            "clf",
        ):
            candidate = payload.get(key)
            if candidate is not None and hasattr(candidate, "predict"):
                model = candidate
                print(f"[INFO] {name} predictor from payload['{key}']")
                break
        if model is None and hasattr(payload, "predict"):
            model = payload
        if model is None:
            raise RuntimeError(
                f"No predictor with .predict() in {name} payload. Keys: {list(payload.keys())}"
            )
    else:
        model = payload
    if not hasattr(model, "predict"):
        raise RuntimeError(f"{name} model has no predict(): {type(model).__name__}")
    if meta_path.exists():
        try:
            meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[WARN] Cannot read {name} metadata: {type(exc).__name__}")
    print(f"[INFO] Loaded {name} model: {model_path}")
    print(f"[INFO] {name} model type: {type(model).__name__}")
    if meta.get("labels") is not None:
        print(f"[INFO] {name} labels: {meta.get('labels')}")
    return model, meta


def patch_simple_imputer_compat(
    estimator: Any, sample_feature: np.ndarray | None = None
) -> None:
    seen: set[int] = set()

    def walk(obj: Any) -> list[Any]:
        if obj is None or id(obj) in seen:
            return []
        seen.add(id(obj))
        found = [obj]
        steps = getattr(obj, "steps", None)
        if isinstance(steps, list):
            for _, step in steps:
                found.extend(walk(step))
        transformers = getattr(obj, "transformers", None)
        if isinstance(transformers, list):
            for item in transformers:
                if isinstance(item, tuple) and len(item) >= 2:
                    found.extend(walk(item[1]))
        transformer_list = getattr(obj, "transformer_list", None)
        if isinstance(transformer_list, list):
            for _, step in transformer_list:
                found.extend(walk(step))
        return found

    for obj in walk(estimator):
        if obj.__class__.__name__ != "SimpleImputer" or hasattr(obj, "_fill_dtype"):
            continue
        fill_dtype = None
        statistics = getattr(obj, "statistics_", None)
        if statistics is not None:
            try:
                fill_dtype = np.asarray(statistics).dtype
            except Exception:
                fill_dtype = None
        if fill_dtype is None and sample_feature is not None:
            try:
                fill_dtype = np.asarray(sample_feature).dtype
            except Exception:
                fill_dtype = np.float64
        obj._fill_dtype = fill_dtype or np.float64
        print(f"[WARN] Patched SimpleImputer._fill_dtype = {obj._fill_dtype}")


def predict_one_label(model: Any, feature: np.ndarray, meta: dict[str, Any]) -> str:
    patch_simple_imputer_compat(model, feature)
    pred = model.predict(feature)[0]
    label_encoder = meta.get("label_encoder")
    if label_encoder is not None and not isinstance(pred, str):
        try:
            pred = label_encoder.inverse_transform([pred])[0]
        except Exception:
            pass
    return str(pred)


def two_finger_pose_features(
    landmarks: Any,
) -> tuple[float, float, float, float, float] | None:
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
    tip_gap = float(
        np.hypot(index_tip[0] - middle_tip[0], index_tip[1] - middle_tip[1])
    )
    index_len = float(
        np.hypot(index_tip[0] - index_base[0], index_tip[1] - index_base[1])
    )
    middle_len = float(
        np.hypot(middle_tip[0] - middle_base[0], middle_tip[1] - middle_base[1])
    )
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


def classify_two_finger_example_pose(
    points: deque[tuple[float, float, float, float, float, str]],
) -> str | None:
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

    # Đã đổi ngược up/down:
    # ngón khép gần hơn -> up, ngón mở rộng hơn -> down.
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


def maybe_handle_two_finger_dynamic(
    state: ControlState,
    stable_static_pred: str,
    handedness: str,
    landmarks: Any,
    pose_buffer: deque[tuple[float, float, float, float, float, str]],
) -> tuple[str, str, int]:
    now = time.perf_counter()
    if not state.active:
        state.dynamic_collecting = False
        pose_buffer.clear()
        return "inactive", "None", 0
    if stable_static_pred != DYNAMIC_GATE_LABEL:
        state.dynamic_collecting = False
        pose_buffer.clear()
        return f"waiting two_fingers ({stable_static_pred})", "None", 0

    if now - state.last_navigation_ts < NAVIGATION_ACTION_COOLDOWN_SEC:
        state.dynamic_collecting = False
        pose_buffer.clear()
        remain = NAVIGATION_ACTION_COOLDOWN_SEC - (now - state.last_navigation_ts)
        return f"navigation cooldown {remain:.1f}s", "None", 0

    pose = two_finger_pose_features(landmarks)
    if pose is None:
        state.dynamic_collecting = False
        pose_buffer.clear()
        return "invalid two_finger pose", "None", 0
    x, y, vx, vy, spread_ratio = pose
    if not state.dynamic_collecting:
        state.dynamic_collecting = True
        pose_buffer.clear()
    pose_buffer.append((x, y, vx, vy, spread_ratio, handedness))
    if now - state.last_dynamic_ts < DYNAMIC_COOLDOWN_SEC:
        return f"cooldown {len(pose_buffer)}/{POSE_BUFFER_SIZE}", "None", 0
    gesture = classify_two_finger_example_pose(pose_buffer)
    if gesture is None:
        return f"collecting {len(pose_buffer)}/{POSE_BUFFER_SIZE}", "None", 0

    if gesture == "up":
        ok, msg = safe_pyautogui(
            f"Up scroll (+{SCROLL_AMOUNT})", lambda: pyautogui.scroll(SCROLL_AMOUNT)
        )
    elif gesture == "down":
        ok, msg = safe_pyautogui(
            f"Down scroll (-{SCROLL_AMOUNT})", lambda: pyautogui.scroll(-SCROLL_AMOUNT)
        )
    elif gesture == "back":
        ok, msg = safe_pyautogui("Back", lambda: pyautogui.hotkey("alt", "left"))
    else:
        ok, msg = safe_pyautogui("Next", lambda: pyautogui.hotkey("alt", "right"))

    state.last_action = msg
    state.dynamic_collecting = False
    pose_buffer.clear()
    if ok:
        state.last_dynamic_ts = now
        if gesture in {"back", "next"}:
            state.last_navigation_ts = now
        log_dynamic(state, f"two_fingers -> {gesture}", force=True)
        return msg, gesture, 1
    return msg, gesture, 0


def reset_dynamic_runtime(
    dynamic_history: deque[str],
    pose_buffer: deque[tuple[float, float, float, float, float, str]],
) -> None:
    dynamic_history.clear()
    pose_buffer.clear()


def draw_panel(
    frame: np.ndarray,
    fps: float,
    label_text: str,
    state: ControlState,
    action_history: deque[str],
) -> None:
    lines = [
        f"FPS: {fps:.1f}",
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
            0.75,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )
        y += 32


def build_label_text(
    stable_static_pred: str, stable_dynamic_pred: str, active: bool
) -> str:
    if stable_dynamic_pred != "None":
        return f"{stable_static_pred} -> {stable_dynamic_pred} | active={active}"
    return f"{stable_static_pred} | active={active}"


def update_action_history(
    action_history: deque[str], last_logged_action: str, current_action: str
) -> str:
    if (
        current_action
        and current_action != "None"
        and current_action != last_logged_action
    ):
        action_history.appendleft(current_action)
        return current_action
    return last_logged_action


def change_preview_scale(state: ControlState, delta: float) -> None:
    state.preview_scale = max(
        PREVIEW_SCALE_MIN, min(PREVIEW_SCALE_MAX, state.preview_scale + delta)
    )
    configure_fixed_preview_window(PREVIEW_WINDOW_NAME, state, force=True)
    state.last_action = f"Preview zoom {state.preview_scale:.2f}x"
    print(f"[INFO] Preview zoom = {state.preview_scale:.2f}x")


def main() -> None:
    static_model, static_meta = load_model_predictor(
        STATIC_MODEL_PATH, STATIC_MODEL_META_PATH, name="static"
    )
    print(
        "[INFO] Dynamic model is disabled. Two-fingers actions follow rule-based poses."
    )
    print(
        "[INFO] Tracker uses 2 hands: primary hand controls all actions; second hand only supports pinch click."
    )
    print(
        "[INFO] Primary hand is locked by hand position continuity to avoid cursor jumps when the second hand appears."
    )

    webcam = WebcamCapture(
        WebcamConfig(
            camera_index=0,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            mirror=True,
        )
    )
    tracker = create_tracker()
    state = ControlState()

    static_pred_history: deque[str] = deque(maxlen=SMOOTHING_WINDOW)
    secondary_pred_history: deque[str] = deque(maxlen=5)
    dynamic_history: deque[str] = deque(maxlen=3)
    pose_buffer: deque[tuple[float, float, float, float, float, str]] = deque(
        maxlen=POSE_BUFFER_SIZE
    )
    action_history: deque[str] = deque(maxlen=ACTION_HISTORY_SIZE)
    last_logged_action = "None"

    raw_static_pred, stable_static_pred, static_votes = "None", "None", 0
    raw_dynamic_pred, stable_dynamic_pred, dynamic_votes = "None", "None", 0
    secondary_static_pred, secondary_static_votes = "None", 0
    no_hand_frames = 0
    show_panel = DRAW_DEBUG_PANEL
    frame_index = 0
    prev_time = time.perf_counter()
    start_time = time.perf_counter()

    try:
        webcam.open()
        cv.namedWindow(PREVIEW_WINDOW_NAME, cv.WINDOW_NORMAL)
        preview_w, preview_h = get_preview_size(state)
        cv.resizeWindow(PREVIEW_WINDOW_NAME, preview_w, preview_h)
        configure_fixed_preview_window(PREVIEW_WINDOW_NAME, state, force=True)
        print("[INFO] Webcam opened. Press q to quit.")
        print(
            "[INFO] Keys: q=quit | r=reset | e=toggle active | p=toggle panel | +=zoom in | -=zoom out"
        )

        while True:
            ok, frame = webcam.read()
            if not ok or frame is None:
                continue

            frame_index += 1
            configure_fixed_preview_window(PREVIEW_WINDOW_NAME, state)
            preview_renderable = is_preview_renderable(PREVIEW_WINDOW_NAME)

            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            try:
                annotated_frame, detections = run_tracker(tracker, frame, timestamp_ms)
            except Exception as exc:
                annotated_frame = frame.copy()
                detections = []
                if preview_renderable:
                    cv.putText(
                        annotated_frame,
                        f"Tracker error: {type(exc).__name__}",
                        (20, 35),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv.LINE_AA,
                    )

            handedness = "Unknown"
            landmarks = None
            secondary_landmarks = None
            secondary_static_pred = "None"
            secondary_static_votes = 0

            if detections and len(detections) > 0:
                no_hand_frames = 0

                primary_det, secondary_det = choose_primary_and_secondary(
                    detections, state
                )
                if primary_det is None:
                    detections = []
                    continue

                # Bàn tay chủ: giữ ổn định theo vị trí cũ, không bị đổi sang tay mới xuất hiện.
                landmarks = get_landmarks(primary_det)
                handedness = get_handedness(primary_det)

                # Bàn tay thứ hai: chỉ dùng cho pinch click.
                if secondary_det is not None:
                    secondary_landmarks = get_landmarks(secondary_det)
                else:
                    secondary_pred_history.clear()
                    state.secondary_pinch_touching = False

                if preview_renderable and DRAW_LANDMARKS_ON_PREVIEW:
                    for det_to_draw in detections[:2]:
                        lm_to_draw = get_landmarks(det_to_draw)
                        if lm_to_draw is not None and len(lm_to_draw) == 21:
                            annotated_frame = draw_landmarks(
                                annotated_frame, lm_to_draw
                            )

                # Static prediction cho bàn tay chính.
                try:
                    static_feature = landmarks_to_feature_vector(landmarks).reshape(
                        1, -1
                    )
                    raw_static_pred = predict_one_label(
                        static_model, static_feature, static_meta
                    )
                    static_pred_history.append(raw_static_pred)
                    voted_label, vote_count = majority_vote(
                        static_pred_history, min_votes=SMOOTHING_MIN_VOTES
                    )
                    if voted_label is not None:
                        stable_static_pred = voted_label
                        static_votes = vote_count

                    maybe_toggle_active(state, stable_static_pred)
                    maybe_move_mouse(state, stable_static_pred, landmarks)

                    # Pinch bằng 1 tay vẫn dùng được, nhưng phải thật sự chạm 2 đầu ngón.
                    maybe_left_click(
                        state, stable_static_pred, landmarks, source="primary"
                    )
                except Exception as exc:
                    raw_static_pred = f"ERR: {type(exc).__name__}"

                # Static prediction cho bàn tay thứ hai: chỉ kiểm tra pinch click.
                if secondary_landmarks is not None and len(secondary_landmarks) == 21:
                    try:
                        secondary_feature = landmarks_to_feature_vector(
                            secondary_landmarks
                        ).reshape(1, -1)
                        secondary_raw_pred = predict_one_label(
                            static_model, secondary_feature, static_meta
                        )
                        secondary_pred_history.append(secondary_raw_pred)
                        secondary_voted, secondary_vote_count = majority_vote(
                            secondary_pred_history, min_votes=3
                        )
                        if secondary_voted is not None:
                            secondary_static_pred = secondary_voted
                            secondary_static_votes = secondary_vote_count

                        maybe_left_click(
                            state,
                            secondary_static_pred,
                            secondary_landmarks,
                            source="secondary",
                        )
                    except Exception as exc:
                        secondary_static_pred = f"ERR: {type(exc).__name__}"
                else:
                    state.secondary_pinch_touching = False

                # Dynamic gesture chỉ chạy với bàn tay chính.
                try:
                    raw_dynamic_pred, stable_dynamic_pred, dynamic_votes = (
                        maybe_handle_two_finger_dynamic(
                            state,
                            stable_static_pred,
                            handedness,
                            landmarks,
                            pose_buffer,
                        )
                    )
                    if stable_dynamic_pred != "None":
                        dynamic_history.append(stable_dynamic_pred)
                except Exception as exc:
                    raw_dynamic_pred = f"ERR: {type(exc).__name__}"
                    stable_dynamic_pred = "None"
                    dynamic_votes = 0
                    traceback.print_exc()
            else:
                raw_static_pred, stable_static_pred, static_votes = "No hand", "None", 0
                raw_dynamic_pred, stable_dynamic_pred, dynamic_votes = (
                    "No hand",
                    "None",
                    0,
                )
                secondary_static_pred, secondary_static_votes = "None", 0
                no_hand_frames += 1
                static_pred_history.clear()
                secondary_pred_history.clear()
                reset_dynamic_runtime(dynamic_history, pose_buffer)
                state.dynamic_collecting = False
                state.primary_pinch_touching = False
                state.secondary_pinch_touching = False
                state.primary_center = None
                if no_hand_frames >= NO_HAND_RESET_FRAMES and state.active:
                    state.active = False
                    state.last_action = "Auto LOCK"

            last_logged_action = update_action_history(
                action_history, last_logged_action, state.last_action
            )

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - prev_time, EPS)
            prev_time = current_time
            should_render = frame_index % max(1, RENDER_EVERY_N_FRAMES) == 0

            if should_render and preview_renderable:
                if show_panel:
                    label_text = build_label_text(
                        stable_static_pred, stable_dynamic_pred, state.active
                    )
                    if secondary_static_pred != "None":
                        label_text += f" | second={secondary_static_pred}"
                    draw_panel(annotated_frame, fps, label_text, state, action_history)
                cv.imshow(
                    PREVIEW_WINDOW_NAME, make_preview_frame(annotated_frame, state)
                )

            state.prev_stable_static_pred = stable_static_pred
            state.prev_stable_dynamic_pred = stable_dynamic_pred

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                state = ControlState()
                static_pred_history.clear()
                secondary_pred_history.clear()
                reset_dynamic_runtime(dynamic_history, pose_buffer)
                action_history.clear()
                last_logged_action = "None"
                raw_static_pred, stable_static_pred, static_votes = "None", "None", 0
                raw_dynamic_pred, stable_dynamic_pred, dynamic_votes = "None", "None", 0
                secondary_static_pred, secondary_static_votes = "None", 0
                configure_fixed_preview_window(PREVIEW_WINDOW_NAME, state, force=True)
                print("[INFO] Reset state")
            elif key == ord("e"):
                state.active = not state.active
                if not state.active:
                    state.dynamic_collecting = False
                    reset_dynamic_runtime(dynamic_history, pose_buffer)
                    state.primary_pinch_touching = False
                    state.secondary_pinch_touching = False
                    state.primary_center = None
                state.last_action = f"Manual toggle -> {state.active}"
                print(f"[INFO] Manual active = {state.active}")
            elif key == ord("p"):
                show_panel = not show_panel
                print(f"[INFO] Show panel = {show_panel}")
            elif key in (ord("+"), ord("=")):
                change_preview_scale(state, PREVIEW_SCALE_STEP)
            elif key in (ord("-"), ord("_")):
                change_preview_scale(state, -PREVIEW_SCALE_STEP)
    finally:
        if hasattr(tracker, "close"):
            try:
                tracker.close()
            except Exception:
                pass
        webcam.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
