from __future__ import annotations
from dataclasses import dataclass

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print(f"[INFO] PROJECT_ROOT set to: {PROJECT_ROOT}")

import inspect
import timeq
from collections import Counter, deque
from typing import Any

import cv2 as cv
import joblib
import numpy as np
import pyautogui

from src.capture.webcam import WebcamCapture, WebcamConfig
from src.perception.hand_tracker import HandTracker


# =========================
# PATHS
# =========================
MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "static_svm_rbf.joblib"
HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

# =========================
# CONFIG
# =========================
SMOOTHING_WINDOW = 7
SMOOTHING_MIN_VOTES = 5
NO_HAND_RESET_FRAMES = 8
EPS = 1e-6

MOUSE_SMOOTHING = 0.25
CLICK_COOLDOWN_SEC = 0.60
MODE_SWITCH_COOLDOWN_SEC = 0.80
SWIPE_COOLDOWN_SEC = 0.90
SWIPE_WINDOW = 8
SWIPE_MIN_DX = 0.18
SWIPE_MAX_DY = 0.10
MOVE_ONLY_WHEN_ACTIVE = True
DRAW_MOUSE_PREVIEW = True
MOVE_DEADZONE = 0.01
POINTER_MARGIN_X = 0.18
POINTER_MARGIN_Y = 0.22

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


# =========================
# STATE
# =========================
@dataclass
class ControlState:
    active: bool = False  # open_palm -> active, fist -> lock
    last_click_ts: float = 0.0
    last_swipe_ts: float = 0.0
    last_mode_ts: float = 0.0
    last_action: str = "None"
    mouse_x: float | None = None
    mouse_y: float | None = None


# =========================
# UTILS
# =========================
def normalize_landmarks(landmarks_xyz: np.ndarray) -> np.ndarray:
    wrist = landmarks_xyz[0].copy()
    centered = landmarks_xyz - wrist

    distances = np.linalg.norm(centered, axis=1)
    scale = float(np.max(distances))
    if scale < EPS:
        scale = 1.0

    normalized = centered / scale
    return normalized


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


def get_score(det: Any) -> float:
    if det is None:
        return 0.0

    value = getattr(det, "score", None)
    if value is None and isinstance(det, dict):
        value = det.get("score")

    if value is None:
        return 0.0

    if isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]

    try:
        return float(value)
    except Exception:
        return 0.0


def detection_to_feature_vector(det: Any) -> np.ndarray:
    landmarks = get_landmarks(det)
    if landmarks is None or len(landmarks) != 21:
        raise ValueError("Detection không hợp lệ hoặc không đủ 21 landmarks.")

    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    normalized = normalize_landmarks(coords)
    feature = normalized.reshape(-1)

    if feature.shape[0] != 63:
        raise ValueError(f"Feature dimension sai: {feature.shape[0]}")

    return feature


def majority_vote(history: deque[str]) -> tuple[str | None, int]:
    if not history:
        return None, 0

    counter = Counter(history)
    label, count = counter.most_common(1)[0]

    if count >= SMOOTHING_MIN_VOTES:
        return label, count

    return None, count


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
        px = int(float(x) * w)
        py = int(float(y) * h)
        points.append((px, py))

    hand_connections = [
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

    for a, b in hand_connections:
        if a < len(points) and b < len(points):
            cv.line(canvas, points[a], points[b], (0, 255, 0), 2)

    for idx, (px, py) in enumerate(points):
        radius = 5 if idx in {0, 4, 8, 12, 16, 20} else 4
        cv.circle(canvas, (px, py), radius, (0, 0, 255), -1)

    return canvas


def create_tracker() -> Any:
    init_sig = inspect.signature(HandTracker.__init__)
    supported = init_sig.parameters

    candidate_kwargs = {
        "model_path": HAND_MODEL_PATH,
        "max_num_hands": 1,
        "min_detection_confidence": 0.5,
        "min_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    }

    kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported}

    print("[INFO] HandTracker.__init__ signature:", init_sig)
    print("[INFO] Tracker kwargs dùng thực tế:", kwargs)

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


def run_tracker(
    tracker: Any, frame: np.ndarray, timestamp_ms: int
) -> tuple[np.ndarray, list[Any]]:
    process_sig = inspect.signature(tracker.process)
    printed = getattr(run_tracker, "_printed", False)
    if not printed:
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


def normalized_point_to_screen(
    nx: float, ny: float, screen_w: int, screen_h: int
) -> tuple[int, int]:
    # mirror=True nên x trong ảnh đã phản chiếu theo cảm nhận người dùng. Ta map trực tiếp.
    x = (nx - POINTER_MARGIN_X) / max(1e-6, 1.0 - 2 * POINTER_MARGIN_X)
    y = (ny - POINTER_MARGIN_Y) / max(1e-6, 1.0 - 2 * POINTER_MARGIN_Y)
    x = clamp01(x)
    y = clamp01(y)
    return int(x * (screen_w - 1)), int(y * (screen_h - 1))


def smooth_mouse_target(state: ControlState, tx: int, ty: int) -> tuple[int, int]:
    if state.mouse_x is None or state.mouse_y is None:
        state.mouse_x = float(tx)
        state.mouse_y = float(ty)
    else:
        state.mouse_x = (1.0 - MOUSE_SMOOTHING) * state.mouse_x + MOUSE_SMOOTHING * tx
        state.mouse_y = (1.0 - MOUSE_SMOOTHING) * state.mouse_y + MOUSE_SMOOTHING * ty
    return int(state.mouse_x), int(state.mouse_y)


def maybe_move_mouse(state: ControlState, stable_pred: str, landmarks: Any) -> None:
    if MOVE_ONLY_WHEN_ACTIVE and not state.active:
        return
    if stable_pred != "point":
        return

    pt = landmark_xy(landmarks, 8)  # index fingertip
    if pt is None:
        return

    screen_w, screen_h = pyautogui.size()
    target_x, target_y = normalized_point_to_screen(pt[0], pt[1], screen_w, screen_h)
    mx, my = smooth_mouse_target(state, target_x, target_y)

    current_x, current_y = pyautogui.position()
    dx = abs(mx - current_x) / max(screen_w, 1)
    dy = abs(my - current_y) / max(screen_h, 1)
    if dx < MOVE_DEADZONE and dy < MOVE_DEADZONE:
        return

    pyautogui.moveTo(mx, my)
    state.last_action = f"Move mouse -> ({mx}, {my})"


def maybe_left_click(state: ControlState, stable_pred: str) -> None:
    if not state.active:
        return
    if stable_pred != "pinch":
        return

    now = time.perf_counter()
    if now - state.last_click_ts < CLICK_COOLDOWN_SEC:
        return

    pyautogui.click(button="left")
    state.last_click_ts = now
    state.last_action = "Left click"


def maybe_toggle_active(state: ControlState, stable_pred: str) -> None:
    now = time.perf_counter()
    if now - state.last_mode_ts < MODE_SWITCH_COOLDOWN_SEC:
        return

    if stable_pred == "open_palm" and not state.active:
        state.active = True
        state.last_mode_ts = now
        state.last_action = "ACTIVE"
    elif stable_pred == "fist" and state.active:
        state.active = False
        state.last_mode_ts = now
        state.last_action = "LOCK"


def detect_swipe(trajectory: deque[tuple[float, float]]) -> str | None:
    if len(trajectory) < SWIPE_WINDOW:
        return None

    start_x, start_y = trajectory[0]
    end_x, end_y = trajectory[-1]
    dx = end_x - start_x
    dy = end_y - start_y

    if abs(dy) > SWIPE_MAX_DY:
        return None

    if dx <= -SWIPE_MIN_DX:
        return "swipe_left"
    if dx >= SWIPE_MIN_DX:
        return "swipe_right"
    return None


def maybe_handle_swipe(
    state: ControlState,
    stable_pred: str,
    landmarks: Any,
    trajectory: deque[tuple[float, float]],
) -> None:
    if not state.active:
        return
    if stable_pred not in {"no_gesture", "point", "open_palm"}:
        # vẫn cho phép tích lũy trajectory khi tay ở trạng thái không quá co cụm
        trajectory.clear()
        return

    pt = landmark_xy(landmarks, 0)  # wrist cho ổn định hơn fingertip
    if pt is None:
        trajectory.clear()
        return

    trajectory.append(pt)
    swipe = detect_swipe(trajectory)
    if swipe is None:
        return

    now = time.perf_counter()
    if now - state.last_swipe_ts < SWIPE_COOLDOWN_SEC:
        return

    if swipe == "swipe_left":
        pyautogui.hotkey("alt", "left")
        state.last_action = "Swipe left -> Back"
    elif swipe == "swipe_right":
        pyautogui.hotkey("alt", "right")
        state.last_action = "Swipe right -> Next"

    state.last_swipe_ts = now
    trajectory.clear()


def draw_mouse_preview(frame: np.ndarray, state: ControlState, landmarks: Any) -> None:
    if not DRAW_MOUSE_PREVIEW:
        return
    if landmarks is None:
        return

    pt = landmark_xy(landmarks, 8)
    if pt is None:
        return

    h, w = frame.shape[:2]
    px = int(pt[0] * w)
    py = int(pt[1] * h)
    cv.circle(frame, (px, py), 12, (255, 0, 255), 2)
    cv.putText(
        frame,
        f"Mouse preview: ({pt[0]:.2f}, {pt[1]:.2f})",
        (20, h - 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 255),
        2,
        cv.LINE_AA,
    )


def draw_panel(
    frame: np.ndarray,
    fps: float,
    raw_pred: str,
    stable_pred: str,
    handedness: str,
    score_text: str,
    history: deque[str],
    no_hand_frames: int,
    state: ControlState,
) -> None:
    lines = [
        f"FPS: {fps:.1f}",
        "Control mode: mouse + browser",
        f"Control active: {state.active}",
        "Mapping: open_palm=ACTIVE | fist=LOCK | point=MOVE | pinch=LEFT CLICK",
        "         swipe_left=BACK | swipe_right=NEXT | no_gesture=IDLE",
        f"Handedness: {handedness}",
        f"Raw prediction: {raw_pred}",
        f"Stable prediction: {stable_pred}",
        f"Detection score: {score_text}",
        f"Last action: {state.last_action}",
        f"History: {list(history)}",
        f"No-hand frames: {no_hand_frames}",
        "Keys: q=quit | r=reset state | e=toggle active manually",
    ]

    x, y = 20, 35
    for line in lines:
        cv.putText(
            frame,
            line,
            (x, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )
        y += 30


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model SVM: {MODEL_PATH}")

    if not HAND_MODEL_PATH.exists():
        print(f"[WARN] Không tìm thấy hand model tại: {HAND_MODEL_PATH}")
        print(
            "[WARN] Sẽ vẫn thử tạo HandTracker theo API hiện tại nếu class không cần model_path."
        )

    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Loaded SVM model: {MODEL_PATH}")

    webcam = WebcamCapture(
        WebcamConfig(
            camera_index=0,
            width=1280,
            height=720,
            fps=30,
            mirror=True,
        )
    )

    tracker = create_tracker()
    state = ControlState()

    pred_history: deque[str] = deque(maxlen=SMOOTHING_WINDOW)
    swipe_trajectory: deque[tuple[float, float]] = deque(maxlen=SWIPE_WINDOW)

    raw_pred = "None"
    stable_pred = "None"
    stable_votes = 0
    handedness = "Unknown"
    score_text = "0.00"
    no_hand_frames = 0

    prev_time = time.perf_counter()
    start_time = time.perf_counter()

    try:
        webcam.open()
        print("[INFO] Webcam đã mở. Mouse-control demo đang chạy. Nhấn 'q' để thoát.")

        while True:
            ok, frame = webcam.read()
            if not ok or frame is None:
                print("[WARN] Không đọc được frame từ webcam.")
                continue

            timestamp_ms = int((time.perf_counter() - start_time) * 1000)

            try:
                annotated_frame, detections = run_tracker(tracker, frame, timestamp_ms)
            except Exception as exc:
                annotated_frame = frame.copy()
                detections = []
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

            landmarks = None

            if detections and len(detections) > 0:
                no_hand_frames = 0

                det = detections[0]
                landmarks = get_landmarks(det)

                if landmarks is not None and len(landmarks) == 21:
                    annotated_frame = draw_landmarks(annotated_frame, landmarks)

                handedness = get_handedness(det)
                score_text = f"{get_score(det):.2f}"

                try:
                    feature = detection_to_feature_vector(det).reshape(1, -1)
                    pred = model.predict(feature)[0]
                    raw_pred = str(pred)

                    pred_history.append(raw_pred)
                    voted_label, vote_count = majority_vote(pred_history)
                    if voted_label is not None:
                        stable_pred = voted_label
                        stable_votes = vote_count

                    maybe_toggle_active(state, stable_pred)
                    maybe_move_mouse(state, stable_pred, landmarks)
                    maybe_left_click(state, stable_pred)
                    maybe_handle_swipe(state, stable_pred, landmarks, swipe_trajectory)

                except Exception as exc:
                    raw_pred = f"ERR: {type(exc).__name__}"

                draw_mouse_preview(annotated_frame, state, landmarks)

            else:
                raw_pred = "No hand"
                handedness = "Unknown"
                score_text = "0.00"
                no_hand_frames += 1
                swipe_trajectory.clear()

                if no_hand_frames >= NO_HAND_RESET_FRAMES:
                    pred_history.clear()
                    stable_pred = "None"
                    stable_votes = 0

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            draw_panel(
                annotated_frame,
                fps=fps,
                raw_pred=raw_pred,
                stable_pred=f"{stable_pred} (votes={stable_votes})",
                handedness=handedness,
                score_text=score_text,
                history=pred_history,
                no_hand_frames=no_hand_frames,
                state=state,
            )

            cv.imshow("Live Mouse Control Demo", annotated_frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                state = ControlState()
                pred_history.clear()
                swipe_trajectory.clear()
                stable_pred = "None"
                stable_votes = 0
                print("[INFO] Reset state")
            elif key == ord("e"):
                state.active = not state.active
                state.last_action = f"Manual toggle -> {state.active}"
                print(f"[INFO] Manual active = {state.active}")

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
