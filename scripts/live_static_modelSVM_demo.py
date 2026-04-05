from __future__ import annotations

import inspect
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import cv2 as cv
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.capture.webcam import WebcamCapture, WebcamConfig
from src.perception.hand_tracker import HandTracker


# ===== Paths =====
MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "static_svm_rbf.joblib"
HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

# ===== Config =====
SMOOTHING_WINDOW = 7
SMOOTHING_MIN_VOTES = 5
NO_HAND_RESET_FRAMES = 8
EPS = 1e-6


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


def draw_panel(
    frame: np.ndarray,
    fps: float,
    raw_pred: str,
    stable_pred: str,
    handedness: str,
    score_text: str,
    history: deque[str],
    no_hand_frames: int,
) -> None:
    lines = [
        f"FPS: {fps:.1f}",
        f"Handedness: {handedness}",
        f"Raw prediction: {raw_pred}",
        f"Stable prediction: {stable_pred}",
        f"Detection score: {score_text}",
        f"History: {list(history)}",
        f"No-hand frames: {no_hand_frames}",
        "Mode: Preview only",
        "Press q to quit",
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

    pred_history: deque[str] = deque(maxlen=SMOOTHING_WINDOW)
    raw_pred = "None"
    stable_pred = "None"
    handedness = "Unknown"
    score_text = "0.00"
    no_hand_frames = 0

    prev_time = time.perf_counter()
    start_time = time.perf_counter()

    try:
        webcam.open()
        print("[INFO] Webcam đã mở. Demo live SVM đang chạy. Nhấn 'q' để thoát.")

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

            if detections and len(detections) > 0:
                no_hand_frames = 0

                det = detections[0]
                landmarks = get_landmarks(det)

                if landmarks is not None and len(landmarks) == 21:
                    # Nếu tracker không tự vẽ thì mình tự vẽ
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

                except Exception as exc:
                    raw_pred = f"ERR: {type(exc).__name__}"

            else:
                raw_pred = "No hand"
                handedness = "Unknown"
                score_text = "0.00"
                no_hand_frames += 1

                if no_hand_frames >= NO_HAND_RESET_FRAMES:
                    pred_history.clear()
                    stable_pred = "None"

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            draw_panel(
                annotated_frame,
                fps=fps,
                raw_pred=raw_pred,
                stable_pred=stable_pred,
                handedness=handedness,
                score_text=score_text,
                history=pred_history,
                no_hand_frames=no_hand_frames,
            )

            cv.imshow("Live Static SVM Demo - Preview Only", annotated_frame)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

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
