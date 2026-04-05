from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[0]
# Cho phép copy file này vào thư mục scripts/ hoặc chạy từ project root.
for candidate in [
    Path.cwd(),
    Path(__file__).resolve().parents[1],
    Path(__file__).resolve().parents[0],
]:
    if (candidate / "src").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        PROJECT_ROOT = candidate
        break

from src.perception.hand_tracker import HandTracker


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
print(f"[INFO] Dự kiến HandTracker model path: {DEFAULT_MODEL_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test HandTracker trên 1 video, hiển thị preview và thống kê detect."
    )
    parser.add_argument(
        "--video", required=True, help="Đường dẫn tới file video cần test"
    )
    parser.add_argument(
        "--save",
        default="",
        help="Đường dẫn video output có vẽ landmarks. Bỏ trống nếu không lưu.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Chỉ detect mỗi N frame để test nhanh hơn. Mặc định: 1",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Giới hạn số frame cần test. 0 = đọc hết video.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Không mở cửa sổ preview.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Ngưỡng confidence mặc định khi tạo tracker.",
    )
    return parser.parse_args()


def create_tracker(conf: float = 0.5) -> Any:
    init_sig = inspect.signature(HandTracker.__init__)
    supported = init_sig.parameters

    candidate_kwargs = {
        "model_path": DEFAULT_MODEL_PATH,
        "max_num_hands": 1,
        "min_detection_confidence": conf,
        "min_presence_confidence": conf,
        "min_tracking_confidence": conf,
    }

    kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported}

    print("[INFO] HandTracker.__init__ signature:", init_sig)
    print("[INFO] Tracker kwargs dùng thực tế:", kwargs)

    return HandTracker(**kwargs)


def unpack_process_result(result: Any) -> list[Any]:
    if result is None:
        return []

    if isinstance(result, tuple):
        # Ưu tiên phần tử cuối cùng vì script cũ dùng (_, detections)
        maybe = result[-1]
        if maybe is None:
            return []
        if isinstance(maybe, list):
            return maybe
        return [maybe]

    if isinstance(result, list):
        return result

    return [result]


def run_tracker(tracker: Any, frame: np.ndarray, timestamp_ms: int) -> list[Any]:
    process_sig = inspect.signature(tracker.process)
    print_once = getattr(run_tracker, "_printed", False)
    if not print_once:
        print("[INFO] HandTracker.process signature:", process_sig)
        run_tracker._printed = True

    try:
        params = process_sig.parameters
        if "timestamp_ms" in params:
            result = tracker.process(frame, timestamp_ms=timestamp_ms)
        elif len(params) >= 2:
            # Trường hợp bound method vẫn hiện (frame, timestamp_ms)
            result = tracker.process(frame, timestamp_ms)
        else:
            result = tracker.process(frame)
    except TypeError:
        # fallback cứng nếu chữ ký introspection không đúng như kỳ vọng
        try:
            result = tracker.process(frame, timestamp_ms)
        except TypeError:
            result = tracker.process(frame)

    return unpack_process_result(result)


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
    return str(value)


def get_score(det: Any) -> float:
    if det is None:
        return 0.0
    value = getattr(det, "score", None)
    if value is None and isinstance(det, dict):
        value = det.get("score")
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


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

    HAND_CONNECTIONS = [
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

    for a, b in HAND_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv.line(canvas, points[a], points[b], (0, 255, 0), 2)

    for idx, (px, py) in enumerate(points):
        radius = 5 if idx in {0, 4, 8, 12, 16, 20} else 4
        cv.circle(canvas, (px, py), radius, (0, 0, 255), -1)

    return canvas


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)

    writer = None
    if args.save:
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(args.save, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Không tạo được video output: {args.save}")

    tracker = create_tracker(conf=args.conf)

    processed = 0
    sampled = 0
    detected = 0
    first_detect_frame = None
    start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_idx = int(cap.get(cv.CAP_PROP_POS_FRAMES)) - 1
            processed += 1

            view = frame.copy()
            do_detect = frame_idx % max(args.sample_every, 1) == 0
            dets = []

            if do_detect:
                sampled += 1
                timestamp_ms = int((frame_idx / max(fps, 1e-6)) * 1000)
                try:
                    dets = run_tracker(tracker, frame, timestamp_ms)
                except Exception as e:
                    print(f"[WARN] frame={frame_idx}: process lỗi: {e}")
                    dets = []

                if dets:
                    det0 = dets[0]
                    landmarks = get_landmarks(det0)
                    if landmarks and len(landmarks) == 21:
                        detected += 1
                        if first_detect_frame is None:
                            first_detect_frame = frame_idx
                        view = draw_landmarks(view, landmarks)
                        handedness = get_handedness(det0)
                        score = get_score(det0)
                        cv.putText(
                            view,
                            f"DETECTED | {handedness} | score={score:.3f}",
                            (20, 35),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                            cv.LINE_AA,
                        )
                    else:
                        cv.putText(
                            view,
                            "No valid 21-point landmarks",
                            (20, 35),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                            cv.LINE_AA,
                        )
                else:
                    cv.putText(
                        view,
                        "NO HAND DETECTED",
                        (20, 35),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv.LINE_AA,
                    )
            else:
                cv.putText(
                    view,
                    "Skipped detection on this frame",
                    (20, 35),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv.LINE_AA,
                )

            cv.putText(
                view,
                f"frame={frame_idx}/{max(total_frames-1, 0)} | sampled={sampled} | detected={detected}",
                (20, height - 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )

            if writer is not None:
                writer.write(view)

            if not args.no_show:
                cv.imshow("HandTracker Test", view)
                key = cv.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if args.max_frames > 0 and processed >= args.max_frames:
                break

    finally:
        if hasattr(tracker, "close"):
            try:
                tracker.close()
            except Exception:
                pass
        cap.release()
        if writer is not None:
            writer.release()
        cv.destroyAllWindows()

    elapsed = time.time() - start
    detection_rate = detected / max(sampled, 1)

    print("\n===== SUMMARY =====")
    print(f"Video           : {video_path}")
    print(f"Total frames    : {total_frames}")
    print(f"Processed       : {processed}")
    print(f"Sampled         : {sampled}")
    print(f"Detected        : {detected}")
    print(f"Detection rate  : {detection_rate:.4f}")
    print(f"First detection : {first_detect_frame}")
    print(f"Elapsed         : {elapsed:.2f}s")
    if args.save:
        print(f"Saved output    : {args.save}")


if __name__ == "__main__":
    main()
