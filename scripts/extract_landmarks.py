from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.hand_tracker import HandTracker


# =========================
# CONFIG
# =========================
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "landmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_LABELS = {
    "open_palm",
    "point",
    "pinch",
    "fist",
    "no_gesture",
    "two_fingers",
    # "thumbs_up",
}

DYNAMIC_LABELS = {
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "no_motion",
}

DATASET_CONFIG = {
    "static": {
        "manifest_path": PROJECT_ROOT
        / "data"
        / "raw"
        / "self_collected"
        / "metadata"
        / "manifest_static_v1.csv",
        "output_csv": OUTPUT_DIR / "static_landmarks_v1.csv",
        "quality_csv": OUTPUT_DIR / "static_landmarks_quality_v1.csv",
        "labels": STATIC_LABELS,
        "sampling_mode": "sparse",
        "frames_per_video": 10,
    },
    "dynamic": {
        "manifest_path": PROJECT_ROOT
        / "data"
        / "raw"
        / "self_collected"
        / "metadata"
        / "manifest_dynamic_v1.csv",
        "output_csv": OUTPUT_DIR / "dynamic_landmarks_v1.csv",
        "quality_csv": OUTPUT_DIR / "dynamic_landmarks_quality_v1.csv",
        "labels": DYNAMIC_LABELS,
        "sampling_mode": "sequence",
        "frames_per_video": None,
    },
}


# =========================
# UTILS
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hand landmarks for static or dynamic gesture dataset."
    )
    parser.add_argument(
        "--dataset-type",
        choices=["static", "dynamic"],
        required=True,
        help="Chọn loại dataset cần extract: static hoặc dynamic.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Override đường dẫn manifest nếu cần.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Override đường dẫn file output landmarks.",
    )
    parser.add_argument(
        "--quality-csv",
        type=Path,
        default=None,
        help="Override đường dẫn file output quality.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=None,
        help="Số frame sample cho static. Bỏ qua với dynamic.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Giới hạn số frame tối đa mỗi video ở dynamic để debug nhanh.",
    )
    return parser.parse_args()


def choose_frame_indices_static(
    frame_count: int, frames_per_video: int = 10
) -> list[int]:
    if frame_count <= 0:
        return []

    start_idx = int(frame_count * 0.2)
    end_idx = int(frame_count * 0.8)

    if end_idx <= start_idx:
        start_idx = 0
        end_idx = frame_count

    usable = end_idx - start_idx
    if usable <= 0:
        return []

    if usable <= frames_per_video:
        return list(range(start_idx, end_idx))

    indices = np.linspace(start_idx, end_idx - 1, num=frames_per_video, dtype=int)
    return indices.tolist()


def choose_frame_indices_dynamic(
    frame_count: int,
    max_frames: int | None = None,
) -> list[int]:
    if frame_count <= 0:
        return []

    indices = list(range(frame_count))
    if max_frames is not None and max_frames > 0:
        indices = indices[:max_frames]
    return indices


def build_frame_indices(
    dataset_type: str,
    frame_count: int,
    frames_per_video: int | None = None,
    max_frames: int | None = None,
) -> list[int]:
    if dataset_type == "static":
        return choose_frame_indices_static(
            frame_count=frame_count,
            frames_per_video=frames_per_video or 10,
        )
    if dataset_type == "dynamic":
        return choose_frame_indices_dynamic(
            frame_count=frame_count,
            max_frames=max_frames,
        )
    raise ValueError(f"dataset_type không hợp lệ: {dataset_type}")


def empty_row(
    video_path: Path,
    subject_id: Any,
    session_id: Any,
    lighting_condition: Any,
    label: str,
    frame_idx: int,
    dataset_type: str,
) -> dict[str, Any]:
    row = {
        "dataset_type": dataset_type,
        "video_path": str(video_path),
        "video_name": video_path.name,
        "subject_id": subject_id,
        "session_id": session_id,
        "lighting_condition": lighting_condition,
        "label": label,
        "frame_idx": frame_idx,
        "detected": 0,
        "handedness": "Unknown",
        "score": 0.0,
    }
    for i in range(21):
        row[f"x{i}"] = np.nan
        row[f"y{i}"] = np.nan
        row[f"z{i}"] = np.nan
    return row


def create_tracker() -> Any:
    init_sig = inspect.signature(HandTracker.__init__)
    supported = init_sig.parameters

    candidate_kwargs = {
        "model_path": DEFAULT_MODEL_PATH,
        "max_num_hands": 1,
        "min_detection_confidence": 0.5,
        "min_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    }

    kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported}

    print("[INFO] HandTracker.__init__ signature:", init_sig)
    print("[INFO] Tracker kwargs dùng thực tế:", kwargs)

    return HandTracker(**kwargs)


def unpack_process_result(result: Any) -> list[Any]:
    if result is None:
        return []

    if isinstance(result, tuple):
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


def resolve_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = DATASET_CONFIG[args.dataset_type].copy()
    if args.manifest_path is not None:
        cfg["manifest_path"] = args.manifest_path
    if args.output_csv is not None:
        cfg["output_csv"] = args.output_csv
    if args.quality_csv is not None:
        cfg["quality_csv"] = args.quality_csv
    if args.frames_per_video is not None:
        cfg["frames_per_video"] = args.frames_per_video
    cfg["max_frames"] = args.max_frames
    return cfg


# =========================
# MAIN
# =========================
def main() -> None:
    args = parse_args()
    cfg = resolve_runtime_config(args)

    dataset_type = args.dataset_type
    manifest_path = Path(cfg["manifest_path"])
    output_csv = Path(cfg["output_csv"])
    quality_csv = Path(cfg["quality_csv"])
    labels = set(cfg["labels"])
    frames_per_video = cfg.get("frames_per_video")
    max_frames = cfg.get("max_frames")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    quality_csv.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] PROJECT_ROOT  :", PROJECT_ROOT)
    print("[INFO] DATASET_TYPE  :", dataset_type)
    print("[INFO] MANIFEST_PATH :", manifest_path)
    print("[INFO] OUTPUT_CSV    :", output_csv)
    print("[INFO] QUALITY_CSV   :", quality_csv)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Không tìm thấy manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)

    required_cols = {
        "video_path",
        "subject_id",
        "session_id",
        "lighting_condition",
        "label",
        "frame_count",
        "is_readable",
    }
    missing = required_cols - set(manifest.columns)
    if missing:
        raise RuntimeError(f"Manifest thiếu cột: {sorted(missing)}")

    manifest = manifest[manifest["label"].isin(labels)].copy()
    manifest = manifest[manifest["is_readable"] == 1].reset_index(drop=True)

    if len(manifest) == 0:
        raise RuntimeError(
            f"Không có video {dataset_type} hợp lệ trong manifest sau khi filter labels."
        )

    all_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for idx, row in manifest.iterrows():
        video_path = Path(row["video_path"])
        subject_id = row["subject_id"]
        session_id = row["session_id"]
        lighting_condition = row["lighting_condition"]
        label = str(row["label"])
        frame_count = int(row["frame_count"])

        print(f"\n[{idx + 1}/{len(manifest)}] {video_path.name}")

        if not video_path.exists():
            print(f"[SKIP] Không tìm thấy video: {video_path}")
            continue

        cap = cv.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[SKIP] Không mở được video: {video_path}")
            continue

        fps = cap.get(cv.CAP_PROP_FPS) or 30.0
        selected_indices = build_frame_indices(
            dataset_type=dataset_type,
            frame_count=frame_count,
            frames_per_video=frames_per_video,
            max_frames=max_frames,
        )
        tracker = create_tracker()

        detected_count = 0

        try:
            for frame_idx in selected_indices:
                cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()

                if not ok or frame is None:
                    all_rows.append(
                        empty_row(
                            video_path,
                            subject_id,
                            session_id,
                            lighting_condition,
                            label,
                            frame_idx,
                            dataset_type,
                        )
                    )
                    continue

                timestamp_ms = int((frame_idx / max(fps, 1e-6)) * 1000)

                try:
                    detections = run_tracker(tracker, frame, timestamp_ms)
                except Exception as e:
                    print(
                        f"[WARN] {video_path.name} frame {frame_idx}: process lỗi: {e}"
                    )
                    all_rows.append(
                        empty_row(
                            video_path,
                            subject_id,
                            session_id,
                            lighting_condition,
                            label,
                            frame_idx,
                            dataset_type,
                        )
                    )
                    continue

                if not detections:
                    all_rows.append(
                        empty_row(
                            video_path,
                            subject_id,
                            session_id,
                            lighting_condition,
                            label,
                            frame_idx,
                            dataset_type,
                        )
                    )
                    continue

                det = detections[0]
                landmarks = get_landmarks(det)

                if landmarks is None or len(landmarks) != 21:
                    all_rows.append(
                        empty_row(
                            video_path,
                            subject_id,
                            session_id,
                            lighting_condition,
                            label,
                            frame_idx,
                            dataset_type,
                        )
                    )
                    continue

                detected_count += 1

                out_row = {
                    "dataset_type": dataset_type,
                    "video_path": str(video_path),
                    "video_name": video_path.name,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "lighting_condition": lighting_condition,
                    "label": label,
                    "frame_idx": frame_idx,
                    "detected": 1,
                    "handedness": get_handedness(det),
                    "score": get_score(det),
                }

                for i, lm in enumerate(landmarks):
                    out_row[f"x{i}"] = float(lm.x)
                    out_row[f"y{i}"] = float(lm.y)
                    out_row[f"z{i}"] = float(lm.z)

                all_rows.append(out_row)

        finally:
            if hasattr(tracker, "close"):
                try:
                    tracker.close()
                except Exception:
                    pass
            cap.release()

        quality_rows.append(
            {
                "dataset_type": dataset_type,
                "video_path": str(video_path),
                "video_name": video_path.name,
                "subject_id": subject_id,
                "session_id": session_id,
                "lighting_condition": lighting_condition,
                "label": label,
                "selected_frames": len(selected_indices),
                "detected_frames": detected_count,
                "detection_rate": round(
                    detected_count / max(len(selected_indices), 1), 4
                ),
            }
        )

        print(
            f"[INFO] selected={len(selected_indices)} | "
            f"detected={detected_count} | "
            f"rate={detected_count / max(len(selected_indices), 1):.4f}"
        )

    if len(all_rows) == 0:
        raise RuntimeError("Không tạo được dòng dữ liệu nào từ landmarks.")

    df = pd.DataFrame(all_rows)
    q = pd.DataFrame(quality_rows)

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    q.to_csv(quality_csv, index=False, encoding="utf-8-sig")

    print(f"\n[DONE] Saved: {output_csv}")
    print(f"[DONE] Saved: {quality_csv}")
    print(f"[INFO] Frame rows: {len(df)}")

    print("\n[COUNTS BY LABEL / DETECTED]")
    print(df.groupby(["label", "detected"]).size())

    print("\n[MEAN DETECTION RATE BY LABEL]")
    print(q.groupby("label")["detection_rate"].mean().round(4))


if __name__ == "__main__":
    main()
