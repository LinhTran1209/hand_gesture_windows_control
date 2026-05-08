from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANDMARK_COUNT = 21
EPS = 1e-6
DYNAMIC_PER_FRAME_FEATURE_DIM = 69

DATASET_CONFIG = {
    "static": {
        "input_csv": PROJECT_ROOT
        / "data"
        / "interim"
        / "landmarks"
        / "static_landmarks_v1.csv",
        "output_dir": PROJECT_ROOT / "data" / "processed" / "static",
        "output_csv": "static_dataset_v1.csv",
    },
    "dynamic": {
        "input_csv": PROJECT_ROOT
        / "data"
        / "interim"
        / "landmarks"
        / "dynamic_landmarks_v1.csv",
        "output_dir": PROJECT_ROOT / "data" / "processed" / "dynamic",
        "output_csv": "dynamic_dataset_v1.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed dataset for static or dynamic data."
    )
    parser.add_argument("--dataset-type", choices=["static", "dynamic"], required=True)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--target-frame-count",
        type=int,
        default=None,
        help="Dynamic: số frame cố định để pad/truncate. Mặc định lấy max số frame detect được trong dataset.",
    )
    parser.add_argument(
        "--dynamic-crop-mode",
        choices=["tail", "center", "head"],
        default="tail",
        help=(
            "Dynamic: cách lấy sequence khi video dài hơn target-frame-count. "
            "tail = lấy cuối video, hợp với video có đoạn giữ 2 ngón tay ở đầu; "
            "center = lấy giữa video; head = lấy đầu video."
        ),
    )
    return parser.parse_args()


def extract_landmarks_from_row(row: pd.Series) -> np.ndarray:
    coords = []
    for i in range(LANDMARK_COUNT):
        coords.append([row[f"x{i}"], row[f"y{i}"], row[f"z{i}"]])
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    wrist = landmarks[0].copy()
    centered = landmarks - wrist
    distances = np.linalg.norm(centered, axis=1)
    scale = float(np.max(distances))
    if scale < EPS:
        scale = 1.0
    return centered / scale


def flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
    return landmarks.reshape(-1)


def build_static_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        landmarks = extract_landmarks_from_row(row)
        if np.isnan(landmarks).any():
            continue
        normalized = normalize_landmarks(landmarks)
        flat = flatten_landmarks(normalized)
        out_row = {
            "dataset_type": "static",
            "video_path": row["video_path"],
            "video_name": row["video_name"],
            "subject_id": row["subject_id"],
            "session_id": row["session_id"],
            "lighting_condition": row.get("lighting_condition"),
            "label": row["label"],
            "frame_idx": row.get("frame_idx"),
            "handedness": row.get("handedness", "Unknown"),
            "score": row.get("score", 0.0),
        }
        for j, value in enumerate(flat):
            out_row[f"f{j}"] = float(value)
        rows.append(out_row)
    return pd.DataFrame(rows).reset_index(drop=True)


def pad_or_truncate_sequence(
    seq: np.ndarray,
    target_len: int,
    crop_mode: str = "tail",
) -> tuple[np.ndarray, int, int, str]:
    """Pad/truncate a dynamic sequence to a fixed length.

    Important for this project: recorded dynamic videos often start with a
    short two_fingers hold before the real motion. The old implementation used
    seq[:target_len], which teaches the model too much preparation/no-motion.
    Defaulting to tail keeps the end of the clip, where the actual swipe/move
    is usually located.
    """
    current_len = len(seq)
    if current_len <= 0:
        raise ValueError("Sequence rỗng, không thể pad/truncate.")

    if current_len == target_len:
        return seq, 0, current_len, "exact"

    if current_len > target_len:
        if crop_mode == "head":
            start = 0
        elif crop_mode == "center":
            start = max(0, (current_len - target_len) // 2)
        elif crop_mode == "tail":
            start = current_len - target_len
        else:
            raise ValueError(f"crop_mode không hợp lệ: {crop_mode}")
        end = start + target_len
        return seq[start:end], start, end, crop_mode

    pad = np.zeros((target_len - current_len, seq.shape[1]), dtype=np.float32)
    padded = np.vstack([seq, pad])
    return padded, 0, current_len, "pad_tail_zeros"


def build_dynamic_dataset(
    df: pd.DataFrame,
    target_frame_count: int | None,
    crop_mode: str = "tail",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["video_path", "video_name", "subject_id", "session_id", "label"]

    grouped = []
    max_detected_frames = 0

    for keys, group in df.groupby(group_cols, dropna=False):
        group = group.sort_values("frame_idx").reset_index(drop=True)
        valid_group = (
            group[group["detected"] == 1].copy()
            if "detected" in group.columns
            else group.copy()
        )
        if len(valid_group) == 0:
            continue
        grouped.append((keys, valid_group))
        max_detected_frames = max(max_detected_frames, len(valid_group))

    if len(grouped) == 0:
        return pd.DataFrame()

    target_len = target_frame_count or max_detected_frames
    print("[INFO] Dynamic target_frame_count:", target_len)
    print("[INFO] Dynamic crop_mode:", crop_mode)

    for keys, valid_group in grouped:
        per_frame_features = []
        wrist_positions = []
        handedness_values = []
        scores = []
        lighting_values = []

        for _, row in valid_group.iterrows():
            landmarks = extract_landmarks_from_row(row)
            if np.isnan(landmarks).any():
                continue

            normalized = normalize_landmarks(landmarks)
            flat = flatten_landmarks(normalized)  # 63
            per_frame_features.append(flat)
            wrist_positions.append(landmarks[0].copy())
            handedness_values.append(str(row.get("handedness", "Unknown")))
            scores.append(float(row.get("score", 0.0)))
            lighting_values.append(row.get("lighting_condition"))

        if len(per_frame_features) == 0:
            continue

        features = np.vstack(per_frame_features)  # (T,63)
        wrists = np.vstack(wrist_positions)  # (T,3)

        wrist_origin = wrists[0].copy()
        wrist_rel = wrists - wrist_origin
        wrist_scale = float(np.max(np.linalg.norm(wrist_rel, axis=1)))
        if wrist_scale < EPS:
            wrist_scale = 1.0
        wrist_rel = wrist_rel / wrist_scale

        wrist_delta = np.zeros_like(wrist_rel)
        if len(wrist_rel) > 1:
            wrist_delta[1:] = wrist_rel[1:] - wrist_rel[:-1]

        seq_full = np.concatenate([features, wrist_rel, wrist_delta], axis=1)  # (T,69)
        seq, crop_start, crop_end, crop_applied = pad_or_truncate_sequence(
            seq_full, target_len, crop_mode=crop_mode
        )

        if seq.shape[1] != DYNAMIC_PER_FRAME_FEATURE_DIM:
            raise RuntimeError(
                f"Dynamic per-frame feature dim sai: {seq.shape[1]} != {DYNAMIC_PER_FRAME_FEATURE_DIM}"
            )

        video_path, video_name, subject_id, session_id, label = keys
        handedness = (
            pd.Series(handedness_values).mode().iloc[0]
            if handedness_values
            else "Unknown"
        )
        lighting_condition = (
            pd.Series(lighting_values).mode().iloc[0] if lighting_values else None
        )
        mean_score = float(np.mean(scores)) if scores else 0.0

        out_row = {
            "dataset_type": "dynamic",
            "video_path": video_path,
            "video_name": video_name,
            "subject_id": subject_id,
            "session_id": session_id,
            "lighting_condition": lighting_condition,
            "label": label,
            "target_frame_count": target_len,
            "original_detected_frames": len(seq_full),
            "crop_mode": crop_mode,
            "crop_applied": crop_applied,
            "crop_start": crop_start,
            "crop_end": crop_end,
            "handedness": handedness,
            "score": mean_score,
        }

        flat = seq.reshape(-1)
        for j, value in enumerate(flat):
            out_row[f"f{j}"] = float(value)

        rows.append(out_row)

    return pd.DataFrame(rows).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset_type]
    input_csv = args.input_csv or cfg["input_csv"]
    output_dir = args.output_csv.parent if args.output_csv else cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or output_dir / cfg["output_csv"]

    if not input_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy file input: {input_csv}")

    df = pd.read_csv(input_csv)
    if len(df) == 0:
        raise RuntimeError("Input CSV rỗng.")

    if args.dataset_type == "static":
        if "detected" in df.columns:
            df = df[df["detected"] == 1].copy()
        out_df = build_static_dataset(df)
    else:
        out_df = build_dynamic_dataset(
            df,
            target_frame_count=args.target_frame_count,
            crop_mode=args.dynamic_crop_mode,
        )

    if len(out_df) == 0:
        raise RuntimeError("Không tạo được sample nào cho processed dataset.")

    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved processed dataset to: {output_csv}")
    print(f"[INFO] Shape: {out_df.shape}")

    if args.dataset_type == "dynamic":
        print(
            f"[INFO] Dynamic feature count: {len([c for c in out_df.columns if c.startswith('f')])}"
        )


if __name__ == "__main__":
    main()
