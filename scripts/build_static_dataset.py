from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "interim" / "landmarks" / "static_landmarks_v1.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "static"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "static_dataset_v1.csv"


LANDMARK_COUNT = 21
EPS = 1e-6


def extract_landmarks_from_row(row: pd.Series) -> np.ndarray:
    coords = []
    for i in range(LANDMARK_COUNT):
        coords.append([row[f"x{i}"], row[f"y{i}"], row[f"z{i}"]])
    return np.array(coords, dtype=np.float32)  # shape: (21, 3)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa:
    1. Lấy wrist (landmark 0) làm gốc tọa độ
    2. Scale theo khoảng cách lớn nhất từ wrist tới các điểm còn lại
    """
    wrist = landmarks[0].copy()
    centered = landmarks - wrist

    distances = np.linalg.norm(centered, axis=1)
    scale = float(np.max(distances))

    if scale < EPS:
        scale = 1.0

    normalized = centered / scale
    return normalized


def flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
    return landmarks.reshape(-1)  # 21 * 3 = 63


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy file input: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # Chỉ giữ frame detect được tay
    df = df[df["detected"] == 1].copy()

    if len(df) == 0:
        raise RuntimeError("Không có frame nào detect được tay để build dataset.")

    rows = []

    for idx, row in df.iterrows():
        landmarks = extract_landmarks_from_row(row)

        if np.isnan(landmarks).any():
            continue

        normalized = normalize_landmarks(landmarks)
        flat = flatten_landmarks(normalized)

        out_row = {
            "video_path": row["video_path"],
            "video_name": row["video_name"],
            "subject_id": row["subject_id"],
            "session_id": row["session_id"],
            "lighting_condition": row["lighting_condition"],
            "label": row["label"],
            "frame_idx": row["frame_idx"],
            "handedness": row["handedness"],
            "score": row["score"],
        }

        for j, value in enumerate(flat):
            out_row[f"f{j}"] = float(value)

        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_df = out_df.reset_index(drop=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[DONE] Saved processed static dataset to: {OUTPUT_CSV}")
    print(f"[INFO] Shape: {out_df.shape}")

    print("\n[SỐ LƯỢNG THEO LABEL]")
    print(out_df["label"].value_counts())

    print("\n[SỐ LƯỢNG THEO SUBJECT / SESSION / LABEL]")
    print(out_df.groupby(["subject_id", "session_id", "label"]).size())


if __name__ == "__main__":
    main()