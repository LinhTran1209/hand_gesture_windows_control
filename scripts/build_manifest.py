from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "data" / "raw" / "self_collected" / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

SESSION_LIGHT_MAP = {
    "session_01": "normal_light",
    "session_02": "lamp_light",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manifest for static or dynamic self-collected videos."
    )
    parser.add_argument("--dataset-type", choices=["static", "dynamic"], required=True)
    parser.add_argument("--videos-root", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def get_video_info(video_path: Path) -> dict:
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"fps": 0.0, "frame_count": 0, "duration_sec": 0.0, "is_readable": 0}

    fps = float(cap.get(cv.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "is_readable": 1,
    }


def main() -> None:
    args = parse_args()

    dataset_type = args.dataset_type
    videos_root = args.videos_root or (
        PROJECT_ROOT / "data" / "raw" / "self_collected" / "videos" / dataset_type
    )
    output_csv = args.output_csv or METADATA_DIR / f"manifest_{dataset_type}_v1.csv"

    print("[INFO] dataset_type:", dataset_type)
    print("[INFO] videos_root :", videos_root)
    print("[INFO] output_csv  :", output_csv)

    if not videos_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục video: {videos_root}")

    rows = []
    video_files = sorted(videos_root.rglob("*.mp4"))
    if not video_files:
        raise RuntimeError(f"Không tìm thấy file .mp4 nào trong {videos_root}")

    for video_path in video_files:
        try:
            label = video_path.parent.name
            session_id = video_path.parent.parent.name
            subject_id = video_path.parent.parent.parent.name
        except Exception as exc:
            print(f"[SKIP] Không parse được path: {video_path} | {exc}")
            continue

        if not subject_id.startswith("subject_"):
            print(f"[SKIP] Sai subject_id format: {video_path}")
            continue
        if not session_id.startswith("session_"):
            print(f"[SKIP] Sai session_id format: {video_path}")
            continue

        light_condition = SESSION_LIGHT_MAP.get(session_id, "unknown")
        info = get_video_info(video_path)

        rows.append(
            {
                "dataset_type": dataset_type,
                "video_path": str(video_path.resolve()),
                "video_name": video_path.name,
                "subject_id": subject_id,
                "session_id": session_id,
                "label": label,
                "lighting_condition": light_condition,
                "fps": info["fps"],
                "frame_count": info["frame_count"],
                "duration_sec": round(info["duration_sec"], 4),
                "is_readable": info["is_readable"],
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values(["subject_id", "session_id", "label", "video_name"])
        .reset_index(drop=True)
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] Đã tạo manifest: {output_csv}")
    print(f"[INFO] Tổng số video: {len(df)}")
    print("\n[SỐ LƯỢNG THEO LABEL]")
    print(df.groupby("label").size())
    print("\n[SỐ LƯỢNG THEO SUBJECT / SESSION / LABEL]")
    print(df.groupby(["subject_id", "session_id", "label"]).size())


if __name__ == "__main__":
    main()
