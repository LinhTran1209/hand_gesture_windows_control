from __future__ import annotations

from pathlib import Path
import cv2 as cv
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEOS_ROOT = PROJECT_ROOT / "data" / "raw" / "self_collected" / "videos"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "self_collected" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "manifest_v1.csv"


SESSION_LIGHT_MAP = {
    "session_01": "normal_light",
    "session_02": "lamp_light",
}


def get_video_info(video_path: Path) -> dict:
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "fps": 0.0,
            "frame_count": 0,
            "duration_sec": 0.0,
            "is_readable": 0,
        }

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
    if not VIDEOS_ROOT.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục video: {VIDEOS_ROOT}")

    rows = []

    video_files = sorted(VIDEOS_ROOT.rglob("*.mp4"))
    if not video_files:
        raise RuntimeError("Không tìm thấy file .mp4 nào trong dataset.")

    for video_path in video_files:
        try:
            label = video_path.parent.name
            session_id = video_path.parent.parent.name
            subject_id = video_path.parent.parent.parent.name
        except Exception as exc:
            print(f"[SKIP] Không parse được path: {video_path} | {exc}")
            continue

        if not subject_id.startswith("subject_"):
            continue
        if not session_id.startswith("session_"):
            continue

        light_condition = SESSION_LIGHT_MAP.get(session_id, "unknown")
        info = get_video_info(video_path)

        rows.append(
            {
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

    df = pd.DataFrame(rows)
    df = df.sort_values(["subject_id", "session_id", "label", "video_name"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[DONE] Đã tạo manifest: {OUTPUT_CSV}")
    print(f"[INFO] Tổng số video: {len(df)}")

    print("\n[SỐ LƯỢNG THEO LABEL]")
    print(df.groupby("label").size())

    print("\n[SỐ LƯỢNG THEO SUBJECT / SESSION / LABEL]")
    print(df.groupby(["subject_id", "session_id", "label"]).size())

    unreadable = df[df["is_readable"] == 0]
    if len(unreadable) > 0:
        print("\n[WARN] Có video không đọc được:")
        print(unreadable[["video_path"]])


if __name__ == "__main__":
    main()