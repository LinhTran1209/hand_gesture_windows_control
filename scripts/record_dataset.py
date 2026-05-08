from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2 as cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record self-collected hand gesture dataset clips."
    )
    parser.add_argument(
        "--label", type=str, required=True, help="Gesture label, ví dụ: open_palm"
    )
    parser.add_argument(
        "--subject", type=str, required=True, help="Subject ID, ví dụ: 01"
    )
    parser.add_argument(
        "--session", type=str, required=True, help="Session ID, ví dụ: 01"
    )
    parser.add_argument(
        "--duration", type=float, default=3.0, help="Thời lượng mỗi clip (giây)"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="FPS ghi video")
    parser.add_argument("--mirror", action="store_true", help="Lật gương khung hình")
    return parser.parse_args()


def create_output_dir(
    project_root: Path, subject: str, session: str, label: str
) -> Path:
    output_dir = (
        project_root
        / "data"
        / "raw"
        / "self_collected"
        / "videos"
        / "static"  # static or dynamic -----------------------------------------------------------------------------
        / f"subject_{subject}"
        / f"session_{session}"
        / label
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_output_path(output_dir: Path, label: str, subject: str, session: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_subject_{subject}_session_{session}_{timestamp}.mp4"
    return output_dir / filename


def put_text_block(frame, lines, start_x=20, start_y=30, color=(0, 255, 255)):
    y = start_y
    for line in lines:
        cv.putText(
            frame,
            line,
            (start_x, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv.LINE_AA,
        )
        y += 32


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = create_output_dir(project_root, args.subject, args.session, args.label)

    cap = cv.VideoCapture(args.camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam.")

    writer = None
    recording = False
    record_start = 0.0
    current_output_path = None
    clip_count = len(list(output_dir.glob("*.mp4")))

    print(f"[INFO] Đang lưu dữ liệu vào: {output_dir}")
    print("[HƯỚNG DẪN]")
    print(" - Nhấn 'r' để bắt đầu quay 1 clip")
    print(" - Mỗi clip sẽ tự dừng sau số giây đã cấu hình")
    print(" - Nhấn 'q' để thoát")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Không đọc được frame từ webcam.")
                continue

            if args.mirror:
                frame = cv.flip(frame, 1)

            display_frame = frame.copy()

            if recording and writer is not None:
                elapsed = time.time() - record_start
                remaining = max(args.duration - elapsed, 0.0)

                writer.write(frame)

                cv.circle(display_frame, (30, 110), 10, (0, 0, 255), -1)
                put_text_block(
                    display_frame,
                    [
                        f"Label: {args.label}",
                        f"Subject: {args.subject} | Session: {args.session}",
                        f"REC... remaining: {remaining:.1f}s",
                        f"Saved clips: {clip_count}",
                    ],
                )

                if elapsed >= args.duration:
                    writer.release()
                    writer = None
                    recording = False
                    clip_count += 1
                    print(f"[SAVED] {current_output_path}")
                    current_output_path = None
            else:
                put_text_block(
                    display_frame,
                    [
                        f"Label: {args.label}",
                        f"Subject: {args.subject} | Session: {args.session}",
                        f"Duration: {args.duration:.1f}s / clip",
                        f"Saved clips: {clip_count}",
                        "Press 'r' to record | Press 'q' to quit",
                    ],
                )

            cv.imshow("Record Dataset", display_frame)
            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r") and not recording:
                current_output_path = build_output_path(
                    output_dir, args.label, args.subject, args.session
                )
                fourcc = cv.VideoWriter_fourcc(*"mp4v")
                writer = cv.VideoWriter(
                    str(current_output_path),
                    fourcc,
                    args.fps,
                    (frame.shape[1], frame.shape[0]),
                )

                if not writer.isOpened():
                    writer = None
                    raise RuntimeError("Không tạo được file video output.")

                recording = True
                record_start = time.time()
                print(f"[REC] Bắt đầu ghi: {current_output_path.name}")

    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
