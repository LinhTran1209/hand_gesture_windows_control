from __future__ import annotations
from typing import Any
import numpy as np

EPS = 1e-6


# chuẩn hóa landmarks: chuẩn hóa điểm cổ tay (wrist) về 0,0,0 và chuẩn hóa khoảng cách lớn nhất về 1 theo wirst
def normalize_landmarks(landmarks_xyz: np.ndarray) -> np.ndarray:
    if landmarks_xyz.shape != (21, 3):
        raise ValueError(f"Expected shape (21, 3), got {landmarks_xyz.shape}")

    wrist = landmarks_xyz[0].copy()
    centered = landmarks_xyz - wrist

    distances = np.linalg.norm(centered, axis=1)
    scale = float(np.max(distances))
    if scale < EPS:
        scale = 1.0

    return centered / scale


# hàm chuyển list landmarks (dạng mediapipe) thành mảng numpy shape (21, 3) với thứ tự x,y,z
def landmarks_to_xyz_array(landmarks: Any) -> np.ndarray:
    if landmarks is None or len(landmarks) != 21:
        raise ValueError("Landmarks không hợp lệ hoặc không đủ 21 điểm.")

    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    if coords.shape != (21, 3):
        raise ValueError(f"Expected coords shape (21, 3), got {coords.shape}")

    return coords


# hàm chuyển mang 21 điểm x,y,z thành vector 63 chiều đã được chuẩn hóa
def xyz_array_to_feature_vector(coords: np.ndarray) -> np.ndarray:
    normalized = normalize_landmarks(coords)
    feature = normalized.reshape(-1)

    if feature.shape[0] != 63:
        raise ValueError(f"Feature dimension sai: {feature.shape[0]}")

    return feature.astype(np.float32)


# hàm chính: nhận landmarks (dạng mediapipe) và trả về vector đặc trưng 63 chiều đã được chuẩn hóa
def landmarks_to_feature_vector(landmarks: Any) -> np.ndarray:
    coords = landmarks_to_xyz_array(landmarks)
    return xyz_array_to_feature_vector(coords)
