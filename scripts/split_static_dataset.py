from __future__ import annotations

from pathlib import Path
import math
import random

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.joinpath("data", "processed", "static")

INPUT_CSV = DATA_DIR.joinpath("static_dataset_v1.csv")
TRAIN_CSV = DATA_DIR.joinpath("train_static_v1.csv")
VAL_CSV = DATA_DIR.joinpath("val_static_v1.csv")
TEST_CSV = DATA_DIR.joinpath("test_static_v1.csv")

RANDOM_SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2


def split_subjects(
    subject_ids: list[str], seed: int = RANDOM_SEED
) -> tuple[list[str], list[str], list[str]]:
    """
    Chia subject theo tỷ lệ train/val/test.
    Mỗi subject chỉ được nằm trong đúng 1 split.
    """
    if len(subject_ids) < 3:
        raise ValueError("Cần ít nhất 3 subject để chia train / val / test.")

    shuffled = sorted(subject_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)

    train_count = max(1, math.floor(total * TRAIN_RATIO))
    val_count = max(1, math.floor(total * VAL_RATIO))
    test_count = total - train_count - val_count

    # đảm bảo test có ít nhất 1 subject
    if test_count < 1:
        test_count = 1
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1

    # nếu tổng bị lệch do chỉnh tay thì sửa lại
    while train_count + val_count + test_count > total:
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            test_count -= 1

    while train_count + val_count + test_count < total:
        train_count += 1

    train_subjects = shuffled[:train_count]
    val_subjects = shuffled[train_count : train_count + val_count]
    test_subjects = shuffled[train_count + val_count :]

    return train_subjects, val_subjects, test_subjects


def assign_split(
    subject_id: str,
    train_subjects: set[str],
    val_subjects: set[str],
    test_subjects: set[str],
) -> str:
    if subject_id in train_subjects:
        return "train"
    if subject_id in val_subjects:
        return "val"
    if subject_id in test_subjects:
        return "test"
    return "unknown"


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy file dataset: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    unique_subjects = sorted(df["subject_id"].dropna().unique().tolist())
    print("[INFO] Danh sách subject:", unique_subjects)

    train_subjects, val_subjects, test_subjects = split_subjects(
        unique_subjects, seed=RANDOM_SEED
    )

    print("\n[INFO] Subject split")
    print("Train subjects:", train_subjects)
    print("Val subjects:  ", val_subjects)
    print("Test subjects: ", test_subjects)

    train_subjects_set = set(train_subjects)
    val_subjects_set = set(val_subjects)
    test_subjects_set = set(test_subjects)

    df["split"] = df["subject_id"].apply(
        lambda subject_id: assign_split(
            subject_id, train_subjects_set, val_subjects_set, test_subjects_set
        )
    )

    unknown_df = df[df["split"] == "unknown"].copy()
    if len(unknown_df) > 0:
        print("\n[WARN] Có sample không được gán split:")
        print(unknown_df[["subject_id"]].drop_duplicates())

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    val_df.to_csv(VAL_CSV, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

    print(f"\n[DONE] Saved train: {TRAIN_CSV}")
    print(f"[DONE] Saved val:   {VAL_CSV}")
    print(f"[DONE] Saved test:  {TEST_CSV}")

    print("\n[SHAPE]")
    print("train:", train_df.shape)
    print("val:  ", val_df.shape)
    print("test: ", test_df.shape)

    print("\n[LABEL COUNTS - TRAIN]")
    print(train_df["label"].value_counts())

    print("\n[LABEL COUNTS - VAL]")
    print(val_df["label"].value_counts())

    print("\n[LABEL COUNTS - TEST]")
    print(test_df["label"].value_counts())

    print("\n[SUBJECT COUNTS BY SPLIT]")
    print(df.groupby("split")["subject_id"].nunique())


if __name__ == "__main__":
    main()
