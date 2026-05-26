from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

DATASET_CONFIG = {
    "static": {
        "data_dir": PROJECT_ROOT / "data" / "processed" / "static",
        "input_csv": "static_dataset_v1.csv",
        "train_csv": "train_static_v1.csv",
        "val_csv": "val_static_v1.csv",
        "test_csv": "test_static_v1.csv",
    },
    "dynamic": {
        "data_dir": PROJECT_ROOT / "data" / "processed" / "dynamic",
        "input_csv": "dynamic_dataset_v1.csv",
        "train_csv": "train_dynamic_v1.csv",
        "val_csv": "val_dynamic_v1.csv",
        "test_csv": "test_dynamic_v1.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split processed dataset into train/val/test with 8:1:1 ratio."
    )
    parser.add_argument("--dataset-type", choices=["static", "dynamic"], required=True)
    parser.add_argument(
        "--split-mode",
        choices=["sample", "subject"],
        default="sample",
        help=(
            "sample: chia đúng tỷ lệ 8:1:1 theo số mẫu và giữ phân bố label; "
            "subject: chia theo subject_id để tránh data leakage. Default: sample"
        ),
    )
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    return parser.parse_args()


def split_subjects(
    subject_ids: list[str], seed: int = RANDOM_SEED
) -> tuple[list[str], list[str], list[str]]:
    if len(subject_ids) < 3:
        raise ValueError(
            "Cần ít nhất 3 subject để chia train / val / test theo subject."
        )

    shuffled = sorted(subject_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = max(1, math.floor(total * TRAIN_RATIO))
    val_count = max(1, math.floor(total * VAL_RATIO))
    test_count = total - train_count - val_count

    if test_count < 1:
        test_count = 1
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1

    while train_count + val_count + test_count > total:
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            test_count -= 1

    while train_count + val_count + test_count < total:
        train_count += 1

    return (
        shuffled[:train_count],
        shuffled[train_count : train_count + val_count],
        shuffled[train_count + val_count :],
    )


def assign_split(
    subject_id: Any,
    train_subjects: set[str],
    val_subjects: set[str],
    test_subjects: set[str],
) -> str:
    subject_id = str(subject_id)
    if subject_id in train_subjects:
        return "train"
    if subject_id in val_subjects:
        return "val"
    if subject_id in test_subjects:
        return "test"
    return "unknown"


def _counts_811(n: int) -> tuple[int, int, int]:
    """Return train/val/test counts as close as possible to 8:1:1.

    For normal class sizes, this gives approximately 80/10/10 per label.
    For tiny classes, it still keeps at least one train sample.
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    train_n = int(round(n * TRAIN_RATIO))
    val_n = int(round(n * VAL_RATIO))
    test_n = n - train_n - val_n

    # Đảm bảo val/test có mẫu nếu class đủ lớn.
    if n >= 10:
        if val_n < 1:
            val_n = 1
        if test_n < 1:
            test_n = 1
        train_n = n - val_n - test_n
    elif n >= 3:
        if test_n < 1:
            test_n = 1
            train_n -= 1
        if train_n < 1:
            train_n = 1
            test_n = n - train_n - val_n

    # Sửa sai số do round hoặc class nhỏ.
    while train_n + val_n + test_n > n:
        if train_n >= val_n and train_n >= test_n and train_n > 1:
            train_n -= 1
        elif val_n >= test_n and val_n > 0:
            val_n -= 1
        elif test_n > 0:
            test_n -= 1
        else:
            break

    while train_n + val_n + test_n < n:
        train_n += 1

    return train_n, val_n, test_n


def stratified_sample_split(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Split sample-level theo label để đạt tỷ lệ 8:1:1 sát nhất.

    Cách này phù hợp khi mục tiêu chính là đúng tỷ lệ số lượng mẫu.
    """
    if "label" not in df.columns:
        raise ValueError("Dataset phải có cột 'label' để chia stratified sample-level.")

    pieces: list[pd.DataFrame] = []

    for label, group in df.groupby("label", dropna=False, sort=True):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(group)
        train_n, val_n, test_n = _counts_811(n)

        split = ["train"] * train_n + ["val"] * val_n + ["test"] * test_n
        split = split[:n]

        group = group.copy()
        group["split"] = split
        pieces.append(group)

    result = pd.concat(pieces, ignore_index=True)
    result = result.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return result


def subject_split(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    if "subject_id" not in df.columns:
        raise ValueError("Dataset phải có cột 'subject_id' để chia theo subject.")

    unique_subjects = sorted(df["subject_id"].dropna().astype(str).unique().tolist())
    train_subjects, val_subjects, test_subjects = split_subjects(
        unique_subjects, seed=seed
    )

    train_set, val_set, test_set = (
        set(train_subjects),
        set(val_subjects),
        set(test_subjects),
    )

    result = df.copy()
    result["split"] = (
        result["subject_id"]
        .astype(str)
        .apply(lambda s: assign_split(s, train_set, val_set, test_set))
    )
    return result


def print_split_report(df: pd.DataFrame) -> None:
    total = len(df)
    print("\n[INFO] Split summary")
    for name in ["train", "val", "test"]:
        count = int((df["split"] == name).sum())
        ratio = count / total if total > 0 else 0.0
        print(f"  {name:<5}: {count:>6} samples ({ratio:.2%})")

    if "label" in df.columns:
        print("\n[INFO] Label distribution by split")
        table = pd.crosstab(df["label"], df["split"])
        for col in ["train", "val", "test"]:
            if col not in table.columns:
                table[col] = 0
        print(table[["train", "val", "test"]])


def main() -> None:
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset_type]
    data_dir = cfg["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)

    input_csv = args.input_csv or data_dir / cfg["input_csv"]
    train_csv = args.train_csv or data_dir / cfg["train_csv"]
    val_csv = args.val_csv or data_dir / cfg["val_csv"]
    test_csv = args.test_csv or data_dir / cfg["test_csv"]

    if not input_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy file dataset: {input_csv}")

    df = pd.read_csv(input_csv)
    if len(df) == 0:
        raise RuntimeError("Dataset rỗng, không thể split.")

    if args.split_mode == "sample":
        df = stratified_sample_split(df, seed=RANDOM_SEED)
    else:
        df = subject_split(df, seed=RANDOM_SEED)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_csv, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_csv, index=False, encoding="utf-8-sig")

    print_split_report(df)
    print(f"\n[DONE] Saved train: {train_csv}")
    print(f"[DONE] Saved val:   {val_csv}")
    print(f"[DONE] Saved test:  {test_csv}")


if __name__ == "__main__":
    main()
