from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

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
        description="Split processed dataset into train/val/test."
    )
    parser.add_argument("--dataset-type", choices=["static", "dynamic"], required=True)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--fallback-sample-split", action="store_true")
    return parser.parse_args()


def split_subjects(
    subject_ids: list[str], seed: int = RANDOM_SEED
) -> tuple[list[str], list[str], list[str]]:
    if len(subject_ids) < 3:
        raise ValueError("Cần ít nhất 3 subject để chia train / val / test.")
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
    if subject_id in train_subjects:
        return "train"
    if subject_id in val_subjects:
        return "val"
    if subject_id in test_subjects:
        return "test"
    return "unknown"


def stratified_sample_split(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    pieces = []
    for _, group in df.groupby("label", dropna=False):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(group)
        train_n = max(1, math.floor(n * TRAIN_RATIO)) if n >= 3 else max(1, n)
        val_n = max(1, math.floor(n * VAL_RATIO)) if n >= 5 else 0
        test_n = n - train_n - val_n

        if n >= 3 and test_n < 1:
            test_n = 1
            if train_n > val_n and train_n > 1:
                train_n -= 1
            elif val_n > 1:
                val_n -= 1

        split = ["train"] * train_n + ["val"] * val_n + ["test"] * max(0, test_n)
        split = split[:n]
        if len(split) < n:
            split.extend(["train"] * (n - len(split)))

        group = group.copy()
        group["split"] = split
        pieces.append(group)

    return pd.concat(pieces, ignore_index=True)


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

    unique_subjects = sorted(df["subject_id"].dropna().astype(str).unique().tolist())

    if len(unique_subjects) >= 3:
        train_subjects, val_subjects, test_subjects = split_subjects(
            unique_subjects, seed=RANDOM_SEED
        )
        train_set, val_set, test_set = (
            set(train_subjects),
            set(val_subjects),
            set(test_subjects),
        )
        df = df.copy()
        df["split"] = (
            df["subject_id"]
            .astype(str)
            .apply(lambda s: assign_split(s, train_set, val_set, test_set))
        )
    else:
        if not args.fallback_sample_split:
            raise ValueError(
                "Số subject < 3. Dùng --fallback-sample-split nếu muốn chia theo sample-level."
            )
        print("[WARN] Số subject < 3, fallback sang sample-level split theo label.")
        df = stratified_sample_split(df, seed=RANDOM_SEED)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_csv, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] Saved train: {train_csv}")
    print(f"[DONE] Saved val:   {val_csv}")
    print(f"[DONE] Saved test:  {test_csv}")


if __name__ == "__main__":
    main()
