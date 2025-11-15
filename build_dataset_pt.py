#!/usr/bin/env python3
"""
Utility script to build a `dataset.pt` bundle that mirrors the preprocessing
used by the LSTM/GRU baseline notebooks for the Pirate Pain challenge.

Pipeline overview:
  • load raw CSVs from the dataset directory,
  • drop the constant `joint_30` feature,
  • encode categorical columns (`n_legs`, `n_hands`, `n_eyes`),
  • normalise continuous features using train-set statistics,
  • split pirates (subjects) into train/validation/test folds,
  • roll sliding windows (window size 50, stride 10 by default),
  • save tensors plus helpful metadata (window ids, label mapping, etc.).

The resulting bundle exposes:
  - X_train / y_train
  - X_val / y_val
  - X_test / y_test          (held-out pirates from training set)
  - X_inference              (windows built from Kaggle test set)
  - train/val/test subject lists and per-window subject ids
  - label mapping + categorical mappings + preprocessing meta

Example:
    python build_dataset_pt.py --output dataset.pt \\
        --val-users 60 --test-users 60 --window-size 50 --stride 10 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


FEATURE_COLUMNS = [
    "pain_survey_1",
    "pain_survey_2",
    "pain_survey_3",
    "pain_survey_4",
    "n_legs",
    "n_hands",
    "n_eyes",
] + [f"joint_{i:02d}" for i in range(30)]  # joint_30 removed

SCALE_COLUMNS = [
    "pain_survey_1",
    "pain_survey_2",
    "pain_survey_3",
    "pain_survey_4",
] + [f"joint_{i:02d}" for i in range(30)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_raw_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dtype_spec = {"sample_index": str}
    train_df = pd.read_csv(data_dir / "pirate_pain_train.csv", dtype=dtype_spec)
    labels_df = pd.read_csv(data_dir / "pirate_pain_train_labels.csv", dtype=dtype_spec)
    kaggle_df = pd.read_csv(data_dir / "pirate_pain_test.csv", dtype=dtype_spec)
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv", dtype=dtype_spec)
    return train_df, labels_df, kaggle_df, sample_submission


def zero_pad_sample_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sample_index"] = df["sample_index"].astype(str).str.zfill(3)
    return df


def drop_constant_joint(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "joint_30" in df.columns:
        df = df.drop(columns=["joint_30"])
    return df


def encode_categoricals(
    *frames: pd.DataFrame,
) -> Tuple[Tuple[pd.DataFrame, ...], Dict[str, Dict[str, int]]]:
    categorical_cols = ["n_legs", "n_hands", "n_eyes"]
    combined = pd.concat([df[categorical_cols] for df in frames], axis=0)
    mappings: Dict[str, Dict[str, int]] = {}
    encoded_frames: List[pd.DataFrame] = []

    for col in categorical_cols:
        categories = sorted(combined[col].astype(str).unique())
        mappings[col] = {cat: idx for idx, cat in enumerate(categories)}

    for df in frames:
        encoded = df.copy()
        for col, mapping in mappings.items():
            encoded[col] = encoded[col].astype(str).map(mapping).astype(np.float32)
        encoded_frames.append(encoded)

    return tuple(encoded_frames), mappings


def train_val_test_split(
    sample_indices: Iterable[str],
    val_users: int,
    test_users: int,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    users = list(sample_indices)
    if val_users + test_users >= len(users):
        raise ValueError(
            f"val_users + test_users ({val_users + test_users}) must be less than total subjects ({len(users)})"
        )
    rng = random.Random(seed)
    rng.shuffle(users)
    train_users = users[: len(users) - val_users - test_users]
    val_slice = users[len(users) - val_users - test_users : len(users) - test_users]
    test_slice = users[len(users) - test_users :]
    return train_users, val_slice, test_slice


def build_sequences(
    df_data: pd.DataFrame,
    feature_columns: List[str],
    label_lookup: Dict[str, int],
    window: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    assert window % stride == 0, "Window must be divisible by stride"

    dataset: List[np.ndarray] = []
    labels: List[int] = []
    window_subjects: List[str] = []

    for sample_idx, group in df_data.groupby("sample_index"):
        values = group[feature_columns].values.astype(np.float32)
        num_features = values.shape[1]
        padding_len = window - (len(values) % window)
        if padding_len > 0:
            padding = np.zeros((padding_len, num_features), dtype=np.float32)
            values = np.concatenate((values, padding), axis=0)

        idx = 0
        while idx + window <= len(values):
            dataset.append(values[idx : idx + window])
            labels.append(label_lookup[sample_idx])
            window_subjects.append(sample_idx)
            idx += stride

    return (
        np.stack(dataset),
        np.array(labels, dtype=np.int64),
        window_subjects,
    )


def build_sequences_inference(
    df_data: pd.DataFrame,
    feature_columns: List[str],
    window: int,
    stride: int,
) -> Tuple[np.ndarray, List[str]]:
    assert window % stride == 0, "Window must be divisible by stride"

    dataset: List[np.ndarray] = []
    window_subjects: List[str] = []

    for sample_idx, group in df_data.groupby("sample_index"):
        values = group[feature_columns].values.astype(np.float32)
        num_features = values.shape[1]
        padding_len = window - (len(values) % window)
        if padding_len > 0:
            padding = np.zeros((padding_len, num_features), dtype=np.float32)
            values = np.concatenate((values, padding), axis=0)

        idx = 0
        while idx + window <= len(values):
            dataset.append(values[idx : idx + window])
            window_subjects.append(sample_idx)
            idx += stride

    return np.stack(dataset), window_subjects


def normalise_frame(df: pd.DataFrame, mins: pd.Series, maxs: pd.Series) -> pd.DataFrame:
    df = df.copy()
    for column in SCALE_COLUMNS:
        denom = maxs[column] - mins[column]
        denom = denom if denom != 0 else 1.0
        df[column] = (df[column] - mins[column]) / denom
    return df


def normalise_frame_per_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each subject independently - better for handling inter-subject variability."""
    df = df.copy()
    for sample_idx, group in df.groupby("sample_index"):
        for column in SCALE_COLUMNS:
            col_min = group[column].min()
            col_max = group[column].max()
            denom = col_max - col_min
            denom = denom if denom != 0 else 1.0
            df.loc[df["sample_index"] == sample_idx, column] = (
                (group[column] - col_min) / denom
            )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dataset.pt from raw Pirate Pain CSV files.")
    parser.add_argument("--data-dir", default="dataset", type=str, help="Directory containing pirate_pain_*.csv files.")
    parser.add_argument("--output", default="dataset.pt", type=str, help="Path of the output torch file.")
    parser.add_argument("--val-users", default=60, type=int, help="Number of pirates to reserve for validation.")
    parser.add_argument("--test-users", default=60, type=int, help="Number of pirates to reserve for the internal test.")
    parser.add_argument("--window-size", default=50, type=int, help="Sliding window length.")
    parser.add_argument("--stride", default=10, type=int, help="Stride used when rolling windows.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--per-subject-norm", action="store_true", help="Normalize each subject independently.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    set_seed(args.seed)

    train_df, labels_df, kaggle_df, sample_submission = read_raw_data(data_dir)
    train_df = zero_pad_sample_index(train_df)
    labels_df = zero_pad_sample_index(labels_df)
    kaggle_df = zero_pad_sample_index(kaggle_df)
    sample_submission = zero_pad_sample_index(sample_submission)

    train_df = drop_constant_joint(train_df)
    kaggle_df = drop_constant_joint(kaggle_df)

    (train_df, kaggle_df), categorical_mappings = encode_categoricals(train_df, kaggle_df)

    label_mapping = {label: idx for idx, label in enumerate(sorted(labels_df["label"].unique()))}
    idx_to_label = {idx: label for label, idx in label_mapping.items()}
    labels_df = labels_df.copy()
    labels_df["label"] = labels_df["label"].map(label_mapping)
    label_lookup = labels_df.set_index("sample_index")["label"].to_dict()

    train_users, val_users, test_users = train_val_test_split(
        train_df["sample_index"].unique(), args.val_users, args.test_users, args.seed
    )

    df_train_split = train_df[train_df["sample_index"].isin(train_users)].copy()
    df_val_split = train_df[train_df["sample_index"].isin(val_users)].copy()
    df_test_split = train_df[train_df["sample_index"].isin(test_users)].copy()
    df_kaggle = kaggle_df.copy()

    if args.per_subject_norm:
        # Per-subject normalization (better for inter-subject variability)
        print("Using per-subject normalization...")
        df_train_split = normalise_frame_per_subject(df_train_split)
        df_val_split = normalise_frame_per_subject(df_val_split)
        df_test_split = normalise_frame_per_subject(df_test_split)
        df_kaggle = normalise_frame_per_subject(df_kaggle)
        # Still compute global stats for reference (but not used)
        mins = df_train_split[SCALE_COLUMNS].min()
        maxs = df_train_split[SCALE_COLUMNS].max()
    else:
        # Global normalization (original approach)
        mins = df_train_split[SCALE_COLUMNS].min()
        maxs = df_train_split[SCALE_COLUMNS].max()
        df_train_split = normalise_frame(df_train_split, mins, maxs)
        df_val_split = normalise_frame(df_val_split, mins, maxs)
        df_test_split = normalise_frame(df_test_split, mins, maxs)
        df_kaggle = normalise_frame(df_kaggle, mins, maxs)

    X_train, y_train, train_window_ids = build_sequences(
        df_train_split, FEATURE_COLUMNS, label_lookup, args.window_size, args.stride
    )
    X_val, y_val, val_window_ids = build_sequences(
        df_val_split, FEATURE_COLUMNS, label_lookup, args.window_size, args.stride
    )
    X_test, y_test, test_window_ids = build_sequences(
        df_test_split, FEATURE_COLUMNS, label_lookup, args.window_size, args.stride
    )

    X_inference, inference_window_ids = build_sequences_inference(
        df_kaggle, FEATURE_COLUMNS, args.window_size, args.stride
    )

    bundle = {
        "X_train": torch.from_numpy(X_train).float(),
        "y_train": torch.from_numpy(y_train).long(),
        "X_val": torch.from_numpy(X_val).float(),
        "y_val": torch.from_numpy(y_val).long(),
        "X_test": torch.from_numpy(X_test).float(),
        "y_test": torch.from_numpy(y_test).long(),
        "X_inference": torch.from_numpy(X_inference).float(),
        "train_window_ids": train_window_ids,
        "val_window_ids": val_window_ids,
        "test_window_ids": test_window_ids,
        "inference_window_ids": inference_window_ids,
        "train_subjects": train_users,
        "val_subjects": val_users,
        "test_subjects": test_users,
        "kaggle_ids": sample_submission["sample_index"].tolist(),
        "label_mapping": label_mapping,
        "idx_to_label": idx_to_label,
        "categorical_mappings": categorical_mappings,
        "feature_columns": FEATURE_COLUMNS,
        "scale_columns": SCALE_COLUMNS,
        "window_size": args.window_size,
        "stride": args.stride,
        "val_users": args.val_users,
        "test_users": args.test_users,
        "mins": torch.tensor(mins.values, dtype=torch.float32),
        "maxs": torch.tensor(maxs.values, dtype=torch.float32),
        "meta": {
            "seed": args.seed,
            "data_dir": str(data_dir.resolve()),
        },
    }

    output_path = Path(args.output)
    torch.save(bundle, output_path)

    summary = {
        "output": str(output_path.resolve()),
        "X_train": tuple(bundle["X_train"].shape),
        "X_val": tuple(bundle["X_val"].shape),
        "X_test": tuple(bundle["X_test"].shape),
        "X_inference": tuple(bundle["X_inference"].shape),
        "labels_train": {idx_to_label[int(k)]: int((bundle["y_train"] == int(k)).sum()) for k in label_mapping.values()},
        "window_size": args.window_size,
        "stride": args.stride,
        "val_users": args.val_users,
        "test_users": args.test_users,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


