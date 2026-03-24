"""
Pre-train tests: data validation.
These run BEFORE model training to catch data issues early.
"""

import os

import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "data/prepared")


def test_train_file_exists():
    path = os.path.join(DATA_DIR, "train.csv")
    assert os.path.exists(path), f"train.csv not found: {path}"


def test_test_file_exists():
    path = os.path.join(DATA_DIR, "test.csv")
    assert os.path.exists(path), f"test.csv not found: {path}"


def test_data_schema():
    for split in ("train", "test"):
        path = os.path.join(DATA_DIR, f"{split}.csv")
        df = pd.read_csv(path)
        required = {"clean_tweet", "label"}
        missing = required - set(df.columns)
        assert not missing, f"Missing columns in {split}.csv: {missing}"


def test_no_nulls():
    for split in ("train", "test"):
        df = pd.read_csv(os.path.join(DATA_DIR, f"{split}.csv"))
        assert df["label"].notna().all(), f"label has nulls in {split}.csv"
        assert df["clean_tweet"].notna().all(), f"clean_tweet has nulls in {split}.csv"


def test_minimum_rows():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    assert df.shape[0] >= 100, f"Too few rows in train.csv: {df.shape[0]}"


def test_label_values():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    assert set(df["label"].unique()).issubset({0, 1}), "Labels must be 0 or 1"


def test_no_empty_tweets():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    empty = df["clean_tweet"].str.strip().eq("").sum()
    assert empty == 0, f"{empty} empty tweets found in train.csv"


def test_class_balance_not_extreme():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    minority_ratio = df["label"].mean()
    assert minority_ratio >= 0.01, f"Minority class ratio too low: {minority_ratio:.3f}"
