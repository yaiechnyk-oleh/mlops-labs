"""
MLOps Lab 2: Data Preparation Script
=====================================
Cleans raw tweet data, preprocesses text, and splits into train/test sets.
This is the first stage of the DVC pipeline.

Usage:
    python src/prepare.py data/raw/twitter.csv data/prepared
"""

import os
import re
import sys

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ──────────────────────────────────────────────
# Text preprocessing
# ──────────────────────────────────────────────
def download_nltk_resources():
    """Download required NLTK data."""
    for resource in ["stopwords", "punkt"]:
        try:
            nltk.data.find(
                f"corpora/{resource}" if resource == "stopwords" else f"tokenizers/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)


def clean_tweet(text: str) -> str:
    """Clean a single tweet: remove URLs, mentions, special chars, lowercase."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)  # mentions
    text = re.sub(r"#(\w+)", r"\1", text)  # hashtags (keep word)
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # special chars & numbers
    text = re.sub(r"\s+", " ", text).strip()  # extra whitespace
    return text


def remove_stopwords(text: str, stop_words: set) -> str:
    """Remove English stopwords from text."""
    return " ".join(word for word in text.split() if word not in stop_words)


def prepare(input_file: str, output_dir: str):
    """Full data preparation pipeline."""
    print(f"{'='*60}")
    print(f"  Data Preparation (DVC Stage: prepare)")
    print(f"{'='*60}\n")

    # ── 1. Load raw data ──────────────────────
    print(f"[1/4] Loading raw data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"      Loaded {len(df)} rows, columns: {list(df.columns)}")

    # ── 2. Clean text ─────────────────────────
    print("[2/4] Cleaning text...")
    download_nltk_resources()
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))

    df["clean_tweet"] = df["tweet"].apply(clean_tweet)
    df["clean_tweet"] = df["clean_tweet"].apply(lambda t: remove_stopwords(t, stop_words))

    # Remove empty rows after cleaning
    before = len(df)
    df = df[df["clean_tweet"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"      Removed {before - len(df)} empty rows. Remaining: {len(df)}")

    # ── 3. Split into train/test ──────────────
    print(f"[3/4] Splitting data (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"]
    )
    print(f"      Train: {len(train_df)} rows")
    print(f"      Test:  {len(test_df)} rows")
    print(f"      Class distribution (train):\n{train_df['label'].value_counts().to_string()}")

    # ── 4. Save prepared data ─────────────────
    print(f"\n[4/4] Saving to {output_dir}/...")
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"      ✅ {train_path} ({len(train_df)} rows)")
    print(f"      ✅ {test_path} ({len(test_df)} rows)")
    print(f"\nDone! 🎉")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/prepare.py <input_csv> <output_dir>")
        print("Example: python src/prepare.py data/raw/twitter.csv data/prepared")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    prepare(input_file, output_dir)
