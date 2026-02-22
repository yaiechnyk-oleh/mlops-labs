"""
MLOps Lab 1: Twitter Sentiment Analysis Training Script
========================================================
Trains a text classification model to detect hate speech in tweets.
Uses MLflow for experiment tracking.

Usage:
    python src/train.py
    python src/train.py --model_type logistic_regression --C 1.0 --max_features 5000
    python src/train.py --model_type random_forest --n_estimators 200
    python src/train.py --model_type svm --C 2.0
"""

import argparse
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import nltk

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "twitter.csv")
EXPERIMENT_NAME = "Twitter_Sentiment_Analysis"
RANDOM_STATE = 42


# ──────────────────────────────────────────────
# Text preprocessing
# ──────────────────────────────────────────────
def download_nltk_resources():
    """Download required NLTK data."""
    for resource in ["stopwords", "punkt"]:
        try:
            nltk.data.find(f"corpora/{resource}" if resource == "stopwords" else f"tokenizers/{resource}")
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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full text preprocessing pipeline."""
    download_nltk_resources()
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))

    df = df.copy()
    df["clean_tweet"] = df["tweet"].apply(clean_tweet)
    df["clean_tweet"] = df["clean_tweet"].apply(lambda t: remove_stopwords(t, stop_words))
    # Remove empty rows after cleaning
    df = df[df["clean_tweet"].str.strip().astype(bool)].reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────
def get_model(model_type: str, args: argparse.Namespace):
    """Return a scikit‑learn estimator based on CLI arguments."""
    if model_type == "logistic_regression":
        return LogisticRegression(
            C=args.C,
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_type == "svm":
        return LinearSVC(
            C=args.C,
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ──────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, output_path: str):
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Clean (0)", "Hate (1)"],
        yticklabels=["Clean (0)", "Hate (1)"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(model, vectorizer, model_type: str, output_path: str, top_n: int = 20):
    """Save a feature importance / top coefficient bar chart."""
    feature_names = vectorizer.get_feature_names_out()

    if model_type == "logistic_regression" or model_type == "svm":
        importances = model.coef_[0]
    elif model_type == "random_forest":
        importances = model.feature_importances_
    else:
        return

    # Top positive and negative features
    indices = np.argsort(importances)
    top_positive = indices[-top_n:]
    top_negative = indices[:top_n]
    top_indices = np.concatenate([top_negative, top_positive])

    plt.figure(figsize=(10, 8))
    plt.barh(
        range(len(top_indices)),
        importances[top_indices],
        color=["#e74c3c" if i < top_n else "#2ecc71" for i in range(len(top_indices))],
    )
    plt.yticks(range(len(top_indices)), feature_names[top_indices], fontsize=8)
    plt.xlabel("Coefficient / Importance")
    plt.title(f"Top {top_n} Features (positive & negative)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ──────────────────────────────────────────────
# Main training pipeline
# ──────────────────────────────────────────────
def train(args: argparse.Namespace):
    """Full training pipeline with MLflow tracking."""
    print(f"{'='*60}")
    print(f"  MLOps Lab 1 — Twitter Sentiment Analysis")
    print(f"  Model: {args.model_type} | C={args.C} | max_features={args.max_features}")
    print(f"{'='*60}\n")

    # ── 1. Load data ──────────────────────────
    print("[1/6] Loading data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            "Please download it from Kaggle and place the CSV file there.\n"
            "URL: https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech"
        )
    df = pd.read_csv(DATA_PATH)
    print(f"      Loaded {len(df)} rows, columns: {list(df.columns)}")

    # ── 2. Preprocess ─────────────────────────
    print("[2/6] Preprocessing text...")
    df = preprocess_data(df)
    print(f"      After cleaning: {len(df)} rows")
    print(f"      Class distribution:\n{df['label'].value_counts().to_string()}\n")

    # ── 3. Feature extraction ─────────────────
    print("[3/6] Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=2,
    )

    X = vectorizer.fit_transform(df["clean_tweet"])
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"      Feature matrix shape: {X.shape}\n")

    # ── 4. Train model ────────────────────────
    print(f"[4/6] Training {args.model_type}...")
    model = get_model(args.model_type, args)
    model.fit(X_train, y_train)

    # ── 5. Evaluate ───────────────────────────
    print("[5/6] Evaluating...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "train_f1": f1_score(y_train, y_pred_train),
        "train_precision": precision_score(y_train, y_pred_train),
        "train_recall": recall_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_f1": f1_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test),
        "test_recall": recall_score(y_test, y_pred_test),
    }

    print(f"\n      {'Metric':<20} {'Train':>10} {'Test':>10}")
    print(f"      {'-'*42}")
    for name in ["accuracy", "f1", "precision", "recall"]:
        print(f"      {name:<20} {metrics[f'train_{name}']:>10.4f} {metrics[f'test_{name}']:>10.4f}")
    print()

    # ── 6. Log to MLflow ──────────────────────
    print("[6/6] Logging to MLflow...\n")
    mlflow.set_tracking_uri(f"file://{os.path.join(PROJECT_ROOT, 'mlruns')}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # Tags
        mlflow.set_tag("model_type", args.model_type)
        mlflow.set_tag("author", "student")
        mlflow.set_tag("dataset_version", "v1.0")
        mlflow.set_tag("task", "binary_classification")

        # Parameters
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("ngram_max", args.ngram_max)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("C", args.C)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])

        # Metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Artifacts — confusion matrix
        cm_path = os.path.join(PROJECT_ROOT, "confusion_matrix.png")
        plot_confusion_matrix(y_test, y_pred_test, cm_path)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # Artifacts — feature importance
        fi_path = os.path.join(PROJECT_ROOT, "feature_importance.png")
        plot_feature_importance(model, vectorizer, args.model_type, fi_path)
        mlflow.log_artifact(fi_path)
        os.remove(fi_path)

        # Log model
        if args.model_type == "svm":
            mlflow.sklearn.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id
        print(f"      ✅ MLflow Run ID: {run_id}")
        print(f"      📊 Experiment: {EXPERIMENT_NAME}")
        print(f"      🔗 View: mlflow ui → http://127.0.0.1:5000\n")

    print("Done! 🎉")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Twitter Sentiment Analysis model with MLflow tracking."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "random_forest", "svm"],
        help="Type of model to train (default: logistic_regression)",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=5000,
        help="Max number of TF-IDF features (default: 5000)",
    )
    parser.add_argument(
        "--ngram_max",
        type=int,
        default=2,
        help="Maximum n-gram size for TF-IDF (default: 2)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization parameter for LR/SVM (default: 1.0)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees for Random Forest (default: 100)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Max depth for Random Forest (default: None)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
