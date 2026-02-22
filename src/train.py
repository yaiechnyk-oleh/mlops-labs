"""
MLOps Lab 1 & 2: Twitter Sentiment Analysis Training Script
=============================================================
Trains a text classification model to detect hate speech in tweets.
Uses MLflow for experiment tracking.

Lab 2: Refactored to accept prepared data from DVC pipeline.

Usage (standalone):
    python src/train.py data/prepared data/models

Usage (via DVC pipeline):
    dvc repro
"""

import argparse
import json
import os
import sys
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_NAME = "Twitter_Sentiment_Analysis"
RANDOM_STATE = 42


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

    if model_type in ("logistic_regression", "svm"):
        importances = model.coef_[0]
    elif model_type == "random_forest":
        importances = model.feature_importances_
    else:
        return

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
    """Training pipeline using prepared data, with MLflow tracking."""
    print(f"{'='*60}")
    print(f"  MLOps Lab 2 — Twitter Sentiment Analysis (DVC Pipeline)")
    print(f"  Model: {args.model_type} | C={args.C} | max_features={args.max_features}")
    print(f"{'='*60}\n")

    # ── 1. Load prepared data ─────────────────
    print(f"[1/5] Loading prepared data from {args.input_dir}...")
    train_path = os.path.join(args.input_dir, "train.csv")
    test_path = os.path.join(args.input_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Prepared data not found in {args.input_dir}.\n"
            "Run 'dvc repro' or 'python src/prepare.py data/raw/twitter.csv data/prepared' first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"      Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    # ── 2. Feature extraction ─────────────────
    print("[2/5] Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=2,
    )

    X_train = vectorizer.fit_transform(train_df["clean_tweet"])
    X_test = vectorizer.transform(test_df["clean_tweet"])
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    print(f"      Feature matrix: train={X_train.shape}, test={X_test.shape}\n")

    # ── 3. Train model ────────────────────────
    print(f"[3/5] Training {args.model_type}...")
    model = get_model(args.model_type, args)
    model.fit(X_train, y_train)

    # ── 4. Evaluate ───────────────────────────
    print("[4/5] Evaluating...")
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

    # ── 5. Save outputs & log to MLflow ───────
    print("[5/5] Saving model & logging to MLflow...\n")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save metrics as JSON (for DVC metrics tracking)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save plots
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred_test, cm_path)

    fi_path = os.path.join(args.output_dir, "feature_importance.png")
    plot_feature_importance(model, vectorizer, args.model_type, fi_path)

    # Log to MLflow
    mlflow.set_tracking_uri(f"file://{os.path.join(PROJECT_ROOT, 'mlruns')}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # Tags
        mlflow.set_tag("model_type", args.model_type)
        mlflow.set_tag("author", "student")
        mlflow.set_tag("dataset_version", "v1.0")
        mlflow.set_tag("task", "binary_classification")
        mlflow.set_tag("pipeline", "dvc")

        # Parameters
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("ngram_max", args.ngram_max)
        mlflow.log_param("C", args.C)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])

        # Metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Artifacts
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)
        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id
        print(f"      ✅ MLflow Run ID: {run_id}")
        print(f"      📊 Experiment: {EXPERIMENT_NAME}")
        print(f"      💾 Model & metrics saved to: {args.output_dir}")
        print(f"      🔗 View: mlflow ui → http://127.0.0.1:5000\n")

    print("Done! 🎉")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Twitter Sentiment Analysis model with MLflow tracking."
    )
    # Positional arguments for DVC pipeline
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory with prepared train.csv and test.csv",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save model artifacts and metrics",
    )
    # Model hyperparameters
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
