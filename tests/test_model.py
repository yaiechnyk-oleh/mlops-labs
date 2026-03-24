"""
Post-train tests: artifact validation + Quality Gate.
These run AFTER model training.
"""

import json
import os

MODELS_DIR = os.getenv("MODELS_DIR", "data/models")


def test_model_artifact_exists():
    path = os.path.join(MODELS_DIR, "model.pkl")
    assert os.path.exists(path), f"model.pkl not found: {path}"


def test_metrics_json_exists():
    path = os.path.join(MODELS_DIR, "metrics.json")
    assert os.path.exists(path), f"metrics.json not found: {path}"


def test_confusion_matrix_exists():
    path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    assert os.path.exists(path), f"confusion_matrix.png not found: {path}"


def test_metrics_json_structure():
    path = os.path.join(MODELS_DIR, "metrics.json")
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    for key in ["f1", "accuracy", "test_f1", "test_accuracy"]:
        assert key in metrics, f"Missing key '{key}' in metrics.json"
        assert 0.0 <= float(metrics[key]) <= 1.0, f"Metric '{key}' out of [0,1] range"


def test_quality_gate_f1():
    """Quality Gate: model must achieve minimum F1 score."""
    threshold = float(os.getenv("F1_THRESHOLD", "0.35"))
    path = os.path.join(MODELS_DIR, "metrics.json")
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    f1 = float(metrics["f1"])
    assert f1 >= threshold, (
        f"Quality Gate FAILED: f1={f1:.4f} < threshold={threshold:.2f}"
    )
