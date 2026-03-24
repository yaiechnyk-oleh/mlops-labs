"""
MLOps Lab 3: Hyperparameter Optimization with Optuna + Hydra + MLflow
======================================================================
Runs HPO on the Twitter hate speech classification pipeline.
Each Optuna trial is logged as a nested MLflow run inside a parent study run.

Usage:
    # Default: TPE sampler, logistic regression, 20 trials
    python src/optimize.py

    # Switch model or sampler via Hydra overrides
    python src/optimize.py model=random_forest
    python src/optimize.py hpo=random
    python src/optimize.py model=random_forest hpo=random hpo.n_trials=30

    # Compare samplers (run both, then view in MLflow UI)
    python src/optimize.py hpo=optuna
    python src/optimize.py hpo=random
"""

import os
import random

import hydra
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_data(train_path: str, test_path: str):
    train_df = pd.read_csv(to_absolute_path(train_path))
    test_df = pd.read_csv(to_absolute_path(test_path))
    return (
        train_df["clean_tweet"].values,
        test_df["clean_tweet"].values,
        train_df["label"].values,
        test_df["label"].values,
    )


# ──────────────────────────────────────────────
# Search space suggestion
# ──────────────────────────────────────────────
def suggest_params(trial: optuna.Trial, model_type: str, cfg: DictConfig) -> dict:
    """Suggest hyperparameters for TF-IDF + classifier."""
    space = cfg.model.search_space
    params = {}

    # TF-IDF params (included in search space for both models)
    params["max_features"] = trial.suggest_int(
        "max_features", space.max_features.low, space.max_features.high
    )
    params["ngram_max"] = trial.suggest_categorical("ngram_max", list(space.ngram_max))

    if model_type == "logistic_regression":
        params["C"] = trial.suggest_float("C", space.C.low, space.C.high, log=True)
        params["solver"] = trial.suggest_categorical("solver", list(space.solver))

    elif model_type == "random_forest":
        params["n_estimators"] = trial.suggest_int(
            "n_estimators", space.n_estimators.low, space.n_estimators.high
        )
        params["max_depth"] = trial.suggest_int(
            "max_depth", space.max_depth.low, space.max_depth.high
        )
        params["min_samples_split"] = trial.suggest_int(
            "min_samples_split",
            space.min_samples_split.low,
            space.min_samples_split.high,
        )
        params["min_samples_leaf"] = trial.suggest_int(
            "min_samples_leaf",
            space.min_samples_leaf.low,
            space.min_samples_leaf.high,
        )

    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

    return params


# ──────────────────────────────────────────────
# Pipeline construction
# ──────────────────────────────────────────────
def build_pipeline(model_type: str, params: dict, seed: int) -> Pipeline:
    """Build a TF-IDF + classifier sklearn Pipeline from suggested params."""
    vectorizer = TfidfVectorizer(
        max_features=params["max_features"],
        ngram_range=(1, params["ngram_max"]),
        min_df=2,
    )

    if model_type == "logistic_regression":
        clf = LogisticRegression(
            C=params["C"],
            solver=params["solver"],
            max_iter=1000,
            random_state=seed,
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return float(f1_score(y_test, y_pred, average="binary"))


def evaluate_cv(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_splits: int = 5,
) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        m = clone(pipeline)
        m.fit(X[train_idx], y[train_idx])
        scores.append(float(f1_score(y[test_idx], m.predict(X[test_idx]), average="binary")))
    return float(np.mean(scores))


# ──────────────────────────────────────────────
# Sampler factory
# ──────────────────────────────────────────────
def make_sampler(sampler_name: str, seed: int) -> optuna.samplers.BaseSampler:
    name = sampler_name.lower()
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unsupported sampler: '{sampler_name}'. Use 'tpe' or 'random'.")


# ──────────────────────────────────────────────
# Objective factory (closure over data + config)
# ──────────────────────────────────────────────
def objective_factory(
    cfg: DictConfig,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg.model.type, cfg)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", str(trial.number))
            mlflow.set_tag("model_type", cfg.model.type)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("seed", str(cfg.seed))
            mlflow.log_params(params)

            pipeline = build_pipeline(cfg.model.type, params, cfg.seed)

            if cfg.hpo.use_cv:
                X_all = np.concatenate([X_train, X_test])
                y_all = np.concatenate([y_train, y_test])
                score = evaluate_cv(pipeline, X_all, y_all, cfg.seed, cfg.hpo.cv_folds)
            else:
                score = evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test)

            mlflow.log_metric(cfg.hpo.metric, score)

        return score

    return objective


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)

    # Resolve MLflow tracking URI to absolute path (works with hydra.run.dir: .)
    tracking_uri = cfg.mlflow.tracking_uri
    if not tracking_uri.startswith("http"):
        tracking_uri = f"file://{os.path.abspath(to_absolute_path(tracking_uri))}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Load prepared text data
    X_train, X_test, y_train, y_test = load_data(cfg.data.train_path, cfg.data.test_path)
    print(f"Data: train={len(X_train)}, test={len(X_test)}")
    print(f"Model: {cfg.model.type} | Sampler: {cfg.hpo.sampler} | Trials: {cfg.hpo.n_trials}\n")

    sampler = make_sampler(cfg.hpo.sampler, cfg.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    run_name = f"hpo_{cfg.hpo.sampler}_{cfg.model.type}"
    with mlflow.start_run(run_name=run_name) as parent_run:
        # Log study-level metadata
        mlflow.set_tag("model_type", cfg.model.type)
        mlflow.set_tag("sampler", cfg.hpo.sampler)
        mlflow.set_tag("seed", str(cfg.seed))
        mlflow.set_tag("n_trials", str(cfg.hpo.n_trials))
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config_resolved.json")

        # Run Optuna study
        study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)
        objective = objective_factory(cfg, X_train, X_test, y_train, y_test)
        study.optimize(objective, n_trials=cfg.hpo.n_trials, show_progress_bar=True)

        best = study.best_trial
        print(f"\nBest trial:  #{best.number}")
        print(f"Best {cfg.hpo.metric}:  {best.value:.4f}")
        print(f"Best params: {best.params}")

        # Log best results to parent run
        mlflow.log_metric(f"best_{cfg.hpo.metric}", float(best.value))
        mlflow.log_dict(best.params, "best_params.json")

        # Retrain on best params and evaluate
        best_pipeline = build_pipeline(cfg.model.type, best.params, cfg.seed)
        final_score = evaluate_pipeline(best_pipeline, X_train, y_train, X_test, y_test)
        mlflow.log_metric(f"final_{cfg.hpo.metric}", final_score)

        # Save model artifact
        os.makedirs("models", exist_ok=True)
        model_path = "models/best_model.pkl"
        joblib.dump(best_pipeline, model_path)
        mlflow.log_artifact(model_path)

        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        print(f"\nFinal {cfg.hpo.metric} (best params, retrained): {final_score:.4f}")
        print(f"Model saved to: {model_path}")
        print(f"MLflow run:     {parent_run.info.run_id}")
        print(f"View results:   mlflow ui  →  http://127.0.0.1:5000")


# ──────────────────────────────────────────────
# Hydra entry point
# ──────────────────────────────────────────────
@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
