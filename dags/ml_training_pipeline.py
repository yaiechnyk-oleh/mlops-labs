"""
MLOps Lab 5 — Airflow DAG: ML Training Pipeline
================================================
Orchestrates the full ML lifecycle:
  1. check_data    — FileSensor: verify train.csv exists
  2. prepare_data  — BashOperator: run DVC prepare stage
  3. train_model   — BashOperator: run model training
  4. evaluate_model — BranchPythonOperator: route by F1 threshold
  5a. register_model — PythonOperator: register in MLflow Model Registry
  5b. stop_pipeline  — BashOperator: log quality-gate failure

Schedule: weekly (can be triggered manually via Airflow UI).
"""

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.filesystem import FileSensor

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.getenv("MLOPS_PROJECT_DIR", "/opt/airflow/project")
DATA_DIR = os.path.join(PROJECT_DIR, "data", "prepared")
MODELS_DIR = os.path.join(PROJECT_DIR, "data", "models")
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_DIR}/mlflow.db"
REGISTERED_MODEL_NAME = "TwitterSentimentModel"
F1_THRESHOLD = 0.45  # Quality Gate: minimum acceptable F1 on test set

# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    "owner": "student",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="Automated ML pipeline: prepare → train → evaluate → register",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "lab5", "sentiment"],
) as dag:

    # ── Step 1: Sensor — check data availability ──────────────────────────────
    check_data = FileSensor(
        task_id="check_data",
        filepath=os.path.join(DATA_DIR, "train.csv"),
        poke_interval=30,
        timeout=300,
        mode="poke",
        soft_fail=False,
    )

    # ── Step 2: Prepare data (DVC stage) ──────────────────────────────────────
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/prepare.py data/raw/twitter.csv data/prepared"
        ),
    )

    # ── Step 3: Train model ───────────────────────────────────────────────────
    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/train.py data/prepared data/models"
        ),
    )

    # ── Step 4: Evaluate & branch ─────────────────────────────────────────────
    def evaluate_and_branch(**kwargs):
        """Read metrics.json and decide whether to register the model."""
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        f1 = float(metrics.get("f1", 0))
        print(f"[Quality Gate] F1={f1:.4f} | Threshold={F1_THRESHOLD}")

        # Push metrics to XCom so downstream tasks can read them
        kwargs["ti"].xcom_push(key="metrics", value=metrics)

        if f1 >= F1_THRESHOLD:
            print("Quality Gate PASSED → registering model.")
            return "register_model"
        print("Quality Gate FAILED → stopping pipeline.")
        return "stop_pipeline"

    evaluate_model = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_and_branch,
    )

    # ── Step 5a: Register model in MLflow Model Registry ──────────────────────
    def register_model_fn(**kwargs):
        """Load best model artifact and register it in MLflow Model Registry."""
        import joblib
        import mlflow
        import mlflow.sklearn

        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="evaluate_model", key="metrics") or {}

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("ML_Pipeline_Lab5")

        model_pkl = os.path.join(MODELS_DIR, "model.pkl")
        vectorizer, model = joblib.load(model_pkl)

        with mlflow.start_run(run_name="airflow_registration"):
            # Log params & metrics
            mlflow.log_param("pipeline", "airflow")
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            # Log & register model
            mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=REGISTERED_MODEL_NAME,
            )

            run_id = mlflow.active_run().info.run_id
            print(f"Model registered: {REGISTERED_MODEL_NAME} | Run: {run_id}")

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=register_model_fn,
    )

    # ── Step 5b: Stop — quality gate failed ───────────────────────────────────
    stop_pipeline = BashOperator(
        task_id="stop_pipeline",
        bash_command=(
            f"echo 'Quality Gate FAILED: F1 below {F1_THRESHOLD}. "
            "Model NOT registered.' && exit 0"
        ),
    )

    # ── Dependencies ──────────────────────────────────────────────────────────
    check_data >> prepare_data >> train_model >> evaluate_model
    evaluate_model >> [register_model, stop_pipeline]
