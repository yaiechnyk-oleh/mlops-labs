"""
Lab 5 — DAG Integrity Tests
============================
Verifies that the Airflow DAG file can be parsed and loaded correctly.
Checks: no import errors, expected tasks exist, correct dependencies.

Run locally:
    pip install apache-airflow==2.8.1
    pytest tests/test_dag.py -v
"""

import os
import pytest

# Skip all tests if airflow is not installed (e.g., base ML environment)
airflow = pytest.importorskip("airflow", reason="apache-airflow not installed")

from airflow.models import DagBag  # noqa: E402


DAGS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dags")
DAG_ID = "ml_training_pipeline"


@pytest.fixture(scope="module")
def dag_bag():
    """Load DAGs from the project dags/ folder."""
    bag = DagBag(dag_folder=DAGS_FOLDER, include_examples=False)
    return bag


def test_no_import_errors(dag_bag):
    """DAG files must load without syntax or import errors."""
    assert dag_bag.import_errors == {}, (
        f"DAG import errors: {dag_bag.import_errors}"
    )


def test_dag_exists(dag_bag):
    """The ml_training_pipeline DAG must be present."""
    assert DAG_ID in dag_bag.dags, (
        f"DAG '{DAG_ID}' not found. Available: {list(dag_bag.dags.keys())}"
    )


def test_dag_has_correct_tasks(dag_bag):
    """All required tasks must be present in the DAG."""
    dag = dag_bag.dags[DAG_ID]
    expected_tasks = {
        "check_data",
        "prepare_data",
        "train_model",
        "evaluate_model",
        "register_model",
        "stop_pipeline",
    }
    actual_tasks = set(dag.task_ids)
    missing = expected_tasks - actual_tasks
    assert not missing, f"Missing tasks: {missing}"


def test_dag_task_count(dag_bag):
    """DAG must have exactly 6 tasks."""
    dag = dag_bag.dags[DAG_ID]
    assert len(dag.tasks) == 6, f"Expected 6 tasks, got {len(dag.tasks)}"


def test_dag_dependencies(dag_bag):
    """Verify the core linear dependency chain."""
    dag = dag_bag.dags[DAG_ID]

    def upstream_ids(task_id):
        return {t.task_id for t in dag.get_task(task_id).upstream_list}

    # prepare_data must follow check_data
    assert "check_data" in upstream_ids("prepare_data")
    # train_model must follow prepare_data
    assert "prepare_data" in upstream_ids("train_model")
    # evaluate_model must follow train_model
    assert "train_model" in upstream_ids("evaluate_model")
    # register_model and stop_pipeline must follow evaluate_model
    assert "evaluate_model" in upstream_ids("register_model")
    assert "evaluate_model" in upstream_ids("stop_pipeline")


def test_dag_schedule(dag_bag):
    """DAG must be scheduled weekly."""
    dag = dag_bag.dags[DAG_ID]
    assert dag.schedule_interval == "@weekly", (
        f"Expected '@weekly', got {dag.schedule_interval}"
    )


def test_dag_catchup_disabled(dag_bag):
    """Catchup must be disabled to avoid unintended backfills."""
    dag = dag_bag.dags[DAG_ID]
    assert dag.catchup is False, "catchup must be False"
