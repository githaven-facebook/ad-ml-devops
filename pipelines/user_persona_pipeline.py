"""Kubeflow Pipeline: User Persona model training — daily scheduled run."""

from __future__ import annotations

import os

import kfp
from kfp import dsl

from pipelines.components.data_validation import validate_training_data
from pipelines.components.evaluation import evaluate_model
from pipelines.components.model_registry import register_model
from pipelines.components.notification import send_notification
from pipelines.components.training import launch_training_job

_PIPELINE_NAME = "user-persona-training"
_PIPELINE_DESCRIPTION = (
    "End-to-end training pipeline for the User Persona model. "
    "Runs daily: validates data → distributed GPU training → evaluation → "
    "conditional promotion → registry → Slack/PagerDuty notification."
)


@dsl.pipeline(name=_PIPELINE_NAME, description=_PIPELINE_DESCRIPTION)
def user_persona_pipeline(
    # Data parameters
    dataset_uri: str = "s3://fb-ads-ml-data/user-persona/train/latest",
    schema_uri: str = "s3://fb-ads-ml-data/user-persona/schema/v1.yaml",
    reference_dataset_uri: str = "s3://fb-ads-ml-data/user-persona/reference/latest",
    eval_dataset_uri: str = "s3://fb-ads-ml-data/user-persona/eval/latest",
    dataset_version: str = "{{$.inputs.parameters['dataset_version']}}",
    # Data validation parameters
    max_null_ratio: float = 0.05,
    min_rows: int = 1_000_000,
    max_drift_pvalue: float = 0.01,
    # Training parameters
    base_image: str = "gcr.io/fb-ads-ml/user-persona-trainer:latest",
    num_workers: int = 4,
    gpus_per_worker: int = 4,
    cpu_per_worker: str = "16",
    memory_per_worker: str = "64Gi",
    model_version: str = "{{$.inputs.parameters['model_version']}}",
    training_script: str = "ad_ml.user_persona.train",
    # Hyperparameters
    learning_rate: float = 1e-3,
    batch_size: int = 2048,
    max_epochs: int = 20,
    embedding_dim: int = 128,
    num_layers: int = 6,
    dropout: float = 0.1,
    # MLflow
    mlflow_tracking_uri: str = "http://mlflow.ml-platform.svc.cluster.local:5000",
    mlflow_experiment: str = "user-persona",
    # Evaluation parameters
    baseline_model_version: str = "Production",
    promotion_metric: str = "auc",
    min_improvement_pct: float = 0.5,
    significance_level: float = 0.05,
    # Notification
    slack_webhook_url: str = "",
    pagerduty_routing_key: str = "",
    alert_on_failure: bool = True,
) -> None:
    """User Persona training pipeline with conditional promotion.

    DAG:
        validate_data
            └── train (if validation passes)
                    └── evaluate
                            └── [dsl.If promote] register_model
                            └── notify (always)
    """
    hyperparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "dropout": dropout,
    }

    # ── Step 1: Validate training data ─────────────────────────────────────
    validate_task = validate_training_data(
        dataset_uri=dataset_uri,
        schema_uri=schema_uri,
        reference_dataset_uri=reference_dataset_uri,
        model_name="user-persona",
        max_null_ratio=max_null_ratio,
        min_rows=min_rows,
        max_drift_pvalue=max_drift_pvalue,
    )
    validate_task.set_caching_options(enable_caching=True)
    validate_task.set_retry(num_retries=1)

    # ── Step 2: Launch distributed training ────────────────────────────────
    train_task = launch_training_job(
        model_name="user-persona",
        model_version=model_version,
        dataset_uri=dataset_uri,
        validation_report=validate_task.outputs["validation_report"],
        base_image=base_image,
        num_workers=num_workers,
        gpus_per_worker=gpus_per_worker,
        cpu_per_worker=cpu_per_worker,
        memory_per_worker=memory_per_worker,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        training_script=training_script,
        hyperparams=hyperparams,
    )
    train_task.set_caching_options(enable_caching=False)
    train_task.set_retry(num_retries=0)
    train_task.after(validate_task)

    # ── Step 3: Evaluate candidate vs baseline ─────────────────────────────
    eval_task = evaluate_model(
        trained_model=train_task.outputs["trained_model"],
        eval_dataset_uri=eval_dataset_uri,
        model_name="user-persona",
        mlflow_tracking_uri=mlflow_tracking_uri,
        baseline_model_version=baseline_model_version,
        promotion_metric=promotion_metric,
        min_improvement_pct=min_improvement_pct,
        significance_level=significance_level,
    )
    eval_task.set_caching_options(enable_caching=False)

    # ── Step 4: Conditional promotion ──────────────────────────────────────
    with dsl.If(eval_task.output == True, name="should-promote"):  # noqa: E712
        register_task = register_model(
            trained_model=train_task.outputs["trained_model"],
            evaluation_report=eval_task.outputs["evaluation_report"],
            model_name="user-persona",
            mlflow_tracking_uri=mlflow_tracking_uri,
            target_stage="Staging",
            dataset_version=dataset_version,
            pipeline_run_id=dsl.PIPELINE_RUN_ID_PLACEHOLDER,
        )
        register_task.set_caching_options(enable_caching=False)

        # ── Step 5a: Notify success with version ───────────────────────────
        notify_success = send_notification(
            model_name="user-persona",
            pipeline_run_id=dsl.PIPELINE_RUN_ID_PLACEHOLDER,
            status="SUCCESS",
            message=(
                f"User Persona model {model_version} promoted to Staging. "
                f"Metric: {promotion_metric} improved by >= {min_improvement_pct}%."
            ),
            slack_webhook_url=slack_webhook_url,
            pagerduty_routing_key=pagerduty_routing_key,
            alert_on_failure=alert_on_failure,
            registered_model_version=register_task.output,
        )
        notify_success.after(register_task)

    with dsl.If(eval_task.output == False, name="should-not-promote"):  # noqa: E712
        # ── Step 5b: Notify no-promotion (not a failure) ───────────────────
        notify_no_promote = send_notification(
            model_name="user-persona",
            pipeline_run_id=dsl.PIPELINE_RUN_ID_PLACEHOLDER,
            status="SUCCESS",
            message=(
                f"User Persona model {model_version} trained but did not meet "
                f"promotion criteria ({promotion_metric} improvement < {min_improvement_pct}% "
                f"or not statistically significant). Model retained in MLflow but not promoted."
            ),
            slack_webhook_url=slack_webhook_url,
            pagerduty_routing_key=pagerduty_routing_key,
            alert_on_failure=False,
        )
        notify_no_promote.after(eval_task)


def compile_pipeline(output_path: str = "pipeline_artifacts/user_persona_pipeline.yaml") -> None:
    """Compile the pipeline to a YAML artifact."""
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    kfp.compiler.Compiler().compile(
        pipeline_func=user_persona_pipeline,
        package_path=output_path,
    )
    print(f"Pipeline compiled to {output_path}")


if __name__ == "__main__":
    compile_pipeline()
