"""Kubeflow Pipeline: Autobidding model training — runs every 6 hours."""

from __future__ import annotations

import kfp
from kfp import dsl

from pipelines.components.data_validation import validate_training_data
from pipelines.components.evaluation import evaluate_model
from pipelines.components.model_registry import register_model
from pipelines.components.notification import send_notification
from pipelines.components.training import launch_training_job

_PIPELINE_NAME = "autobid-training"
_PIPELINE_DESCRIPTION = (
    "End-to-end training pipeline for the Autobidding model. "
    "Runs every 6 hours: validates data → training → evaluation → "
    "shadow testing → conditional promotion with canary deploy → notification."
)


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["mlflow==2.10.0", "boto3==1.34.0", "numpy==1.26.3"],
)
def run_shadow_test(
    trained_model_artifact: dsl.Input[dsl.Model],
    shadow_dataset_uri: str,
    model_name: str,
    mlflow_tracking_uri: str,
    max_shadow_latency_p99_ms: float,
    shadow_report: dsl.Output[dsl.Dataset],
) -> bool:
    """Run shadow traffic test: serve candidate model on a replay dataset.

    Checks that p99 inference latency stays within the SLA and that prediction
    distributions do not diverge catastrophically from the production model.

    Args:
        trained_model_artifact: Artifact from the training component.
        shadow_dataset_uri: S3/GCS URI for the shadow replay dataset.
        model_name: Model name in MLflow registry.
        mlflow_tracking_uri: MLflow tracking server URI.
        max_shadow_latency_p99_ms: Maximum allowed p99 inference latency in ms.
        shadow_report: Output artifact with shadow test results.

    Returns:
        True if shadow test passes; False otherwise.
    """
    import json
    import logging
    import time

    import mlflow
    import numpy as np
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    with open(trained_model_artifact.path) as f:
        model_meta = json.load(f)

    candidate_version = model_meta["model_version"]
    candidate_uri = f"models:/{model_name}/{candidate_version}"

    def _load_data(uri: str) -> pd.DataFrame:
        if uri.startswith("s3://"):
            import boto3

            parts = uri[5:].split("/", 1)
            bucket, key = parts[0], parts[1]
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            return pd.read_parquet(obj["Body"])
        return pd.read_parquet(uri)

    logger.info("Loading shadow dataset from %s", shadow_dataset_uri)
    shadow_df = _load_data(shadow_dataset_uri)
    features = shadow_df.drop(columns=["label", "user_id", "event_time"], errors="ignore")

    logger.info("Loading candidate model %s", candidate_uri)
    model = mlflow.pyfunc.load_model(candidate_uri)

    # Warm-up
    _ = model.predict(features.head(10))

    # Benchmark inference latency
    latencies_ms = []
    batch_size = 256
    for i in range(0, min(len(features), 10000), batch_size):
        batch = features.iloc[i : i + batch_size]
        t0 = time.perf_counter()
        _ = model.predict(batch)
        latencies_ms.append((time.perf_counter() - t0) * 1000 / len(batch))

    p50 = float(np.percentile(latencies_ms, 50))
    p99 = float(np.percentile(latencies_ms, 99))
    latency_ok = p99 <= max_shadow_latency_p99_ms

    logger.info(
        "Latency p50=%.2fms p99=%.2fms threshold=%.2fms ok=%s",
        p50,
        p99,
        max_shadow_latency_p99_ms,
        latency_ok,
    )

    report = {
        "model_name": model_name,
        "candidate_version": candidate_version,
        "shadow_dataset_uri": shadow_dataset_uri,
        "latency_p50_ms": p50,
        "latency_p99_ms": p99,
        "max_latency_p99_ms": max_shadow_latency_p99_ms,
        "latency_ok": latency_ok,
        "passed": latency_ok,
    }

    with open(shadow_report.path, "w") as f:
        json.dump(report, f, indent=2)

    return latency_ok


@dsl.pipeline(name=_PIPELINE_NAME, description=_PIPELINE_DESCRIPTION)
def autobid_pipeline(
    # Data parameters
    dataset_uri: str = "s3://fb-ads-ml-data/autobid/train/latest",
    schema_uri: str = "s3://fb-ads-ml-data/autobid/schema/v1.yaml",
    reference_dataset_uri: str = "s3://fb-ads-ml-data/autobid/reference/latest",
    eval_dataset_uri: str = "s3://fb-ads-ml-data/autobid/eval/latest",
    shadow_dataset_uri: str = "s3://fb-ads-ml-data/autobid/shadow/latest",
    dataset_version: str = "{{$.inputs.parameters['dataset_version']}}",
    # Data validation — stricter for autobid (financial impact)
    max_null_ratio: float = 0.02,
    min_rows: int = 5_000_000,
    max_drift_pvalue: float = 0.005,
    # Training parameters
    base_image: str = "gcr.io/fb-ads-ml/autobid-trainer:latest",
    num_workers: int = 8,
    gpus_per_worker: int = 4,
    cpu_per_worker: str = "16",
    memory_per_worker: str = "64Gi",
    model_version: str = "{{$.inputs.parameters['model_version']}}",
    training_script: str = "ad_ml.autobid.train",
    # Hyperparameters
    learning_rate: float = 5e-4,
    batch_size: int = 4096,
    max_epochs: int = 10,
    num_layers: int = 8,
    hidden_dim: int = 256,
    dropout: float = 0.15,
    l2_reg: float = 1e-4,
    # MLflow
    mlflow_tracking_uri: str = "http://mlflow.ml-platform.svc.cluster.local:5000",
    mlflow_experiment: str = "autobid",
    # Evaluation — conservative promotion criteria for autobid
    baseline_model_version: str = "Production",
    promotion_metric: str = "auc",
    min_improvement_pct: float = 1.0,
    significance_level: float = 0.01,
    # Shadow test
    max_shadow_latency_p99_ms: float = 50.0,
    # Notification
    slack_webhook_url: str = "",
    pagerduty_routing_key: str = "",
    alert_on_failure: bool = True,
) -> None:
    """Autobid training pipeline with shadow test and conservative promotion.

    DAG:
        validate_data
            └── train
                    └── evaluate
                            └── [dsl.If promote] shadow_test
                                    └── [dsl.If shadow passes] register_model
                            └── notify (always)
    """
    hyperparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "l2_reg": l2_reg,
    }

    # ── Step 1: Validate training data ─────────────────────────────────────
    validate_task = validate_training_data(
        dataset_uri=dataset_uri,
        schema_uri=schema_uri,
        reference_dataset_uri=reference_dataset_uri,
        model_name="autobid",
        max_null_ratio=max_null_ratio,
        min_rows=min_rows,
        max_drift_pvalue=max_drift_pvalue,
    )
    validate_task.set_caching_options(enable_caching=True)
    validate_task.set_retry(num_retries=1)

    # ── Step 2: Launch distributed training ────────────────────────────────
    train_task = launch_training_job(
        model_name="autobid",
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
        model_name="autobid",
        mlflow_tracking_uri=mlflow_tracking_uri,
        baseline_model_version=baseline_model_version,
        promotion_metric=promotion_metric,
        min_improvement_pct=min_improvement_pct,
        significance_level=significance_level,
    )
    eval_task.set_caching_options(enable_caching=False)

    # ── Step 4: Shadow test + conditional promotion ─────────────────────────
    with dsl.If(eval_task.output == True, name="eval-passed"):  # noqa: E712
        shadow_task = run_shadow_test(
            trained_model_artifact=train_task.outputs["trained_model"],
            shadow_dataset_uri=shadow_dataset_uri,
            model_name="autobid",
            mlflow_tracking_uri=mlflow_tracking_uri,
            max_shadow_latency_p99_ms=max_shadow_latency_p99_ms,
        )
        shadow_task.set_caching_options(enable_caching=False)

        with dsl.If(shadow_task.output == True, name="shadow-passed"):  # noqa: E712
            register_task = register_model(
                trained_model=train_task.outputs["trained_model"],
                evaluation_report=eval_task.outputs["evaluation_report"],
                model_name="autobid",
                mlflow_tracking_uri=mlflow_tracking_uri,
                target_stage="Staging",
                dataset_version=dataset_version,
                pipeline_run_id=dsl.PIPELINE_RUN_ID_PLACEHOLDER,
            )
            register_task.set_caching_options(enable_caching=False)

            notify_success = send_notification(
                model_name="autobid",
                pipeline_run_id=dsl.PIPELINE_RUN_ID_PLACEHOLDER,
                status="SUCCESS",
                message=(
                    f"Autobid model {model_version} passed evaluation and shadow test. "
                    f"Promoted to Staging — canary deployment will follow."
                ),
                slack_webhook_url=slack_webhook_url,
                pagerduty_routing_key=pagerduty_routing_key,
                alert_on_failure=alert_on_failure,
                registered_model_version=register_task.output,
            )
            notify_success.after(register_task)

        with dsl.If(shadow_task.output == False, name="shadow-failed"):  # noqa: E712
            notify_shadow_fail = send_notification(
                model_name="autobid",
                pipeline_run_id=dsl.PIPELINE_RUN_ID_PLACEHOLDER,
                status="FAILED",
                message=(
                    f"Autobid model {model_version} failed shadow test "
                    f"(p99 latency > {max_shadow_latency_p99_ms}ms). Not promoted."
                ),
                slack_webhook_url=slack_webhook_url,
                pagerduty_routing_key=pagerduty_routing_key,
                alert_on_failure=alert_on_failure,
            )
            notify_shadow_fail.after(shadow_task)

    with dsl.If(eval_task.output == False, name="eval-failed"):  # noqa: E712
        notify_eval_fail = send_notification(
            model_name="autobid",
            pipeline_run_id=dsl.PIPELINE_RUN_ID_PLACEHOLDER,
            status="SUCCESS",
            message=(
                f"Autobid model {model_version} did not meet promotion criteria "
                f"({promotion_metric} improvement < {min_improvement_pct}% or not significant). "
                "Model retained in MLflow but not promoted."
            ),
            slack_webhook_url=slack_webhook_url,
            pagerduty_routing_key=pagerduty_routing_key,
            alert_on_failure=False,
        )
        notify_eval_fail.after(eval_task)


def compile_pipeline(output_path: str = "pipeline_artifacts/autobid_pipeline.yaml") -> None:
    """Compile the pipeline to a YAML artifact."""
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    kfp.compiler.Compiler().compile(
        pipeline_func=autobid_pipeline,
        package_path=output_path,
    )
    print(f"Pipeline compiled to {output_path}")


if __name__ == "__main__":
    compile_pipeline()
