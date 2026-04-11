"""KFP component: register a trained model in MLflow Model Registry."""

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "mlflow==2.10.0",
        "boto3==1.34.0",
    ],
)
def register_model(
    trained_model: Input[Model],
    evaluation_report: Input[Dataset],
    model_name: str,
    mlflow_tracking_uri: str,
    target_stage: str,
    dataset_version: str,
    pipeline_run_id: str,
    registered_model_info: Output[Dataset],
) -> str:
    """Register the trained model in MLflow Model Registry with full metadata.

    Tags the registered model version with training data version, evaluation
    metrics, pipeline run ID, and promotes it to the requested stage
    (Staging or Production).

    Args:
        trained_model: Artifact from the training component.
        evaluation_report: Artifact from the evaluation component.
        model_name: MLflow registered model name.
        mlflow_tracking_uri: MLflow tracking server URI.
        target_stage: Registry stage to promote to ("Staging" or "Production").
        dataset_version: Version identifier of the training dataset.
        pipeline_run_id: Kubeflow pipeline run ID for traceability.
        registered_model_info: Output artifact with registration details.

    Returns:
        Registered model version string (e.g. "42").
    """
    import json
    import logging
    from datetime import datetime, timezone

    import mlflow
    from mlflow.tracking import MlflowClient

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    with open(trained_model.path) as f:
        model_meta = json.load(f)

    with open(evaluation_report.path) as f:
        eval_report = json.load(f)

    run_id = model_meta["mlflow_run_id"]
    candidate_version = model_meta["model_version"]
    model_uri = f"runs:/{run_id}/model"

    logger.info("Registering model %s from run %s", model_name, run_id)

    # Ensure the registered model exists
    try:
        client.create_registered_model(
            name=model_name,
            description=f"Ad ML model: {model_name}",
            tags={
                "team": "ml-platform",
                "service": "ad-ml-serving",
            },
        )
        logger.info("Created new registered model: %s", model_name)
    except mlflow.exceptions.MlflowException as exc:
        if "already exists" not in str(exc).lower():
            raise
        logger.info("Registered model %s already exists", model_name)

    # Create model version
    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        description=f"Trained version {candidate_version} — pipeline run {pipeline_run_id}",
        tags={
            "candidate_version": candidate_version,
            "dataset_version": dataset_version,
            "pipeline_run_id": pipeline_run_id,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            f"eval.{eval_report['promotion_metric']}": str(eval_report["candidate_score"]),
            "eval.relative_improvement_pct": str(eval_report["relative_improvement_pct"]),
            "eval.p_value": str(eval_report["p_value"]),
            "eval.baseline_version": eval_report["baseline_version"],
        },
    )
    registered_version = mv.version
    logger.info("Created model version %s for %s", registered_version, model_name)

    # Transition to target stage
    client.transition_model_version_stage(
        name=model_name,
        version=registered_version,
        stage=target_stage,
        archive_existing_versions=(target_stage == "Production"),
    )
    logger.info("Transitioned %s v%s to %s", model_name, registered_version, target_stage)

    # Set run tags for reverse lookup
    client.set_tag(run_id, "registered_model_version", registered_version)
    client.set_tag(run_id, "registry_stage", target_stage)

    result = {
        "model_name": model_name,
        "registered_version": registered_version,
        "stage": target_stage,
        "mlflow_run_id": run_id,
        "dataset_version": dataset_version,
        "pipeline_run_id": pipeline_run_id,
        "model_uri": f"models:/{model_name}/{registered_version}",
    }

    with open(registered_model_info.path, "w") as f:
        json.dump(result, f, indent=2)

    return registered_version
