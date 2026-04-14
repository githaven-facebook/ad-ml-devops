"""KFP component: launch a distributed PyTorch training job on Kubernetes."""

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "kubernetes==29.0.0",
        "mlflow==2.10.0",
        "pyyaml==6.0.1",
        "boto3==1.34.0",
    ],
)
def launch_training_job(
    model_name: str,
    model_version: str,
    dataset_uri: str,
    validation_report: Input[Dataset],
    base_image: str,
    num_workers: int,
    gpus_per_worker: int,
    cpu_per_worker: str,
    memory_per_worker: str,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    training_script: str,
    hyperparams: dict,
    trained_model: Output[Model],
) -> str:
    """Launch a distributed PyTorch DDP training job via a Kubernetes PyTorchJob.

    Creates a PyTorchJob CRD (kubeflow/training-operator) with the specified
    resource requests and waits for completion. Streams logs and writes the
    MLflow run ID to the output artifact.

    Args:
        model_name: Name of the model (user-persona or autobid).
        model_version: Semantic version string for this training run.
        dataset_uri: S3/GCS URI for the training dataset.
        validation_report: Upstream validation report artifact (consumed for lineage).
        base_image: Docker image containing the training code.
        num_workers: Number of DDP worker replicas.
        gpus_per_worker: GPU count per worker (e.g. 1 or 4).
        cpu_per_worker: CPU request per worker (e.g. "8").
        memory_per_worker: Memory request per worker (e.g. "32Gi").
        mlflow_tracking_uri: MLflow server URI.
        mlflow_experiment: MLflow experiment name.
        training_script: Entry-point script path inside the container.
        hyperparams: Dict of hyperparameter key-value pairs.
        trained_model: Output artifact containing MLflow run ID and model URI.

    Returns:
        MLflow run ID for the completed training run.
    """
    import json
    import logging
    import time
    import uuid

    from kubernetes import client, config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()

    run_id = str(uuid.uuid4())[:8]
    job_name = f"{model_name}-train-{model_version.replace('.', '-')}-{run_id}"
    namespace = "ml-platform"

    env_vars = [
        {"name": "MLFLOW_TRACKING_URI", "value": mlflow_tracking_uri},
        {"name": "MLFLOW_EXPERIMENT_NAME", "value": mlflow_experiment},
        {"name": "DATASET_URI", "value": dataset_uri},
        {"name": "MODEL_NAME", "value": model_name},
        {"name": "MODEL_VERSION", "value": model_version},
        {"name": "HYPERPARAMS", "value": json.dumps(hyperparams)},
        {"name": "MASTER_PORT", "value": "29500"},
    ]

    command = [
        "python",
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={gpus_per_worker}",
        "--nnodes=$(NUM_WORKERS)",
        "--node_rank=$(RANK)",
        "--master_addr=$(MASTER_ADDR)",
        "--master_port=29500",
        training_script,
    ]

    worker_spec: dict = {
        "replicas": num_workers,
        "restartPolicy": "OnFailure",
        "template": {
            "metadata": {
                "labels": {
                    "app.kubernetes.io/component": "training",
                    "app.kubernetes.io/name": model_name,
                    "training-run": job_name,
                },
                "annotations": {"prometheus.io/scrape": "false"},
            },
            "spec": {
                "runtimeClassName": "nvidia",
                "tolerations": [
                    {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
                ],
                "nodeSelector": {"cloud.google.com/gke-accelerator": "nvidia-tesla-a100"},
                "securityContext": {"runAsNonRoot": True, "runAsUser": 1000, "fsGroup": 1000},
                "containers": [
                    {
                        "name": "trainer",
                        "image": base_image,
                        "command": command,
                        "env": env_vars,
                        "resources": {
                            "requests": {
                                "cpu": cpu_per_worker,
                                "memory": memory_per_worker,
                                "nvidia.com/gpu": str(gpus_per_worker),
                            },
                            "limits": {
                                "cpu": cpu_per_worker,
                                "memory": memory_per_worker,
                                "nvidia.com/gpu": str(gpus_per_worker),
                            },
                        },
                        "volumeMounts": [
                            {"name": "shm", "mountPath": "/dev/shm"},
                            {"name": "model-storage", "mountPath": "/mnt/models"},
                        ],
                    }
                ],
                "volumes": [
                    {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
                    {"name": "model-storage", "emptyDir": {}},
                ],
            },
        },
    }

    pytorchjob = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/managed-by": "kfp",
                "model-name": model_name,
                "model-version": model_version,
            },
        },
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {**worker_spec, "replicas": 1},
                "Worker": {**worker_spec, "replicas": max(0, num_workers - 1)},
            }
        },
    }

    logger.info(
        "Creating PyTorchJob %s with %d workers, %d GPU each",
        job_name,
        num_workers,
        gpus_per_worker,
    )
    custom_api.create_namespaced_custom_object(
        group="kubeflow.org",
        version="v1",
        namespace=namespace,
        plural="pytorchjobs",
        body=pytorchjob,
    )

    # Wait for job completion
    timeout_seconds = 24 * 3600  # 24h max
    start_time = time.time()
    mlflow_run_id = ""

    while time.time() - start_time < timeout_seconds:
        job_status = custom_api.get_namespaced_custom_object(
            group="kubeflow.org",
            version="v1",
            namespace=namespace,
            plural="pytorchjobs",
            name=job_name,
        )
        conditions = job_status.get("status", {}).get("conditions", [])
        phase = next(
            (c["type"] for c in reversed(conditions) if c.get("status") == "True"), "Running"
        )

        logger.info(
            "PyTorchJob %s phase: %s (elapsed %.0fs)", job_name, phase, time.time() - start_time
        )

        if phase == "Succeeded":
            # Retrieve MLflow run ID from job annotation set by training code
            mlflow_run_id = (
                job_status.get("metadata", {}).get("annotations", {}).get("mlflow/run-id", "")
            )
            logger.info("Training completed. MLflow run ID: %s", mlflow_run_id)
            break
        if phase in ("Failed", "Suspended"):
            raise RuntimeError(f"PyTorchJob {job_name} entered phase {phase}")

        time.sleep(60)
    else:
        raise TimeoutError(f"PyTorchJob {job_name} did not complete within {timeout_seconds}s")

    model_uri = f"models:/{model_name}/{model_version}"
    artifact_data = {
        "job_name": job_name,
        "mlflow_run_id": mlflow_run_id,
        "model_uri": model_uri,
        "model_name": model_name,
        "model_version": model_version,
        "num_workers": num_workers,
        "gpus_per_worker": gpus_per_worker,
    }
    with open(trained_model.path, "w") as f:
        json.dump(artifact_data, f, indent=2)

    trained_model.metadata["mlflow_run_id"] = mlflow_run_id
    trained_model.metadata["model_uri"] = model_uri

    return mlflow_run_id
