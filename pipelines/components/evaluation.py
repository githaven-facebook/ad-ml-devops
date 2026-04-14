"""KFP component: evaluate a trained model against the production baseline."""

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "mlflow==2.10.0",
        "boto3==1.34.0",
        "scipy==1.12.0",
        "numpy==1.26.3",
        "pandas==2.2.0",
        "scikit-learn==1.4.0",
    ],
)
def evaluate_model(
    trained_model: Input[Model],
    eval_dataset_uri: str,
    model_name: str,
    mlflow_tracking_uri: str,
    baseline_model_version: str,
    promotion_metric: str,
    min_improvement_pct: float,
    significance_level: float,
    eval_metrics: Output[Metrics],
    evaluation_report: Output[Dataset],
) -> bool:
    """Evaluate trained model against baseline using holdout eval dataset.

    Computes primary metric (e.g. AUC, NDCG) for both candidate and baseline,
    runs a paired t-test or bootstrap significance test, and returns True if
    the candidate is statistically significantly better.

    Args:
        trained_model: Artifact from the training component containing run ID.
        eval_dataset_uri: S3/GCS URI for the holdout evaluation dataset.
        model_name: Model name for MLflow registry lookups.
        mlflow_tracking_uri: MLflow tracking server URI.
        baseline_model_version: Version string of the current production model.
        promotion_metric: Primary metric name (e.g. "auc", "ndcg_at_10").
        min_improvement_pct: Minimum relative improvement over baseline to promote.
        significance_level: p-value threshold for statistical significance test.
        eval_metrics: Output KFP Metrics artifact with primary metric values.
        evaluation_report: Output JSON artifact with full evaluation report.

    Returns:
        True if candidate meets promotion criteria; False otherwise.
    """
    import json
    import logging

    import mlflow
    import numpy as np
    import pandas as pd
    from scipy import stats

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    with open(trained_model.path) as f:
        model_meta = json.load(f)

    _candidate_run_id = model_meta["mlflow_run_id"]
    candidate_version = model_meta["model_version"]

    logger.info("Loading eval dataset from %s", eval_dataset_uri)

    def _load_parquet(uri: str) -> pd.DataFrame:
        if uri.startswith("s3://"):
            import boto3
            parts = uri[5:].split("/", 1)
            bucket, key = parts[0], parts[1]
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            return pd.read_parquet(obj["Body"])
        return pd.read_parquet(uri)

    eval_df = _load_parquet(eval_dataset_uri)
    labels = eval_df["label"].values
    features = eval_df.drop(columns=["label", "user_id", "event_time"], errors="ignore")

    def _run_inference(model_uri: str, features: pd.DataFrame) -> np.ndarray:
        logger.info("Loading model from %s for inference", model_uri)
        model = mlflow.pyfunc.load_model(model_uri)
        preds = model.predict(features)
        if hasattr(preds, "values"):
            return preds.values
        return np.array(preds)

    candidate_uri = f"models:/{model_name}/{candidate_version}"
    candidate_preds = _run_inference(candidate_uri, features)

    baseline_uri = f"models:/{model_name}/{baseline_model_version}"
    baseline_preds = _run_inference(baseline_uri, features)

    def _compute_metric(metric: str, preds: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
        """Return scalar metric value and per-sample scores for significance testing."""
        if metric == "auc":
            from sklearn.metrics import roc_auc_score
            # Compute per-bootstrap AUC via jackknife
            n = len(labels)
            indices = np.arange(n)
            rng = np.random.default_rng(42)
            bootstrap_scores = []
            for _ in range(200):
                idx = rng.choice(indices, size=n, replace=True)
                try:
                    score = roc_auc_score(labels[idx], preds[idx])
                    bootstrap_scores.append(score)
                except ValueError:
                    pass
            return float(roc_auc_score(labels, preds)), np.array(bootstrap_scores)

        if metric == "ndcg_at_10":
            from sklearn.metrics import ndcg_score
            overall = float(ndcg_score(labels.reshape(1, -1), preds.reshape(1, -1), k=10))
            rng = np.random.default_rng(42)
            n = len(labels)
            bootstrap_scores = []
            for _ in range(200):
                idx = rng.choice(n, size=n, replace=True)
                try:
                    score = ndcg_score(labels[idx].reshape(1, -1), preds[idx].reshape(1, -1), k=10)
                    bootstrap_scores.append(score)
                except ValueError:
                    pass
            return overall, np.array(bootstrap_scores)

        if metric in ("mse", "rmse"):
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(labels, preds)
            return float(np.sqrt(mse) if metric == "rmse" else mse), preds - labels

        raise ValueError(f"Unsupported promotion_metric: {metric}")

    candidate_score, candidate_bootstrap = _compute_metric(promotion_metric, candidate_preds, labels)
    baseline_score, baseline_bootstrap = _compute_metric(promotion_metric, baseline_preds, labels)

    # Statistical significance test (two-sample t-test on bootstrap distributions)
    t_stat, p_value = stats.ttest_ind(candidate_bootstrap, baseline_bootstrap)
    is_significant = p_value < significance_level

    # For metrics where higher is better (auc, ndcg); lower is better (mse, rmse)
    higher_is_better = promotion_metric not in ("mse", "rmse")
    if higher_is_better:
        relative_improvement = (candidate_score - baseline_score) / max(abs(baseline_score), 1e-9)
    else:
        relative_improvement = (baseline_score - candidate_score) / max(abs(baseline_score), 1e-9)

    meets_improvement = relative_improvement >= (min_improvement_pct / 100.0)
    should_promote = meets_improvement and is_significant

    report = {
        "model_name": model_name,
        "candidate_version": candidate_version,
        "baseline_version": baseline_model_version,
        "promotion_metric": promotion_metric,
        "candidate_score": candidate_score,
        "baseline_score": baseline_score,
        "relative_improvement_pct": relative_improvement * 100,
        "min_improvement_pct": min_improvement_pct,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significance_level": significance_level,
        "is_significant": is_significant,
        "meets_improvement_threshold": meets_improvement,
        "should_promote": should_promote,
    }

    logger.info(
        "Evaluation: candidate=%.4f baseline=%.4f improvement=%.2f%% p=%.4f promote=%s",
        candidate_score,
        baseline_score,
        relative_improvement * 100,
        p_value,
        should_promote,
    )

    eval_metrics.log_metric(f"candidate_{promotion_metric}", candidate_score)
    eval_metrics.log_metric(f"baseline_{promotion_metric}", baseline_score)
    eval_metrics.log_metric("relative_improvement_pct", relative_improvement * 100)
    eval_metrics.log_metric("p_value", float(p_value))
    eval_metrics.log_metric("should_promote", float(should_promote))

    with open(evaluation_report.path, "w") as f:
        json.dump(report, f, indent=2)

    return should_promote
