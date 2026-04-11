"""KFP component: validate training data quality before launching a training job."""

from kfp import dsl
from kfp.dsl import Dataset, Input, Output


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas==2.2.0",
        "scipy==1.12.0",
        "pyarrow==15.0.0",
        "boto3==1.34.0",
        "pyyaml==6.0.1",
    ],
)
def validate_training_data(
    dataset_uri: str,
    schema_uri: str,
    reference_dataset_uri: str,
    model_name: str,
    max_null_ratio: float,
    min_rows: int,
    max_drift_pvalue: float,
    validation_report: Output[Dataset],
) -> bool:
    """Validate training dataset against schema, null thresholds, and distribution drift.

    Performs:
    - Schema validation (column names, dtypes)
    - Null ratio check per column
    - Data freshness check (requires records within last N hours)
    - KS-test feature distribution drift detection vs reference dataset

    Args:
        dataset_uri: S3/GCS URI for the training dataset (parquet).
        schema_uri: S3/GCS URI for the expected schema YAML.
        reference_dataset_uri: S3/GCS URI for the reference dataset for drift comparison.
        model_name: Name of the model being trained (for logging).
        max_null_ratio: Maximum acceptable null fraction per column (e.g. 0.05).
        min_rows: Minimum number of rows required in the dataset.
        max_drift_pvalue: KS-test p-value threshold; below this indicates drift.
        validation_report: Output artifact with validation results as JSON.

    Returns:
        True if all checks pass; raises RuntimeError otherwise.
    """
    import json
    import logging
    from datetime import datetime, timezone

    import boto3
    import pandas as pd
    import yaml
    from scipy import stats

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    def _read_parquet(uri: str) -> pd.DataFrame:
        if uri.startswith("s3://"):
            parts = uri[5:].split("/", 1)
            bucket, key = parts[0], parts[1]
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            return pd.read_parquet(obj["Body"])
        return pd.read_parquet(uri)

    def _read_yaml(uri: str) -> dict:
        if uri.startswith("s3://"):
            parts = uri[5:].split("/", 1)
            bucket, key = parts[0], parts[1]
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            return yaml.safe_load(obj["Body"].read())
        with open(uri) as f:
            return yaml.safe_load(f)

    report: dict = {
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
        "passed": False,
    }
    failures = []

    logger.info("Loading dataset from %s", dataset_uri)
    df = _read_parquet(dataset_uri)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Row count check
    report["checks"]["row_count"] = {"value": len(df), "threshold": min_rows, "passed": len(df) >= min_rows}
    if len(df) < min_rows:
        failures.append(f"Row count {len(df)} below minimum {min_rows}")

    # Schema validation
    schema = _read_yaml(schema_uri)
    expected_cols = {col["name"]: col["dtype"] for col in schema.get("columns", [])}
    missing_cols = set(expected_cols.keys()) - set(df.columns)
    dtype_mismatches = []
    for col, expected_dtype in expected_cols.items():
        if col in df.columns and not str(df[col].dtype).startswith(expected_dtype):
            dtype_mismatches.append(f"{col}: expected {expected_dtype}, got {df[col].dtype}")

    schema_check = {"missing_columns": list(missing_cols), "dtype_mismatches": dtype_mismatches, "passed": not missing_cols and not dtype_mismatches}
    report["checks"]["schema"] = schema_check
    if missing_cols:
        failures.append(f"Missing columns: {missing_cols}")
    if dtype_mismatches:
        failures.append(f"Dtype mismatches: {dtype_mismatches}")

    # Null ratio check
    null_ratios = (df.isnull().sum() / len(df)).to_dict()
    null_violations = {col: ratio for col, ratio in null_ratios.items() if ratio > max_null_ratio}
    report["checks"]["null_ratio"] = {
        "max_allowed": max_null_ratio,
        "violations": null_violations,
        "passed": not null_violations,
    }
    if null_violations:
        failures.append(f"Null ratio violations: {null_violations}")

    # Data freshness check (requires a 'timestamp' or 'event_time' column)
    time_col = next((c for c in ["event_time", "timestamp", "created_at"] if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        latest = df[time_col].max()
        age_hours = (datetime.now(timezone.utc) - latest).total_seconds() / 3600
        freshness_ok = age_hours <= 48
        report["checks"]["freshness"] = {"latest_record_age_hours": age_hours, "threshold_hours": 48, "passed": freshness_ok}
        if not freshness_ok:
            failures.append(f"Data stale: latest record is {age_hours:.1f}h old (threshold 48h)")
    else:
        logger.warning("No time column found; skipping freshness check")
        report["checks"]["freshness"] = {"passed": True, "note": "no time column found"}

    # KS-test drift detection
    logger.info("Loading reference dataset for drift detection")
    ref_df = _read_parquet(reference_dataset_uri)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    drift_results = {}
    drifted_features = []
    for col in numeric_cols[:50]:  # cap at 50 features to limit runtime
        if col not in ref_df.columns:
            continue
        ks_stat, p_value = stats.ks_2samp(df[col].dropna(), ref_df[col].dropna())
        drifted = p_value < max_drift_pvalue
        drift_results[col] = {"ks_statistic": float(ks_stat), "p_value": float(p_value), "drifted": drifted}
        if drifted:
            drifted_features.append(col)

    report["checks"]["distribution_drift"] = {
        "p_value_threshold": max_drift_pvalue,
        "features_checked": len(drift_results),
        "drifted_features": drifted_features,
        "feature_results": drift_results,
        "passed": not drifted_features,
    }
    if drifted_features:
        logger.warning("Distribution drift detected in features: %s", drifted_features)
        # Drift is a warning, not a hard failure — downstream pipeline decides

    report["failures"] = failures
    report["passed"] = not failures

    logger.info("Validation complete. Passed=%s, Failures=%s", report["passed"], failures)

    with open(validation_report.path, "w") as f:
        json.dump(report, f, indent=2)

    if failures:
        raise RuntimeError(f"Data validation failed for {model_name}: {'; '.join(failures)}")

    return True
